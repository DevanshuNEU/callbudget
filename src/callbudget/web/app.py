"""FastAPI web app over the CallBudget engine: a polished showcase + JSON API.

A transport adapter, exactly like mcp/server.py and the CLI: the HTTP + browser
surface differs, the logic does not. Prediction comes from the engine; the voice
conversation comes from the one dialogue brain (voice/dialogue.py). No build step;
serves one static page. The engine is built once at startup and shared.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from ..config import REPO_ROOT, RUNS_DIR, load_dotenv
from ..mcp.engine import CallBudgetEngine

STATIC = Path(__file__).parent / "static"
CHART = REPO_ROOT / "figures" / "ecf_flywheel.png"
RESEARCH_JOURNAL = RUNS_DIR / "research_journal.jsonl"
# Same default Cartesia voice as the terminal path; override with CARTESIA_VOICE_ID.
CARTESIA_VOICE = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load .env BEFORE building the engine so the live dialogue + voice see the keys.
    load_dotenv()
    app.state.engine = CallBudgetEngine()
    yield


app = FastAPI(title="CallBudget", lifespan=lifespan)


class ConverseTurn(BaseModel):
    """One line of the call so far: who spoke and what they said."""

    role: str  # "agent" | "pharmacist"
    text: str


class ConverseBody(BaseModel):
    """Request body for /api/converse: the call so far + the drug being asked about."""

    drug: str = "Adderall XR"
    dose: str = "20 mg"
    caller: str = "Devanshu"  # who the agent discloses it is calling on behalf of
    history: list[ConverseTurn] = []


class TTSBody(BaseModel):
    """Request body for /api/tts: the agent line to speak."""

    text: str
    voice_id: str | None = None


@app.get("/api/config")
def config() -> dict:
    """Front-end feature flags. Voice (mic speech-to-text + spoken replies) is on
    only when it is not explicitly disabled AND both provider keys are present. The
    public demo runs typed-only (CALLBUDGET_VOICE=0, no keys), so no audio API is
    ever called and nothing can run up a bill. The typed conversation runs the same
    dialogue brain either way."""
    disabled = os.environ.get("CALLBUDGET_VOICE", "").strip() == "0"
    keyed = bool(os.environ.get("DEEPGRAM_API_KEY") and os.environ.get("CARTESIA_API_KEY"))
    return {"voice": (not disabled) and keyed}


@app.get("/api/predict")
def predict() -> list[dict]:
    """Pharmacies ranked by predicted P(in_stock)."""
    return app.state.engine.predict_stock()


@app.get("/api/plan")
def plan(budget: int | None = None) -> dict:
    """Call order that minimizes expected calls-to-find, plus that ECF."""
    return app.state.engine.plan_calls(budget)


@app.post("/api/call")
def call(npi: str, style: str = "clear") -> dict:
    """Place one simulated call, learn from it if confident, return what happened."""
    engine = app.state.engine
    result = engine.place_call(npi, style=style)
    learned = engine.observe(result)
    return {
        "pharmacy": engine.name_of(npi),
        "transcript": result.transcript,
        "record": result.record.model_dump(mode="json"),
        "learned": learned,
        "n_observations": engine.n_observations,
    }


@app.post("/api/converse")
def converse(body: ConverseBody) -> JSONResponse:
    """One turn of a two-way pharmacy call, delegated to the one dialogue brain
    (voice/dialogue.respond): open with the AI + recording disclosure, ask a
    clarifying question when unsure instead of abstaining, confirm when sure. Live
    Claude when ANTHROPIC_API_KEY is set, else the deterministic offline dialogue.
    This endpoint is a thin transport adapter; the logic lives in dialogue.py.
    """
    from ..voice.dialogue import respond

    history = [m.model_dump() for m in body.history]
    try:
        turn = respond(history, body.drug, body.dose, body.caller)
    except Exception as exc:  # noqa: BLE001 - clean message, never a stack
        return JSONResponse({"error": f"dialogue failed: {exc}"}, status_code=502)
    return JSONResponse(
        {
            "reply": turn.reply,
            "done": turn.done,
            "extractor": turn.source,
            "confidence": turn.confidence,
            "verdict": turn.verdict,
            "record": {"drug": body.drug, "dose": body.dose, **turn.record},
        }
    )


def _deepgram_transcribe(audio: bytes, mimetype: str) -> str:
    """Deepgram prerecorded STT over its REST API (same provider as the terminal)."""
    import urllib.request

    req = urllib.request.Request(
        "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
        data=audio,
        headers={
            "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
            "Content-Type": mimetype or "audio/webm",
        },
    )
    resp = json.load(urllib.request.urlopen(req, timeout=30))
    return resp["results"]["channels"][0]["alternatives"][0]["transcript"]


def _cartesia_tts(text: str, voice_id: str) -> bytes:
    """Cartesia neural TTS over its REST API (same provider/voice as the terminal)."""
    import urllib.request

    body = json.dumps(
        {
            "model_id": "sonic-2", "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.cartesia.ai/tts/bytes", data=body,
        headers={
            "X-API-Key": os.environ["CARTESIA_API_KEY"],
            "Cartesia-Version": "2025-04-16",
            "Content-Type": "application/json",
        },
    )
    return urllib.request.urlopen(req, timeout=30).read()


@app.post("/api/stt")
async def stt(request: Request) -> JSONResponse:
    """Transcribe a recorded mic turn with Deepgram (the same STT the terminal uses)."""
    import asyncio

    audio = await request.body()
    if not audio:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    if not os.environ.get("DEEPGRAM_API_KEY"):
        return JSONResponse(
            {"error": "speech-to-text unavailable (no DEEPGRAM_API_KEY); type instead"},
            status_code=503,
        )
    ctype = request.headers.get("content-type", "audio/webm")
    try:
        transcript = await asyncio.to_thread(_deepgram_transcribe, audio, ctype)
    except Exception as exc:  # noqa: BLE001 - clean message, never a stack
        return JSONResponse({"error": f"stt failed: {exc}"}, status_code=502)
    return JSONResponse({"transcript": transcript})


@app.post("/api/tts")
async def tts(body: TTSBody) -> Response:
    """Speak the agent's reply with Cartesia (the same neural voice the terminal uses)."""
    import asyncio

    if not os.environ.get("CARTESIA_API_KEY"):
        return JSONResponse(
            {"error": "text-to-speech unavailable (no CARTESIA_API_KEY)"}, status_code=503
        )
    voice = body.voice_id or os.environ.get("CARTESIA_VOICE_ID", CARTESIA_VOICE)
    try:
        audio = await asyncio.to_thread(_cartesia_tts, body.text, voice)
    except Exception as exc:  # noqa: BLE001 - clean message, never a stack
        return JSONResponse({"error": f"tts failed: {exc}"}, status_code=502)
    return Response(content=audio, media_type="audio/wav")


@app.get("/api/research")
def research() -> dict:
    """Summarise the EXISTING auto-research journal, if it has been generated.

    Reads runs/research_journal.jsonl (does NOT run the loop, which takes minutes)
    and returns the trials sorted by ECF, plus the safety bar if derivable. Returns
    {"available": false} when the journal is absent. Defensive throughout: a malformed
    line is skipped rather than failing the whole endpoint.
    """
    if not RESEARCH_JOURNAL.exists():
        return {"available": False}

    trials: list[dict] = []
    for line in RESEARCH_JOURNAL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        config = row.get("config") or {}
        trials.append(
            {
                "trial": row.get("trial"),
                "ecf": row.get("ecf"),
                "false_positive_in_stock": row.get("false_positive_in_stock"),
                "abstention_rate": row.get("abstention_rate"),
                "n_samples": config.get("n_samples"),
                "abstain_threshold": config.get("abstain_threshold"),
                "max_depth": config.get("max_depth"),
                "max_iter": config.get("max_iter"),
                "learning_rate": config.get("learning_rate"),
                "feasible": row.get("feasible"),
            }
        )

    if not trials:
        return {"available": False}

    # Display bar: the max false-positive among feasible configs (everything at or
    # below it passed), matching the report's "<= bar" leaderboard semantics.
    feasible_fp = [
        t["false_positive_in_stock"]
        for t in trials
        if t.get("feasible") and isinstance(t.get("false_positive_in_stock"), (int, float))
    ]
    bar = round(max(feasible_fp), 4) if feasible_fp else None

    trials_sorted = sorted(
        trials,
        key=lambda t: (t["ecf"] if isinstance(t.get("ecf"), (int, float)) else float("inf")),
    )
    n_feasible = sum(1 for t in trials if t.get("feasible"))
    return {
        "available": True,
        "n_trials": len(trials),
        "n_feasible": n_feasible,
        "n_rejected": len(trials) - n_feasible,
        "bar": bar,
        "trials": trials_sorted,
    }


@app.get("/api/eval")
def evaluate() -> dict:
    """ECF (naive vs learned) and the safety metrics."""
    return app.state.engine.run_eval()


@app.get("/chart")
def chart() -> FileResponse:
    """The ECF flywheel chart, if it has been built."""
    return FileResponse(CHART)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

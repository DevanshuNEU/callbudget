"""FastAPI web app over the CallBudget engine: a polished showcase + JSON API.

Another adapter over the engine, exactly like mcp/server.py: the transport differs
(HTTP + a browser), the logic does not. No build step; it serves one static page.
The engine is built once at startup and shared (single-user demo).
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import REPO_ROOT, RUNS_DIR
from ..mcp.engine import CallBudgetEngine

STATIC = Path(__file__).parent / "static"
CHART = REPO_ROOT / "figures" / "ecf_flywheel.png"
RESEARCH_JOURNAL = RUNS_DIR / "research_journal.jsonl"


def _load_dotenv() -> None:
    """Populate os.environ from the repo .env so /api/extract can use the live path.

    Mirrors voice/live_call._load_dotenv: .env is this app's source of truth and
    OVERRIDES the shell, so a stale ANTHROPIC_API_KEY exported in the shell cannot
    shadow the good key in .env. Best-effort: silently skip if there is no .env.
    """
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load .env BEFORE building the engine so the live extractor sees the key.
    _load_dotenv()
    app.state.engine = CallBudgetEngine()
    yield


app = FastAPI(title="CallBudget", lifespan=lifespan)


class ExtractBody(BaseModel):
    """Request body for /api/extract: the (noisy) spoken pharmacist answer."""

    transcript: str = ""


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


@app.post("/api/extract")
def extract_transcript(body: ExtractBody) -> JSONResponse:
    """Run the live self-consistency extractor on a spoken/typed pharmacist answer.

    Uses the LIVE Claude extractor when ANTHROPIC_API_KEY is set (n_samples=3 so
    self-consistency is meaningful), and falls back to the deterministic offline
    keyword extractor otherwise (n_samples=1, always works, no key). The live call
    is wrapped so a transient API error returns a clean JSON error, never a 500.
    """
    # Imported here (not at module load) so the offline path never touches the
    # vendor SDK and the page boots even without the [agent] extra installed.
    from ..voice.extract import extract, keyword_extract, make_claude_extractor

    transcript = (body.transcript or "").strip()
    if not transcript:
        return JSONResponse({"error": "empty transcript"}, status_code=400)

    use_live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_live:
        try:
            # Pass the asked drug/dose so the extractor grounds against the question
            # (a natural "we have the 20s" reads as yes, not an abstain).
            record = extract(
                transcript,
                extractor=make_claude_extractor(drug="Adderall XR", dose="20 mg"),
                n_samples=3,
            )
            extractor_name = "claude"
        except Exception as exc:  # noqa: BLE001 - surface a clean message, not a stack
            return JSONResponse(
                {"error": f"live extractor failed: {exc}", "extractor": "claude"},
                status_code=502,
            )
    else:
        record = extract(transcript, extractor=keyword_extract, n_samples=1)
        extractor_name = "keyword"

    if record.abstain:
        verdict = "re-call (not confident, no label stored)"
    else:
        verdict = f"learn (store a labeled observation: in_stock={record.in_stock.value})"

    return JSONResponse(
        {
            "record": record.model_dump(mode="json"),
            "extractor": extractor_name,
            "verdict": verdict,
        }
    )


@app.post("/api/converse")
def converse(body: ConverseBody) -> JSONResponse:
    """One turn of a two-way pharmacy call.

    The agent is the caller trying to learn whether the drug is in stock. Given the
    conversation so far it returns the next thing to say, the stock record built up
    so far, and whether the call is done. When the pharmacist's answer is ambiguous
    it asks a clarifying question instead of abstaining: in a conversation,
    uncertainty becomes a follow-up, not a dead end. Live Claude when
    ANTHROPIC_API_KEY is set, else a deterministic offline dialogue (no key).
    """
    history = [{"role": t.role, "text": t.text} for t in body.history]
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return JSONResponse(_converse_live(body.drug, body.dose, history, body.caller))
        except Exception as exc:  # noqa: BLE001 - clean message, never a stack
            return JSONResponse({"error": f"live converse failed: {exc}"}, status_code=502)
    return JSONResponse(_converse_offline(body.drug, body.dose, history, body.caller))


def _empty_record(drug: str, dose: str) -> dict:
    return {
        "drug": drug, "dose": dose, "in_stock": "unknown", "confidence": 0.0,
        "abstain": True, "brand_vs_generic": "unknown", "evidence_span": "",
    }


def _converse_offline(drug: str, dose: str, history: list[dict], caller: str = "Devanshu") -> dict:
    """Deterministic dialogue (no key): open, ask one clarifying question on an
    ambiguous answer, then confirm or give up. Used in CI and without a key."""
    from ..voice.extract import extract, keyword_extract

    said = [t for t in history if t["role"] == "pharmacist"]
    if not said:
        # Disclose up front that this is an AI agent, and that the call may be
        # recorded (bot-disclosure + recording-consent norms).
        return {"reply": (f"Hi, I'm an AI assistant calling on behalf of {caller}, and this "
                          f"call may be recorded. Do you have {drug} {dose} in stock?"),
                "record": _empty_record(drug, dose), "done": False,
                "confidence": 0.0, "extractor": "keyword"}
    rec = extract(said[-1]["text"], extractor=keyword_extract, n_samples=1)
    record = rec.model_dump(mode="json")
    if not rec.abstain:
        return {"reply": f"Got it, so that's {rec.in_stock.value}. Thank you.",
                "record": record, "done": True, "confidence": rec.confidence,
                "extractor": "keyword"}
    if len(said) < 2:  # one clarifying question, then stop to avoid looping
        return {"reply": f"Just to confirm, do you have the {drug} {dose} in stock right now?",
                "record": record, "done": False, "confidence": rec.confidence,
                "extractor": "keyword"}
    return {"reply": "No problem, I'll call back to confirm. Thanks.",
            "record": record, "done": True, "confidence": rec.confidence,
            "extractor": "keyword"}


def _converse_live(drug: str, dose: str, history: list[dict], caller: str = "Devanshu") -> dict:
    """LLM-driven dialogue: the agent asks, follows up when unsure, confirms, closes.

    Plain Anthropic (no instructor) so it works with just the [voice] extra, with the
    same robust JSON extraction + null coercion as the live answer extractor."""
    import json
    import re

    from anthropic import Anthropic

    from ..schemas import StockState

    convo = "\n".join(f"{t['role']}: {t['text']}" for t in history) or "(call just connected)"
    prompt = (
        f"You are an AI assistant calling a pharmacy on behalf of {caller} to find out whether "
        f"{drug} {dose} is in stock for a patient. Talk like a real person on a quick phone "
        "call: warm, brief, usually one short sentence, no filler, no emdashes. Reply with "
        "STRICT JSON only, "
        "keys: reply (string), in_stock (yes|no|partial|unknown), confidence (0..1), done "
        "(bool), evidence_span (verbatim quote or empty). When the call just connected (no "
        f"prior turns), your reply MUST first say you are an AI assistant for {caller} and that "
        "the call may be recorded, then ask if they have it (bot-disclosure and recording "
        "consent are required). If their answer is ambiguous or incomplete, ask ONE short "
        "clarifying question and set done=false. When confident, confirm in one sentence and "
        f"set done=true with in_stock.\n\nConversation so far:\n{convo}\n\nYour next turn as JSON:"
    )
    msg = Anthropic().messages.create(
        model="claude-sonnet-4-6", max_tokens=400, temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}

    in_stock = data.get("in_stock") or "unknown"
    if in_stock not in (s.value for s in StockState):
        in_stock = "unknown"
    confidence = round(float(data.get("confidence") or 0.0), 3)
    reply = data.get("reply") or f"Sorry, do you have {drug} {dose} in stock?"
    reply = re.sub(r"\s*[—–]\s*", ", ", reply)  # strip em/en dashes (models love them)
    record = {
        "drug": drug, "dose": dose, "in_stock": in_stock, "confidence": confidence,
        "abstain": in_stock == "unknown", "brand_vs_generic": "unknown",
        "evidence_span": data.get("evidence_span") or "",
    }
    return {"reply": reply, "record": record, "done": bool(data.get("done")),
            "confidence": confidence, "extractor": "claude"}


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

    # The safety bar is the smallest false-positive among REJECTED configs minus a
    # hair, but the journal does not store the bar directly. Derive a usable display
    # bar as the max false-positive among feasible configs (everything at or below it
    # passed), which matches the report's "<= bar" semantics for the leaderboard.
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

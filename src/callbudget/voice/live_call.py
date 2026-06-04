"""The LIVE voice call: a real Pipecat pipeline (Deepgram STT -> Claude -> Cartesia TTS).

  ┌──────────────────────────────────────────────────────────────────────┐
  │ RUN LOCALLY, NOT IN CI. The import paths here are verified against     │
  │ Pipecat 1.3.0; what CI cannot exercise is the audio itself (it needs a │
  │ mic + speaker + the portaudio system lib + API keys). The extraction   │
  │ this calls is the SAME self-consistency + abstention path that the     │
  │ offline tests cover (voice/extract.py); only the audio transport is    │
  │ live. If your Pipecat version differs, adjust the import paths.         │
  └──────────────────────────────────────────────────────────────────────┘

What it does: the agent asks "do you have <drug> <dose> in stock?", you answer on
the mic as the pharmacist, and CallBudget runs the live self-consistency extractor
on your spoken answer, prints the structured stock record, and decides whether to
LEARN from it (confident) or RE-CALL (abstain), then speaks that verdict back. This
is STT -> LLM -> TTS with the LLM doing the structured extraction (the valuable
part), not chit-chat.

Run locally:
    brew install portaudio          # the mic/speaker system lib
    uv sync --extra voice
    export DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... ANTHROPIC_API_KEY=...
    uv run python -m callbudget.voice.live_call
"""

from __future__ import annotations

from datetime import UTC


async def run_live_call(
    drug: str = "Adderall XR", dose: str = "20 mg", n_samples: int = 3, caller: str = "Devanshu"
) -> None:
    """Build and run the call -> extract -> abstain loop over the local mic/speaker."""
    import asyncio

    # Quiet Pipecat's DEBUG/INFO flood (loguru) so the extraction output is readable.
    import sys

    from loguru import logger

    # Imports are local so the module imports fine without the [voice] extra.
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    from .extract import extract, make_claude_extractor

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    # Give the extractor the question it is grounding against, so a natural answer
    # like "yeah we have the 20s" reads as a clear yes instead of abstaining.
    extractor = make_claude_extractor(drug=drug, dose=dose)

    class StockCapture(FrameProcessor):
        """Watches the pharmacist's final transcript, runs CallBudget's extraction
        with calibrated abstention, prints the structured record + the learn/re-call
        decision, and speaks the verdict. This is the bridge from raw speech to the
        same ExtractionRecord the offline path produces."""

        def __init__(self) -> None:
            super().__init__()
            self._busy = False

        async def process_frame(self, frame, direction) -> None:
            await super().process_frame(frame, direction)
            text = getattr(frame, "text", "") or ""
            if isinstance(frame, TranscriptionFrame) and text.strip() and not self._busy:
                self._busy = True
                transcript = text.strip()
                # extract() makes blocking API calls n_samples times; run it off the
                # event loop so the audio pipeline is not stalled.
                rec = await asyncio.to_thread(
                    extract, transcript, extractor=extractor, n_samples=n_samples
                )
                print(f'\n  pharmacist: "{transcript}"')
                print(
                    f"  -> in_stock={rec.in_stock.value}  confidence={rec.confidence}  "
                    f"brand_vs_generic={rec.brand_vs_generic}  abstain={rec.abstain}"
                )
                if rec.abstain:
                    print("     VERDICT: not confident -> RE-CALL, no label stored "
                          "(safety over guessing)")
                    verdict = "I'm not totally sure, let me call back to confirm."
                else:
                    print(f"     VERDICT: confident -> would LEARN (store a labeled "
                          f"observation: in_stock={rec.in_stock.value})")
                    verdict = f"Got it, so that's {rec.in_stock.value}. Thank you."
                _log_turn(drug, dose, transcript, rec, verdict)
                await self.push_frame(TTSSpeakFrame(verdict))
                self._busy = False
            await self.push_frame(frame, direction)

    # PRODUCTION TELEPHONY: this LocalAudioTransport is the ONE line that changes to
    # place real outbound calls to a pharmacy's phone number. Ports-and-adapters: the
    # STT -> extract -> TTS chain below is identical; only the transport swaps. For
    # real PSTN calls, replace it with a Pipecat telephony transport (the Twilio/Daily
    # websocket transport fed by a Twilio Media Streams dial-out, or a SIP transport)
    # and supply the dialed number + provider credentials. Not enabled here on purpose:
    # real calls cost money, need a provisioned number, and require recording consent.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True,
                                  vad_analyzer=SileroVADAnalyzer())
    )
    import os

    stt = DeepgramSTTService(api_key=_env("DEEPGRAM_API_KEY"))
    # Cartesia requires a voice_id. Default to a real public voice; override with
    # CARTESIA_VOICE_ID (any id from GET https://api.cartesia.ai/voices on your key).
    voice_id = os.environ.get("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")
    tts = CartesiaTTSService(api_key=_env("CARTESIA_API_KEY"), voice_id=voice_id)

    # STT -> (extract + abstain) -> TTS, over the local mic/speaker.
    pipeline = Pipeline([transport.input(), stt, StockCapture(), tts, transport.output()])
    task = PipelineTask(pipeline)

    print(
        f"\nCallBudget live call. The agent is asking about {drug} {dose}.\n"
        "Speak as the pharmacist, e.g. \"yeah we have the 20s in stock\", "
        "\"we're out of the XR\", or \"uh let me check the back\".\n"
        "Watch the structured record + the learn/re-call decision. Ctrl+C to quit.\n"
    )
    # Best-effort spoken opener; harmless if the version processes it differently.
    # Discloses it is an AI agent up front (bot-disclosure + consent norms).
    opener = (
        f"Hi, I'm an AI assistant calling on behalf of {caller}, and this call may be recorded. "
        f"Do you have {drug} {dose} in stock?"
    )
    try:
        await task.queue_frame(TTSSpeakFrame(opener))
    except Exception:
        pass
    await PipelineRunner().run(task)


def _log_turn(drug: str, dose: str, transcript: str, record, verdict: str) -> None:
    """Append one call turn to a JSONL transcript log: a record of what was actually
    said and extracted, and the raw material for growing the extraction test corpus
    (promote real lines from here into tests/data/extraction_corpus.json)."""
    import json
    from datetime import datetime

    from ..config import RUNS_DIR

    path = RUNS_DIR / "voice_transcripts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(UTC).isoformat(),
        "drug": drug,
        "dose": dose,
        "transcript": transcript,
        "record": record.model_dump(mode="json"),
        "verdict": verdict,
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name} (see the module docstring)")
    return value


def _load_dotenv() -> None:
    """Populate os.environ from the repo .env so the local run just works.

    .env is this app's source of truth and OVERRIDES the shell: a stale
    ANTHROPIC_API_KEY exported in the shell would otherwise shadow the good key
    you put in .env (which is exactly what bit the first run)."""
    import os

    from ..config import REPO_ROOT

    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


if __name__ == "__main__":
    import asyncio

    _load_dotenv()
    asyncio.run(run_live_call())

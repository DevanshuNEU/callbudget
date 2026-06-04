"""The LIVE voice call: a real Pipecat pipeline (Deepgram STT -> Claude -> Cartesia TTS).

  ┌──────────────────────────────────────────────────────────────────────┐
  │ RUN LOCALLY, NOT IN CI. The import paths here are verified against     │
  │ Pipecat 1.3.0; what CI cannot exercise is the audio itself (it needs a │
  │ mic + speaker + the portaudio system lib + API keys). The extraction   │
  │ this calls is the SAME self-consistency + abstention path that the     │
  │ offline tests cover (voice/extract.py); only the audio transport is    │
  │ live. If your Pipecat version differs, adjust the import paths.         │
  └──────────────────────────────────────────────────────────────────────┘

What it does: a two-way call. The agent opens (disclosing it is an AI + recording
consent), you answer on the mic as the pharmacist, and it drives the SAME dialogue
brain as the web and CLI (voice/dialogue.respond): asks a clarifying question when
unsure, confirms when sure, prints the structured record, and speaks its reply back.
STT -> dialogue brain (Claude) -> TTS; only the transport is local here.

Run locally:
    brew install portaudio          # the mic/speaker system lib
    uv sync --extra voice
    export DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... ANTHROPIC_API_KEY=...
    uv run python -m callbudget.voice.live_call
"""

from __future__ import annotations

from datetime import UTC

from ..config import load_dotenv


async def run_live_call(
    drug: str = "Adderall XR", dose: str = "20 mg", caller: str = "Devanshu"
) -> None:
    """Run the two-way call over the local mic/speaker, driving voice/dialogue.respond."""
    import asyncio

    # Quiet Pipecat's DEBUG/INFO flood (loguru) so the conversation output is readable.
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

    from .dialogue import respond

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    class StockCapture(FrameProcessor):
        """Bridges raw speech to the shared dialogue brain: on each final pharmacist
        transcript it appends to the conversation, calls voice/dialogue.respond, prints
        the record + verdict, logs the turn, and speaks the agent's reply. Same brain
        as the web and CLI; this class is only the audio adapter."""

        def __init__(self, history: list[dict]) -> None:
            super().__init__()
            self._history = history
            self._busy = False

        async def process_frame(self, frame, direction) -> None:
            await super().process_frame(frame, direction)
            text = getattr(frame, "text", "") or ""
            if isinstance(frame, TranscriptionFrame) and text.strip() and not self._busy:
                self._busy = True
                transcript = text.strip()
                self._history.append({"role": "pharmacist", "text": transcript})
                # respond() may call the LLM; run it off the event loop so audio is not stalled.
                turn = await asyncio.to_thread(respond, self._history, drug, dose, caller)
                self._history.append({"role": "agent", "text": turn.reply})
                print(f'\n  pharmacist: "{transcript}"')
                print(f"  agent: {turn.reply}")
                if turn.done:
                    print(f"  -> {turn.verdict} (in_stock={turn.in_stock.value}, "
                          f"confidence={turn.confidence})")
                _log_turn(drug, dose, transcript, turn)
                await self.push_frame(TTSSpeakFrame(turn.reply))
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

    # The opener comes from the same dialogue brain (AI disclosure + recording consent).
    opener = respond([], drug, dose, caller)
    history = [{"role": "agent", "text": opener.reply}]

    # STT -> dialogue brain -> TTS, over the local mic/speaker.
    pipeline = Pipeline([transport.input(), stt, StockCapture(history), tts, transport.output()])
    task = PipelineTask(pipeline)

    print(
        f"\nCallBudget live call. The agent is calling about {drug} {dose}.\n"
        "Speak as the pharmacist, e.g. \"yeah we have the 20s\", \"we're out of the XR\", "
        "or \"uh let me check the back\".\n"
        "It converses, asks a follow-up when unsure, prints the record when done. Ctrl+C to quit.\n"
    )
    # Best-effort spoken opener; harmless if the version processes it differently.
    try:
        await task.queue_frame(TTSSpeakFrame(opener.reply))
    except Exception:
        pass
    await PipelineRunner().run(task)


def _log_turn(drug: str, dose: str, transcript: str, turn) -> None:
    """Append one call turn to a JSONL transcript log: what was said and what the brain
    returned, the raw material for growing tests/data/extraction_corpus.json."""
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
        "record": turn.record,
        "reply": turn.reply,
        "done": turn.done,
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name} (see the module docstring)")
    return value


if __name__ == "__main__":
    import asyncio

    load_dotenv()
    asyncio.run(run_live_call())

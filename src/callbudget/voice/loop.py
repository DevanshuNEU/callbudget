"""The call loop. Two paths:

1. `simulate_text_call` (this file, RUNNABLE OFFLINE, no keys/audio): wires
   SimPharmacist -> extract -> CallResult. This exercises the full call->record
   logic (the part that feeds the prediction model) with zero dependencies.

2. The LIVE audio loop (documented below, runs locally with the [voice] extra +
   keys + a mic): Pipecat wiring Deepgram STT -> Claude LLM -> Cartesia TTS.

THEORY (the cascade pipeline + latency budget):
audio --[Deepgram STT]--> text --[Claude]--> text --[Cartesia TTS]--> audio.
Pipecat owns voice-activity detection, turn-taking, and barge-in. Each hop adds
latency; the sum must stay conversational (~sub-second), so a voice agent is
largely a latency budget managed across a pipeline under a real-time deadline.
We chose cascade over speech-to-speech for the debuggable text intermediate.

LIVE RUN (verify on your machine):
    uv sync --extra voice
    export DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... ANTHROPIC_API_KEY=...
    # Build a Pipecat pipeline:
    #   transport(mic/speaker) -> DeepgramSTTService -> context aggregator
    #     -> Anthropic LLM (system prompt = "you are calling a pharmacy to ask if
    #        <drug> <dose> is in stock; when you get an answer, call the
    #        record_stock tool") -> CartesiaTTSService -> transport.out
    # The SimPharmacist can be the other end (a second Pipecat agent) or a human.
    # On each final answer, call voice.extract.extract(transcript, make_claude_extractor()).
This audio path is not exercised in CI (needs audio devices); verify locally.
"""

from __future__ import annotations

from ..schemas import CallResult, DrugRequest, ExtractionRecord
from ..sim.pharmacist import SimPharmacist
from .extract import Extractor, extract, keyword_extract


def simulate_text_call(
    pharmacist: SimPharmacist,
    request: DrugRequest,
    extractor: Extractor = keyword_extract,
    n_samples: int = 3,
    abstain_threshold: float = 0.67,
) -> CallResult:
    """Offline, no-audio simulation of one call: ask -> messy answer -> extract.

    Returns a CallResult whose ExtractionRecord is exactly what the LIVE audio
    loop would produce after STT, so downstream code (turning a call into a
    labeled StockObservation) is identical for both paths. `n_samples` and
    `abstain_threshold` are the self-consistency knobs the auto-research loop
    tunes; the defaults match extract()'s own defaults, so existing callers are
    unchanged.
    """
    question = f"Hi, do you have {request.drug_name} {request.dose} in stock?"
    transcript = pharmacist.answer(question)
    record: ExtractionRecord = extract(
        transcript, extractor=extractor, n_samples=n_samples, abstain_threshold=abstain_threshold
    )
    return CallResult(
        pharmacy_npi=pharmacist.pharmacy_npi,
        request=request,
        record=record,
        transcript=transcript,
    )

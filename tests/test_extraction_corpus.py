"""Regression corpus for extraction + abstention.

THEORY (a transcript corpus is the test that matters for a voice agent):
The thing most likely to silently regress is "did we read the spoken answer right,
and did we abstain when we should have." So we pin a corpus of representative
pharmacist answers (the kind STT produces) to their expected (in_stock, abstain),
and assert against it. The OFFLINE keyword extractor is deterministic, so this runs
in CI with no key. The live LLM extractor is exercised by a separate keyed test.

Grow this corpus by promoting real lines out of runs/voice_transcripts.jsonl (which
the live call logs) into tests/data/extraction_corpus.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from callbudget.schemas import StockState
from callbudget.voice.extract import extract, keyword_extract, make_claude_extractor

CORPUS = json.loads((Path(__file__).parent / "data" / "extraction_corpus.json").read_text())


@pytest.mark.parametrize("case", CORPUS, ids=[c["transcript"][:28] for c in CORPUS])
def test_offline_extractor_matches_corpus(case: dict) -> None:
    """The deterministic offline extractor must reproduce the corpus exactly."""
    rec = extract(case["transcript"], extractor=keyword_extract, n_samples=1)
    assert rec.abstain is case["abstain"], (case["transcript"], rec.in_stock, rec.abstain)
    assert rec.in_stock == StockState(case["in_stock"]), case["transcript"]


def test_live_extractor_with_context_does_not_overabstain() -> None:
    """Keyed live test: with the question context, natural answers that would
    abstain out of context should read as clear yes/no. Skipped without a key."""
    if not os.environ.get("CALLBUDGET_LIVE_TESTS"):
        pytest.skip("set CALLBUDGET_LIVE_TESTS=1 to run live API tests")
    pytest.importorskip("anthropic")  # the SDK extractor needs the [voice]/[agent] extra

    from callbudget.voice.live_call import _load_dotenv

    _load_dotenv()  # prefer the valid .env key over a stale one in the shell
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("needs ANTHROPIC_API_KEY (live extractor)")

    extractor = make_claude_extractor(drug="Adderall XR", dose="20 mg")
    yes = extract("yeah we have the 20s", extractor=extractor, n_samples=3)
    assert not yes.abstain and yes.in_stock in (StockState.YES, StockState.PARTIAL)
    no = extract("nah we're all out right now", extractor=extractor, n_samples=3)
    assert not no.abstain and no.in_stock == StockState.NO

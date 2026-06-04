"""Tests for the one dialogue brain (voice/dialogue.respond), the text-level engine
every voice surface drives. The offline path is deterministic and CI-safe; a keyed
test exercises the live path. We delenv ANTHROPIC_API_KEY for the offline tests so a
stale key in the shell cannot push them onto the live path.
"""

from __future__ import annotations

import os

import pytest

from callbudget.schemas import StockState
from callbudget.voice.dialogue import Turn, respond


def test_offline_opener_discloses_ai_and_recording(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = respond([])
    assert isinstance(t, Turn) and t.done is False
    low = t.reply.lower()
    assert "ai assistant" in low and "record" in low and "in stock" in low


def test_offline_confirms_on_clear_answer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = respond([{"role": "pharmacist", "text": "yeah we have the 20s in stock"}])
    assert t.done is True and t.in_stock == StockState.YES and t.abstain is False


def test_offline_asks_followup_when_unsure(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = respond([{"role": "pharmacist", "text": "uh let me check the back"}])
    # uncertainty becomes a follow-up question, not a dead-end abstain
    assert t.done is False and "?" in t.reply


def test_record_and_verdict_shapes(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = respond([{"role": "pharmacist", "text": "no we don't have the 20 xr"}])
    assert t.record["in_stock"] == "no" and "learn" in t.verdict


def test_live_dialogue_when_keyed():
    """Keyed: the live brain opens with the disclosure and confirms a clear answer."""
    if not os.environ.get("CALLBUDGET_LIVE_TESTS"):
        pytest.skip("set CALLBUDGET_LIVE_TESTS=1 to run live API tests")
    from callbudget.voice.live_call import _load_dotenv

    _load_dotenv()  # prefer the valid .env key over a stale one in the shell
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("needs ANTHROPIC_API_KEY (live dialogue)")

    opener = respond([])
    assert opener.source == "claude" and not opener.done
    assert "ai assistant" in opener.reply.lower()
    clear = respond(
        [
            {"role": "agent", "text": opener.reply},
            {"role": "pharmacist", "text": "yes we have the adderall xr 20 milligram in stock"},
        ]
    )
    assert clear.done and clear.in_stock == StockState.YES

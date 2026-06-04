"""Voice-half logic tests, all offline (no API keys, no audio).

Covers the two things that matter for reliability: the extractor turns messy
answers into the right structured stock state, and it ABSTAINS rather than
guessing when the answer is ambiguous or the parses disagree.
"""

from __future__ import annotations

from callbudget.schemas import DrugRequest, StockState
from callbudget.sim.pharmacist import SimPharmacist
from callbudget.voice.extract import extract, keyword_extract
from callbudget.voice.loop import simulate_text_call

REQUEST = DrugRequest(drug_name="Adderall XR", dose="20 mg", location_zip="02114")


def test_clear_in_stock_is_yes_and_confident():
    rec = extract("yeah we got the 20 XR in stock, come grab it")
    assert rec.in_stock == StockState.YES
    assert rec.abstain is False
    assert rec.confidence == 1.0  # deterministic extractor -> full agreement


def test_clear_out_is_no_not_a_false_positive():
    rec = extract("no we're out of the XR, been out for two weeks")
    assert rec.in_stock == StockState.NO
    assert rec.abstain is False


def test_generic_in_brand_backorder_is_partial():
    rec = extract("we got the generic 10s but the 20 XR is on backorder til next week")
    assert rec.in_stock == StockState.PARTIAL
    assert rec.brand_vs_generic == "generic"


def test_hedge_triggers_abstention():
    rec = extract("uh let me check in the back, i'm not sure we have the 20s")
    assert rec.abstain is True
    assert rec.in_stock == StockState.UNKNOWN


def test_self_consistency_abstains_on_disagreement():
    # A flaky extractor whose parses disagree across runs -> low agreement -> abstain.
    values = iter(["yes", "no", "partial"])

    def flaky(_transcript: str) -> dict:
        return {
            "in_stock": next(values),
            "drug": "Adderall XR",
            "dose": "20 mg",
            "brand_vs_generic": "unknown",
            "abstain": False,
            "evidence_span": "something",
        }

    rec = extract("ambiguous", extractor=flaky, n_samples=3)
    assert rec.abstain is True
    assert rec.confidence < 0.67


def test_keyword_extract_fields():
    out = keyword_extract("yep we have the Adderall XR 20s on the shelf")
    assert out["in_stock"] == "yes"
    assert out["evidence_span"]  # non-empty quote required


def test_simulate_text_call_matches_ground_truth():
    # In-stock pharmacy -> record reads as available (yes or partial), no abstain.
    ph = SimPharmacist("1000000001", truly_in_stock=True, style="clear", seed=1)
    res = simulate_text_call(ph, REQUEST)
    assert res.record.in_stock in (StockState.YES, StockState.PARTIAL)
    assert res.record.abstain is False
    assert res.transcript

    # Hedging pharmacy -> abstain, not a guess.
    ph2 = SimPharmacist("1000000002", truly_in_stock=False, style="hedge", seed=1)
    res2 = simulate_text_call(ph2, REQUEST)
    assert res2.record.abstain is True

"""Web API tests via FastAPI's TestClient (no running server, no browser needed).
The TestClient context manager runs the lifespan, which builds the engine once.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from callbudget.schemas import StockState
from callbudget.web.app import _converse_offline, app


def test_converse_offline_dialogue():
    """The deterministic dialogue (no key): open, clarify on ambiguity, confirm on clear."""
    opener = _converse_offline("Adderall XR", "20 mg", [])
    assert opener["done"] is False and "in stock" in opener["reply"].lower()
    low = opener["reply"].lower()
    assert "ai assistant" in low and "record" in low  # bot-disclosure + recording consent

    clear = _converse_offline(
        "Adderall XR", "20 mg", [{"role": "pharmacist", "text": "yeah we have the 20s in stock"}]
    )
    assert clear["done"] is True and clear["record"]["in_stock"] == StockState.YES.value

    ambiguous = _converse_offline(
        "Adderall XR", "20 mg", [{"role": "pharmacist", "text": "uh let me check the back"}]
    )
    # asks a follow-up question instead of dead-ending in abstain
    assert ambiguous["done"] is False and "?" in ambiguous["reply"]


def test_converse_endpoint_shape():
    """The endpoint returns a usable turn whether it runs live or offline."""
    with TestClient(app) as client:
        turn = client.post("/api/converse", json={"history": []}).json()
        assert turn.get("reply") and "record" in turn and "done" in turn


def test_web_api():
    with TestClient(app) as client:
        assert "CallBudget" in client.get("/").text

        rows = client.get("/api/predict").json()
        assert len(rows) > 0
        probs = [r["p_in_stock"] for r in rows]
        assert probs == sorted(probs, reverse=True)  # ranked, highest first

        ev = client.get("/api/eval").json()
        assert "ecf_learned" in ev and "false_positive_in_stock" in ev

        called = client.post(f"/api/call?npi={rows[0]['npi']}&style=clear").json()
        assert called["transcript"]
        assert "in_stock" in called["record"]

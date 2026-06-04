"""Web API tests via FastAPI's TestClient (no running server, no browser needed).
The TestClient context manager runs the lifespan, which builds the engine once.
The dialogue logic itself is tested in test_dialogue.py; here we test the adapter.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from callbudget.web.app import app


def test_converse_endpoint_shape():
    """The /api/converse adapter returns a usable turn whether it runs live or offline."""
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

"""Smoke test: the foundation powers on.

Checks (1) every package imports, (2) the shared schemas validate and round-trip
through JSON, (3) the one real piece of math (expected_calls_to_find) matches
values we can compute by hand, (4) the greedy order sorts correctly, and (5) the
content-addressed run_id is stable and sensitive to config changes.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from callbudget import config, schemas
from callbudget.policy.budget import expected_calls_to_find, order_calls


def test_all_packages_import():
    # If any module has a syntax/import error, importing the package tree fails.
    import callbudget.data.loaders  # noqa: F401
    import callbudget.evals.grade  # noqa: F401
    import callbudget.mcp.server  # noqa: F401
    import callbudget.predict.model  # noqa: F401
    import callbudget.sim.pharmacist  # noqa: F401
    import callbudget.voice.extract  # noqa: F401


def test_stock_observation_roundtrips():
    obs = schemas.StockObservation(
        pharmacy_npi="1234567890",
        rxcui="1234",
        drug_name="Adderall XR",
        dose="20 mg",
        observed_at=datetime(2026, 6, 1, 12, 0, 0),
        in_stock=schemas.StockState.PARTIAL,
        confidence=0.8,
    )
    reloaded = schemas.StockObservation.model_validate_json(obs.model_dump_json())
    assert reloaded == obs


def test_confidence_is_bounded():
    # Validation at the boundary: an out-of-range confidence must fail loudly.
    with pytest.raises(ValidationError):
        schemas.StockObservation(
            pharmacy_npi="1",
            rxcui=None,
            drug_name="x",
            dose="y",
            observed_at=datetime(2026, 6, 1),
            in_stock=schemas.StockState.YES,
            confidence=1.7,
        )


def test_expected_calls_to_find_matches_hand_math():
    # Certain first hit -> exactly 1 call.
    assert expected_calls_to_find([1.0]) == pytest.approx(1.0)
    # Two coin-flips: 1 + (1-0.5) = 1.5.
    assert expected_calls_to_find([0.5, 0.5]) == pytest.approx(1.5)
    # First always misses, second always hits -> exactly 2 calls.
    assert expected_calls_to_find([0.0, 1.0]) == pytest.approx(2.0)


def test_greedy_orders_by_descending_probability():
    assert order_calls(["a", "b", "c"], [0.1, 0.9, 0.5]) == ["b", "c", "a"]


def test_run_id_is_stable_and_sensitive():
    s = config.Settings()
    rid = s.run_id()
    assert len(rid) == 12
    assert s.run_id() == rid  # deterministic
    assert config.Settings(sim_seed=999).run_id() != rid  # changes with config

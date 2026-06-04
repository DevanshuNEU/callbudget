"""Reliability ablation tests: self-consistency must not increase false positives,
and it trades coverage (more abstention) for safety. Offline, deterministic."""

from __future__ import annotations

from callbudget.data.identity import resolve
from callbudget.data.loaders import (
    load_arcos_throughput,
    load_boston_pharmacies,
    load_shortage_calendar,
)
from callbudget.evals.grade import reliability_panel
from callbudget.sim.callgen import generate_world


def _world():
    pharmacies = resolve(load_boston_pharmacies(limit=20))
    load_arcos_throughput(pharmacies)
    world = generate_world(
        history_per_pharmacy=10, seed=7, pharmacies=pharmacies,
        shortage_dates=load_shortage_calendar(),
    )
    return pharmacies, world


def test_self_consistency_reduces_false_positives():
    pharmacies, world = _world()
    single = reliability_panel(pharmacies, world.true_prob_today, n_samples=1, trials=15, seed=7)
    multi = reliability_panel(pharmacies, world.true_prob_today, n_samples=3, trials=15, seed=7)
    assert multi["false_positive_in_stock"] <= single["false_positive_in_stock"]
    assert multi["abstention_rate"] >= single["abstention_rate"]  # safety/coverage tradeoff


def test_panel_metrics_are_bounded():
    pharmacies, world = _world()
    m = reliability_panel(pharmacies, world.true_prob_today, n_samples=3, trials=10)
    for key in ("false_positive_in_stock", "abstention_rate", "decided_accuracy"):
        assert 0.0 <= m[key] <= 1.0

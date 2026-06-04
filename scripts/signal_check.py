"""Prediction signal check: does the learned model beat a naive baseline at
minimizing expected calls-to-find, and does it improve with data?

The existential test from FOUNDATION F1. Runs end to end on pure synthetic data
(no API keys, no audio), sweeping history depth to show the flywheel: more call
history -> lower ECF for the learned policy, while the random baseline stays flat.

Run:  uv run python scripts/signal_check.py
"""

from __future__ import annotations

import numpy as np

from callbudget.policy.budget import expected_calls_to_find, order_calls
from callbudget.predict.features import build_request_features, build_training_matrix
from callbudget.predict.model import StockModel
from callbudget.sim.callgen import generate_world


def _ecf(ordered_npis: list[str], true_prob: dict[str, float]) -> float:
    """ECF of a given call order, scored against the HIDDEN true probabilities."""
    return expected_calls_to_find([true_prob[n] for n in ordered_npis])


def _baseline_ecf(npis, true_prob, shuffles: int = 300, seed: int = 0) -> float:
    """Average ECF of a random call order (the naive baseline)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(shuffles):
        order = list(npis)
        rng.shuffle(order)
        out.append(_ecf(order, true_prob))
    return float(np.mean(out))


def run(history_levels=(1, 3, 7, 15)) -> None:
    print("Scenario: Adderall XR 20mg, Boston, 60 pharmacies, drug in shortage.\n")
    print(f"{'hist/pharm':>10} | {'ECF base':>9} | {'ECF learned':>11} | {'saved':>7}")
    print("-" * 46)
    last_model = None
    for h in history_levels:
        world = generate_world(history_per_pharmacy=h, seed=7)
        X, y = build_training_matrix(world.observations, world.pharmacies, world.shortage_dates)
        model = StockModel(seed=7).fit(X, y)
        npis, Xr = build_request_features(
            world.pharmacies, world.observations, world.as_of, world.shortage_dates
        )
        probs = model.predict_proba(Xr)
        ecf_learned = _ecf(order_calls(npis, probs), world.true_prob_today)
        ecf_base = _baseline_ecf(npis, world.true_prob_today)
        print(f"{h:>10} | {ecf_base:>9.2f} | {ecf_learned:>11.2f} | {ecf_base - ecf_learned:>7.2f}")
        last_model = model

    print("\nFeature importances (most-data model):")
    for k, v in sorted(last_model.explain().items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:>18}: {v:.3f}")


if __name__ == "__main__":
    run()

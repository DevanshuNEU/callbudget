"""The reliability ablation.

Question: given a deliberately unreliable extractor (25% chance of a confident
misread), does self-consistency (reading n_samples times and abstaining on
disagreement) convert misreads into safe abstentions instead of false positives?

Run:  uv run python scripts/reliability_ablation.py
"""

from __future__ import annotations

from callbudget.data.identity import resolve
from callbudget.data.loaders import (
    load_arcos_throughput,
    load_boston_pharmacies,
    load_shortage_calendar,
)
from callbudget.evals.grade import reliability_panel
from callbudget.sim.callgen import generate_world


def run() -> None:
    pharmacies = resolve(load_boston_pharmacies(limit=28))
    load_arcos_throughput(pharmacies)
    world = generate_world(
        history_per_pharmacy=15, seed=7, pharmacies=pharmacies,
        shortage_dates=load_shortage_calendar(),
    )

    print("Unreliable extractor (25% misread). Does self-consistency keep it safe?\n")
    print(f"{'n':>3} | {'false-pos in-stock':>18} | {'abstention':>10} | {'decided acc':>11}")
    print("-" * 52)
    for n in (1, 3, 5):
        m = reliability_panel(pharmacies, world.true_prob_today, n_samples=n, trials=20)
        print(
            f"{n:>3} | {m['false_positive_in_stock']:>18.3f} | "
            f"{m['abstention_rate']:>10.3f} | {m['decided_accuracy']:>11.3f}"
        )


if __name__ == "__main__":
    run()

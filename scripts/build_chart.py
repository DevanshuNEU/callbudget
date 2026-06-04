"""Build the hero ECF chart on real Boston scaffolding.

Real: pharmacies (NPPES, cached), drug RxCUI (RxNorm 861226), throughput shape
(ARCOS opioid distribution), shortage window (FDA/ASHP Adderall dates).
Synthetic: the in-stock outcomes (that data does not exist anywhere).

Run:  uv run python scripts/build_chart.py
Outputs: figures/ecf_flywheel.png  +  runs/ecf_curve.json
"""

from __future__ import annotations

import json
from pathlib import Path

from callbudget.data.identity import resolve
from callbudget.data.loaders import (
    load_arcos_throughput,
    load_boston_pharmacies,
    load_drug,
    load_shortage_calendar,
)
from callbudget.evals.grade import ecf_curve, render_ecf_chart

REPO = Path(__file__).resolve().parents[1]
CAPTION = (
    "Real Boston pharmacies (NPPES) + RxNorm RxCUI 861226 + ARCOS-shaped throughput + real "
    "Adderall shortage window. In-stock outcomes are SIMULATED (that data does not exist). "
    "ECF averaged over 5 runs."
)


def run() -> None:
    # Same limit as the engine and the CLI chart (demo.flywheel_chart) so all
    # three render the identical chart and report the same resolved pharmacy count.
    pharmacies = resolve(load_boston_pharmacies(limit=28))
    load_arcos_throughput(pharmacies)
    drug = load_drug()
    shortage = load_shortage_calendar()

    levels = [1, 3, 7, 15, 30]
    lv, baseline, learned = ecf_curve(levels, pharmacies, shortage)

    print(f"Scenario: {drug.drug_name} {drug.dose} (rxcui {drug.rxcui}), "
          f"{len(pharmacies)} real Boston pharmacies.\n")
    print(f"{'hist/pharm':>10} | {'ECF base':>9} | {'ECF learned':>11} | {'saved':>7}")
    print("-" * 46)
    for h, b, le in zip(lv, baseline, learned, strict=True):
        print(f"{h:>10} | {b:>9.2f} | {le:>11.2f} | {b - le:>7.2f}")

    png = render_ecf_chart(REPO / "figures" / "ecf_flywheel.png", lv, baseline, learned,
                           len(pharmacies), caption=CAPTION)

    out = REPO / "runs" / "ecf_curve.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"levels": lv, "baseline": baseline, "learned": learned,
         "n_pharmacies": len(pharmacies), "rxcui": drug.rxcui}, indent=2))
    print(f"\nchart -> {png}\ndata  -> {out}")


if __name__ == "__main__":
    run()

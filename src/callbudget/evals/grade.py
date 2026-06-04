"""Grading harness: the metrics + the hero chart.

THEORY (cost metrics over accuracy, and the safety metric):
- EXPECTED CALLS-TO-FIND (ECF): the hero. Plot ECF vs accumulated call history,
  learned vs naive baseline. The downward curve is the flywheel. The ECF math
  lives in policy/budget.py; here we average it over several random
  instantiations of the idiosyncratic stock component to get a stable estimate.
- FALSE-POSITIVE-IN-STOCK rate = P(said in_stock=yes | truth=no), on
  non-abstained calls. The SAFETY metric: a wrong "yes" sends a patient to an
  empty pharmacy. Led with, not raw accuracy, because the two error types have
  wildly asymmetric cost.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..policy.budget import expected_calls_to_find, order_calls  # noqa: E402
from ..predict.features import build_request_features, build_training_matrix  # noqa: E402
from ..predict.model import StockModel  # noqa: E402
from ..schemas import ExtractionRecord, StockState  # noqa: E402
from ..sim.callgen import generate_world  # noqa: E402
from ..sim.pharmacist import SimPharmacist  # noqa: E402
from ..voice.extract import extract, keyword_extract  # noqa: E402


def _order_ecf(order: list[str], true_prob: dict[str, float]) -> float:
    return expected_calls_to_find([true_prob[n] for n in order])


def _baseline_ecf(npis, true_prob, shuffles: int = 400, seed: int = 0) -> float:
    """Average ECF of a random (naive) call order."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(shuffles):
        order = list(npis)
        rng.shuffle(order)
        vals.append(_order_ecf(order, true_prob))
    return float(np.mean(vals))


def ecf_curve(history_levels, pharmacies, shortage_dates, seeds=(1, 2, 3, 4, 5)):
    """ECF for learned vs baseline at each history depth, averaged over seeds.

    Each seed is one random instantiation of the hidden idiosyncratic stock
    component; averaging smooths that Monte-Carlo noise. Returns (levels, baseline, learned).
    """
    learned_out, baseline_out = [], []
    for h in history_levels:
        ls, bs = [], []
        for s in seeds:
            world = generate_world(
                history_per_pharmacy=h, seed=s, pharmacies=pharmacies, shortage_dates=shortage_dates
            )
            X, y = build_training_matrix(world.observations, world.pharmacies, world.shortage_dates)
            model = StockModel(seed=s).fit(X, y)
            npis, Xr = build_request_features(
                world.pharmacies, world.observations, world.as_of, world.shortage_dates
            )
            probs = model.predict_proba(Xr)
            ls.append(_order_ecf(order_calls(npis, probs), world.true_prob_today))
            bs.append(_baseline_ecf(npis, world.true_prob_today, seed=s))
        learned_out.append(float(np.mean(ls)))
        baseline_out.append(float(np.mean(bs)))
    return list(history_levels), baseline_out, learned_out


def render_ecf_chart(path, history_levels, baseline, learned, n_pharmacies, caption=None) -> str:
    """Render the hero flywheel chart to a PNG. Returns the path."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(history_levels, baseline, "o--", color="#9aa0a6", label="naive (random call order)")
    ax.plot(history_levels, learned, "o-", color="#1a73e8", linewidth=2,
            label="CallBudget (learned ranking)")
    ax.set_xlabel("calls of history per pharmacy  (data accumulated)")
    ax.set_ylabel("expected calls to find the drug")
    ax.set_title(
        "CallBudget flywheel: fewer calls as call history accumulates\n"
        f"Adderall XR 20mg, {n_pharmacies} real Boston pharmacies, drug in shortage"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    if caption:
        fig.text(
            0.5, -0.04, caption, ha="center", va="top", fontsize=7.5, color="#5f6368", wrap=True
        )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def false_positive_in_stock_rate(graded: list[tuple[bool, ExtractionRecord]]) -> dict[str, float]:
    """Given (truth_in_stock, extracted_record) pairs, compute the safety metrics.

    false_positive_in_stock = P(said yes | truth no), on NON-abstained calls.
    Also returns abstention_rate and the count used.
    """
    decided = [(t, r) for t, r in graded if not r.abstain]
    truth_no = [(t, r) for t, r in decided if not t]
    fp = sum(1 for _t, r in truth_no if r.in_stock == StockState.YES)
    return {
        "false_positive_in_stock": (fp / len(truth_no)) if truth_no else 0.0,
        "abstention_rate": 1.0 - (len(decided) / len(graded)) if graded else 0.0,
        "n_decided": float(len(decided)),
    }


def make_noisy_extractor(p_flip: float = 0.25, seed: int = 0):
    """Wrap keyword_extract to misread in_stock with probability p_flip.

    Simulates an unreliable LLM/STT: most reads are right, some flip to a
    confident wrong answer. Lets us show that self-consistency (n_samples > 1)
    converts flips into abstentions instead of false positives.
    """
    rng = random.Random(seed)
    options = ["yes", "no", "partial", "unknown"]

    def _noisy(transcript: str) -> dict:
        out = dict(keyword_extract(transcript))
        if rng.random() < p_flip:
            out["in_stock"] = rng.choice([o for o in options if o != out.get("in_stock")])
            out["abstain"] = False  # a confident misread: the dangerous kind
        return out

    return _noisy


def run_calls(pharmacies, true_prob, extractor, n_samples=3, trials=10, seed=7):
    """Simulate calls against SimPharmacist; return (truly_in_stock, record) pairs."""
    rng = random.Random(seed)
    graded = []
    for ph in pharmacies:
        p = true_prob[ph.npi]
        for _ in range(trials):
            truth = rng.random() < p
            style = "hedge" if rng.random() < 0.25 else "clear"
            s = rng.randint(0, 10**6)
            pharm = SimPharmacist(ph.npi, truly_in_stock=truth, style=style, seed=s)
            rec = extract(pharm.answer("in stock?"), extractor=extractor, n_samples=n_samples)
            graded.append((truth, rec))
    return graded


def reliability_panel(pharmacies, true_prob, n_samples=3, p_flip=0.25, trials=10, seed=7):
    """Safety metrics under an unreliable extractor (false-positive-in-stock, etc.)."""
    graded = run_calls(
        pharmacies, true_prob, make_noisy_extractor(p_flip, seed),
        n_samples=n_samples, trials=trials, seed=seed,
    )
    m = false_positive_in_stock_rate(graded)
    decided = [(t, r) for t, r in graded if not r.abstain]
    correct = sum((r.in_stock in (StockState.YES, StockState.PARTIAL)) == t for t, r in decided)
    m["decided_accuracy"] = correct / len(decided) if decided else 0.0
    m["n_calls"] = float(len(graded))
    return m

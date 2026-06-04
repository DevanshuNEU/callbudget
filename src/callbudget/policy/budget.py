"""Policy: turn stock probabilities into a call order, and compute the cost.

================================  THEORY  ====================================

ACTIVE SENSING UNDER A BUDGET.
The prediction layer tells us P(in_stock) for each pharmacy. The policy decides
the ORDER to call them in, to find the drug in as few calls as possible. This is
a sequential decision problem: each call is an "action" that costs 1 and yields
information; we stop the moment we get a confirmed in-stock.

THE GREEDY POLICY (v1, locked).
Call pharmacies in descending P(in_stock). Stop at the first YES. This is
provably optimal for the simplest objective (minimize expected calls to first
success when calls are independent): you should always try the most-likely
remaining option next. We keep v1 here for clarity; a budgeted bandit /
optimal-stopping variant is deferred (ROADMAP).

EXPECTED CALLS-TO-FIND (ECF), the hero metric.
If we call pharmacies in order with success probabilities p1, p2, p3, ... then
the chance we are still searching when we place call k is the chance the first
k-1 all missed. The expected number of calls is:

    ECF = sum over k of  P(reach call k)
        = 1 + (1-p1) + (1-p1)(1-p2) + (1-p1)(1-p2)(1-p3) + ...

Intuition: you always pay for call 1. You pay for call 2 only if call 1 missed
(prob 1-p1). You pay for call 3 only if 1 and 2 both missed, and so on. Sorting
by descending p minimizes this sum, which is exactly why the greedy order is the
right order. The whole project's headline is this number dropping (~10 -> ~2-3)
as the model's p-values sharpen with more data.

WHY THIS BEATS THE BASELINE.
A naive baseline (call by proximity or at random) effectively uses uninformative,
roughly equal p-values, so ECF stays near "call about half of them." A learned
model concentrates probability on the right pharmacies, front-loading the YES,
which collapses ECF. The GAP between baseline-ECF and learned-ECF is the moat,
quantified.

=============================================================================
"""

from __future__ import annotations


def expected_calls_to_find(success_probs: list[float]) -> float:
    """ECF for a given call order (already sorted by the caller).

    Implements the survival-sum above. This one is real (it is pure math, no
    model needed), so the metric is unit-tested on its own.
    """
    ecf = 0.0
    reach_prob = 1.0  # probability we are still searching when we place this call
    for p in success_probs:
        ecf += reach_prob
        reach_prob *= 1.0 - p  # we only place the next call if this one missed
    return ecf


def order_calls(npis: list[str], success_probs: list[float]) -> list[str]:
    """Greedy order: pharmacies sorted by descending P(in_stock)."""
    paired = sorted(zip(npis, success_probs, strict=True), key=lambda t: t[1], reverse=True)
    return [npi for npi, _ in paired]

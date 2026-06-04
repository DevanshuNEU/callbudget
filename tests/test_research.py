"""Auto-research loop + ECF property tests. Offline, deterministic, fast.

Three concerns, kept separate:
  A) the config contract (content-addressed identity + in-space checking),
  B) the loop's integrity invariants on a TINY run that never touches runs/,
  C) the ECF math, exercised as PROPERTIES (Hypothesis) rather than spot values.

Everything runs offline (CALLBUDGET_OFFLINE is forced on at import) and writes any
journal/report into pytest's tmp_path, so the committed runs/ directory is untouched.
"""

from __future__ import annotations

import json
import math
import os

# Force offline before any callbudget import: the evaluator's loaders read this.
os.environ["CALLBUDGET_OFFLINE"] = "1"

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from rich.console import Console  # noqa: E402

from callbudget.policy.budget import expected_calls_to_find  # noqa: E402
from callbudget.research.evaluate import evaluate  # noqa: E402
from callbudget.research.loop import TrialRecord, run_research  # noqa: E402
from callbudget.research.space import (  # noqa: E402
    DEFAULT_CONFIG,
    SEARCH_SPACE,
    ResearchConfig,
)

# ----------------------------------------------------------------------------
# A) Config contract
# ----------------------------------------------------------------------------


def test_run_id_is_stable_and_order_independent():
    a = ResearchConfig(max_depth=3, max_iter=200, learning_rate=0.1, n_samples=5,
                       abstain_threshold=0.67)
    # Same values, kwargs reordered -> identical id (id is the content, not the order).
    b = ResearchConfig(abstain_threshold=0.67, n_samples=5, learning_rate=0.1,
                       max_iter=200, max_depth=3)
    assert a.run_id() == b.run_id()
    assert a.run_id() == a.run_id()  # deterministic across calls
    # A changed value -> a different id.
    c = ResearchConfig(max_depth=3, max_iter=200, learning_rate=0.1, n_samples=7,
                       abstain_threshold=0.67)
    assert c.run_id() != a.run_id()


def test_in_space_accepts_default_and_corners_rejects_out_of_space():
    assert DEFAULT_CONFIG.in_space()
    # The two extreme corners of the discrete grid are legal points.
    low_corner = ResearchConfig(
        max_depth=SEARCH_SPACE["max_depth"][0],
        max_iter=SEARCH_SPACE["max_iter"][0],
        learning_rate=SEARCH_SPACE["learning_rate"][0],
        n_samples=SEARCH_SPACE["n_samples"][0],
        abstain_threshold=SEARCH_SPACE["abstain_threshold"][0],
    )
    high_corner = ResearchConfig(
        max_depth=SEARCH_SPACE["max_depth"][-1],
        max_iter=SEARCH_SPACE["max_iter"][-1],
        learning_rate=SEARCH_SPACE["learning_rate"][-1],
        n_samples=SEARCH_SPACE["n_samples"][-1],
        abstain_threshold=SEARCH_SPACE["abstain_threshold"][-1],
    )
    assert low_corner.in_space()
    assert high_corner.in_space()
    # A value off the grid is rejected (max_depth=5 is not in [2, 3, 4]).
    assert not ResearchConfig(max_depth=5).in_space()


# ----------------------------------------------------------------------------
# B) The loop (tiny params + tmp paths: fast, and never touches runs/)
# ----------------------------------------------------------------------------


def test_research_loop_invariants(tmp_path):
    journal = tmp_path / "j.jsonl"
    report = tmp_path / "r.md"
    outcome = run_research(
        rounds=2,
        trials=3,
        search_seeds=[1, 2],
        validation_seeds=[3, 4],
        journal_path=journal,
        report_path=report,
        console=Console(quiet=True),
    )

    # Core integrity invariant: nothing accepted breaches safety.
    for r in outcome.trials:
        if r.feasible:
            assert r.false_positive_in_stock <= outcome.bar

    # The constraint can bind: a config with no self-consistency breaches the bar.
    # The n_samples=1 anchor should be rejected in the tiny run; if it happens not to
    # bind on these few seeds, we prove the constraint CAN bind via a direct eval.
    rejected_for_safety = [
        r for r in outcome.trials
        if not r.feasible and r.false_positive_in_stock > outcome.bar
    ]
    if rejected_for_safety:
        # The loop itself rejected at least one unsafe config: the bar did real work.
        assert all(r.false_positive_in_stock > outcome.bar for r in rejected_for_safety)
    else:
        # The tiny loop run was too small to bind the constraint. Prove it CAN bind
        # directly: with no self-consistency (n_samples=1) the false-positive rate
        # exceeds the baseline's. This needs enough rounds/seeds to accrue decided
        # truth-no calls, so we use a slightly larger but still fast direct eval.
        seeds = [1, 3, 5, 7]
        unsafe = evaluate(ResearchConfig(n_samples=1, abstain_threshold=0.6), seeds, 12)
        baseline = evaluate(DEFAULT_CONFIG, seeds, 12)
        assert unsafe.false_positive_in_stock > baseline.false_positive_in_stock

    # The journal exists and round-trips: each line is a TrialRecord whose run_id
    # matches the run_id of the config it carries.
    assert journal.exists()
    lines = journal.read_text().splitlines()
    assert lines
    for line in lines:
        rec = TrialRecord.model_validate(json.loads(line))
        assert rec.run_id == ResearchConfig(**rec.config).run_id()

    # The winner was validated on held-out seeds, and the recommendation is a config.
    assert math.isfinite(outcome.best_ecf_validation)
    assert isinstance(outcome.recommended_config, ResearchConfig)


def test_evaluate_is_deterministic():
    a = evaluate(DEFAULT_CONFIG, [1, 2], 3)
    b = evaluate(DEFAULT_CONFIG, [1, 2], 3)
    assert a.ecf == b.ecf
    assert a.ecf_by_round == b.ecf_by_round


# ----------------------------------------------------------------------------
# C) Hypothesis property tests for the ECF math
# ----------------------------------------------------------------------------

_probs = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=12)
_TOL = 1e-9


@given(_probs)
def test_ecf_is_bounded_between_one_and_n(probs):
    ecf = expected_calls_to_find(probs)
    n = len(probs)
    # You always pay for call 1, and never more than n calls exist.
    assert 1.0 - _TOL <= ecf <= n + _TOL


@settings(max_examples=200)
@given(_probs, st.randoms(use_true_random=False))
def test_greedy_order_is_optimal(probs, rng):
    greedy = sorted(probs, reverse=True)
    other = list(probs)
    rng.shuffle(other)
    # Descending order minimizes the survival sum: no permutation does better.
    assert expected_calls_to_find(greedy) <= expected_calls_to_find(other) + _TOL


@given(_probs, st.integers(min_value=0), st.floats(min_value=0.0, max_value=1.0))
def test_ecf_non_increasing_when_a_probability_rises(probs, idx, bump):
    i = idx % len(probs)
    raised = list(probs)
    raised[i] = min(1.0, raised[i] + bump)
    assume(raised[i] >= probs[i])
    # Raising any success probability can only lower (or hold) the expected calls.
    before = expected_calls_to_find(sorted(probs, reverse=True))
    after = expected_calls_to_find(sorted(raised, reverse=True))
    assert after <= before + _TOL

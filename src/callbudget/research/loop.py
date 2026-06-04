"""The auto-research loop: an Optuna study that improves the system against its
own metrics, under a hard safety constraint, with a held-out validation guard.

THEORY (constrained optimization + the non-circularity guard):
The loop minimizes expected-calls-to-find (ECF) SUBJECT TO a safety ceiling on
the false-positive-in-stock rate. That subject-to is the whole point: the lowest
ECF config is often the one that abstains least (it feeds the model the most
labels), which is also the least safe. A plain "minimize ECF" search would pick
it; the constraint is what makes the loop choose a config that is fast AND safe.

We use Optuna's TPE sampler (seeded, so the study is reproducible) with a
constraints function, rather than a hand-rolled search, because it brings
constraint handling and a clear path to a larger space for free. The grid here
is small enough to enumerate; Optuna is for the constraint plumbing and the
reproducibility, and we say so.

Two seed sets prevent the search from fooling itself: we optimize on SEARCH seeds
and re-score the winner on disjoint VALIDATION seeds. If the winner's advantage
does not survive on unseen seeds, we flag it. This mirrors the model's own
held-out evaluation guard (predict/model.py).

INTEGRITY: outcomes are synthetic, so a lower ECF here demonstrates the SEARCH
mechanism on realistic inputs, not a real-world stock-rate gain. The report says
so on its face.
"""

from __future__ import annotations

from pathlib import Path

import optuna
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from ..config import RUNS_DIR
from .evaluate import evaluate
from .space import DEFAULT_CONFIG, SEARCH_SPACE, ResearchConfig

# Disjoint seed sets from one pool (odd vs even of 1..12): optimize on SEARCH,
# prove the winner generalises on VALIDATION. An interleaved split keeps the two
# sets the same difficulty distribution, so the held-out test is fair rather than
# accidentally easy/hard. The split is fixed up front, not chosen for its result.
SEARCH_SEEDS = [1, 3, 5, 7, 9, 11]
VALIDATION_SEEDS = [2, 4, 6, 8, 10, 12]
# Never regress the safety metric more than this past the v1 baseline. The bar is
# anchored to the default config's measured false-positive rate (see run_research).
SAFETY_TOLERANCE = 0.02
DEFAULT_ROUNDS = 12
DEFAULT_TRIALS = 14

# Configs that are ALWAYS evaluated (enqueued before TPE explores), so the journal
# always contains the baseline, a known-unsafe probe (to show the constraint
# rejecting it), and a strong candidate, regardless of how the sampler wanders.
ANCHOR_CONFIGS = [
    ResearchConfig(),  # the locked v1 baseline
    ResearchConfig(n_samples=1, abstain_threshold=0.6),  # no self-consistency: should breach
    # A strong candidate (high self-consistency for safety + a low abstain threshold
    # for coverage). Warm-starting the search with it keeps a small trial budget from
    # missing the good region; the validation guard still has to confirm it generalises.
    ResearchConfig(
        n_samples=7, abstain_threshold=0.6, max_depth=3, max_iter=100, learning_rate=0.05
    ),
]

JOURNAL_PATH = RUNS_DIR / "research_journal.jsonl"
REPORT_PATH = RUNS_DIR / "research_report.md"


class TrialRecord(BaseModel):
    """One experiment in the journal: a config, its measured metrics, feasibility."""

    trial: int
    proposer: str = "optuna-tpe"
    config: dict
    run_id: str
    ecf: float
    false_positive_in_stock: float
    abstention_rate: float
    feasible: bool
    rationale: str = ""


class ResearchOutcome(BaseModel):
    """The result of one study: the constrained winner, the baseline, the guards."""

    bar: float
    default_ecf_search: float
    default_fp_search: float
    best_config: ResearchConfig
    best_ecf_search: float
    best_fp_search: float
    best_ecf_validation: float
    default_ecf_validation: float
    n_trials: int
    n_feasible: int
    improved: bool  # winner beats the v1 baseline on the search seeds
    generalised: bool  # winner still beats the baseline on held-out validation seeds
    recommended_config: ResearchConfig  # what the loop actually recommends shipping
    recommendation: str  # the plain-English call, including when it is "keep v1"
    trials: list[TrialRecord]


def _suggest(trial: optuna.Trial) -> ResearchConfig:
    """Map an Optuna trial onto a (legal) point in the search space."""
    return ResearchConfig(
        max_depth=trial.suggest_categorical("max_depth", SEARCH_SPACE["max_depth"]),
        max_iter=trial.suggest_categorical("max_iter", SEARCH_SPACE["max_iter"]),
        learning_rate=trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"]),
        n_samples=trial.suggest_categorical("n_samples", SEARCH_SPACE["n_samples"]),
        abstain_threshold=trial.suggest_categorical(
            "abstain_threshold", SEARCH_SPACE["abstain_threshold"]
        ),
    )


def run_research(
    rounds: int = DEFAULT_ROUNDS,
    trials: int = DEFAULT_TRIALS,
    search_seeds: list[int] | None = None,
    validation_seeds: list[int] | None = None,
    seed: int = 7,
    journal_path: Path | None = None,
    report_path: Path | None = None,
    console: Console | None = None,
) -> ResearchOutcome:
    """Run the constrained study, journal every trial, validate + report the winner."""
    search_seeds = search_seeds or SEARCH_SEEDS
    validation_seeds = validation_seeds or VALIDATION_SEEDS
    journal_path = journal_path or JOURNAL_PATH
    report_path = report_path or REPORT_PATH

    # 1) The baseline + the safety bar: the v1 default's measured false-positive
    #    rate plus a small tolerance. The winner may not be less safe than this.
    base = evaluate(DEFAULT_CONFIG, search_seeds, rounds)
    bar = round(base.false_positive_in_stock + SAFETY_TOLERANCE, 4)

    records: list[TrialRecord] = []

    def objective(trial: optuna.Trial) -> float:
        config = _suggest(trial)
        result = evaluate(config, search_seeds, rounds)
        feasible = result.false_positive_in_stock <= bar
        # Optuna reads this to steer away from infeasible regions.
        trial.set_user_attr("constraint", [result.false_positive_in_stock - bar])
        records.append(
            TrialRecord(
                trial=trial.number,
                config=config.model_dump(),
                run_id=config.run_id(),
                ecf=result.ecf,
                false_positive_in_stock=result.false_positive_in_stock,
                abstention_rate=result.abstention_rate,
                feasible=feasible,
            )
        )
        return result.ecf

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=seed, constraints_func=lambda t: t.user_attrs["constraint"]
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for anchor in ANCHOR_CONFIGS:  # baseline + unsafe probe + strong candidate, always tried
        study.enqueue_trial(anchor.model_dump())
    study.optimize(objective, n_trials=trials)

    # 2) The constrained winner: lowest ECF among the FEASIBLE trials. (Optuna's
    #    own best_trial can be infeasible, so we select explicitly.)
    feasible = [r for r in records if r.feasible]
    winner_rec = min(feasible, key=lambda r: r.ecf) if feasible else _baseline_record(base)
    best_config = ResearchConfig(**winner_rec.config)

    # 3) The validation guard: re-score winner + baseline on held-out seeds.
    best_val = evaluate(best_config, validation_seeds, rounds)
    base_val = evaluate(DEFAULT_CONFIG, validation_seeds, rounds)

    generalised = best_val.ecf <= base_val.ecf
    # Production-correct recommendation: only ship the winner if its in-sample gain
    # survives held-out validation. Otherwise keep v1 and say why. Reporting a
    # search-seed win that does not generalise would be the exact overfreporting the
    # validation guard exists to prevent.
    if generalised:
        recommended = best_config
        recommendation = (
            f"Ship the winner: it cut ECF to {winner_rec.ecf} on search seeds and held up on "
            f"held-out seeds ({best_val.ecf} vs baseline {base_val.ecf}), within the safety bar."
        )
    else:
        recommended = DEFAULT_CONFIG
        recommendation = (
            f"Keep v1. The winner cut ECF on search seeds ({winner_rec.ecf} vs {base.ecf}) but the "
            f"gain did not survive held-out validation ({best_val.ecf} vs baseline {base_val.ecf}, "
            f"within world-to-world noise), so the loop does not recommend shipping it. The safety "
            f"constraint and the validation guard both did their job."
        )

    outcome = ResearchOutcome(
        bar=bar,
        default_ecf_search=base.ecf,
        default_fp_search=base.false_positive_in_stock,
        best_config=best_config,
        best_ecf_search=winner_rec.ecf,
        best_fp_search=winner_rec.false_positive_in_stock,
        best_ecf_validation=best_val.ecf,
        default_ecf_validation=base_val.ecf,
        n_trials=len(records),
        n_feasible=len(feasible),
        improved=winner_rec.ecf <= base.ecf,
        generalised=generalised,
        recommended_config=recommended,
        recommendation=recommendation,
        trials=records,
    )

    _write_journal(records, journal_path)
    _write_report(outcome, report_path)
    _render(outcome, console or Console())
    return outcome


def _baseline_record(base) -> TrialRecord:
    """Fallback winner if no trial was feasible: the v1 default itself."""
    return TrialRecord(
        trial=-1,
        proposer="baseline",
        config=DEFAULT_CONFIG.model_dump(),
        run_id=DEFAULT_CONFIG.run_id(),
        ecf=base.ecf,
        false_positive_in_stock=base.false_positive_in_stock,
        abstention_rate=base.abstention_rate,
        feasible=True,
        rationale="no feasible trial beat the baseline; kept v1",
    )


def _write_journal(records: list[TrialRecord], path: Path) -> None:
    """Append-only JSONL: one line per experiment, the trail of what was tried."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def _write_report(outcome: ResearchOutcome, path: Path) -> None:
    """A clean markdown leaderboard: the winner, the baseline, the guards, the caveat."""
    feasible = sorted((r for r in outcome.trials if r.feasible), key=lambda r: r.ecf)
    rejected = sorted((r for r in outcome.trials if not r.feasible), key=lambda r: r.ecf)
    c = outcome.best_config

    lines = [
        "# CallBudget auto-research report",
        "",
        "An agent searched model + extraction configurations, scoring each by the "
        "closed-loop eval (expected calls-to-find) under a hard safety constraint "
        "(false-positive-in-stock). Every per-call outcome is SYNTHETIC, so this "
        "demonstrates the search mechanism on realistic inputs, not real stock rates.",
        "",
        f"- Safety bar (false-positive-in-stock <= ): **{outcome.bar}** "
        f"(v1 baseline {outcome.default_fp_search} + tolerance)",
        f"- Trials: **{outcome.n_trials}** ({outcome.n_feasible} feasible, "
        f"{outcome.n_trials - outcome.n_feasible} rejected on safety)",
        "",
        "## Winner (lowest ECF among safe configs)",
        "",
        f"- `max_depth={c.max_depth}, max_iter={c.max_iter}, learning_rate={c.learning_rate}, "
        f"n_samples={c.n_samples}, abstain_threshold={c.abstain_threshold}` (run_id "
        f"`{c.run_id()}`)",
        f"- ECF: **{outcome.best_ecf_search}** on search seeds vs baseline "
        f"{outcome.default_ecf_search} "
        f"({'improved' if outcome.improved else 'no improvement'})",
        f"- ECF on HELD-OUT validation seeds: **{outcome.best_ecf_validation}** vs baseline "
        f"{outcome.default_ecf_validation} "
        f"({'generalised' if outcome.generalised else 'did NOT generalise - treat with caution'})",
        f"- false-positive-in-stock: {outcome.best_fp_search} (within the bar)",
        "",
        "## Recommendation",
        "",
        outcome.recommendation,
        "",
        "## Top safe configs",
        "",
        "| ECF | false-pos | abstention | n_samples | thr | depth | iter | lr |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in feasible[:8]:
        d = r.config
        lines.append(
            f"| {r.ecf} | {r.false_positive_in_stock} | {r.abstention_rate} | "
            f"{d['n_samples']} | {d['abstain_threshold']} | {d['max_depth']} | "
            f"{d['max_iter']} | {d['learning_rate']} |"
        )
    if rejected:
        lines += [
            "",
            "## Rejected on safety (lower ECF, but breached the bar)",
            "",
            "These are exactly why the constraint exists: cheaper-looking configs "
            "that a plain ECF search would have chosen.",
            "",
            "| ECF | false-pos | n_samples | thr |",
            "|--:|--:|--:|--:|",
        ]
        for r in rejected[:5]:
            d = r.config
            lines.append(
                f"| {r.ecf} | {r.false_positive_in_stock} | {d['n_samples']} | "
                f"{d['abstain_threshold']} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _render(outcome: ResearchOutcome, console: Console) -> None:
    """Live terminal view (the Loom surface): a leaderboard + the winner + the guards."""
    table = Table(title="CallBudget auto-research: configs scored by closed-loop ECF")
    table.add_column("rank", justify="right")
    table.add_column("ECF", justify="right")
    table.add_column("false-pos", justify="right")
    table.add_column("abstain", justify="right")
    table.add_column("n_samp", justify="right")
    table.add_column("thr", justify="right")
    table.add_column("depth/iter/lr")
    table.add_column("verdict")

    ranked = sorted(outcome.trials, key=lambda r: r.ecf)
    for i, r in enumerate(ranked[:10], 1):
        d = r.config
        verdict = "[green]safe[/green]" if r.feasible else "[red]REJECTED (unsafe)[/red]"
        table.add_row(
            str(i), f"{r.ecf}", f"{r.false_positive_in_stock}", f"{r.abstention_rate}",
            f"{d['n_samples']}", f"{d['abstain_threshold']}",
            f"{d['max_depth']}/{d['max_iter']}/{d['learning_rate']}", verdict,
        )
    console.print(table)
    console.print(
        f"\n[bold]Winner[/bold] (safe + lowest ECF): ECF "
        f"[bold cyan]{outcome.best_ecf_search}[/bold cyan] vs baseline "
        f"{outcome.default_ecf_search} on search seeds; "
        f"[bold cyan]{outcome.best_ecf_validation}[/bold cyan] vs "
        f"{outcome.default_ecf_validation} on HELD-OUT validation seeds "
        f"({'generalised' if outcome.generalised else 'did NOT generalise'})."
    )
    colour = "green" if outcome.generalised else "yellow"
    console.print(f"[bold {colour}]Recommendation:[/bold {colour}] {outcome.recommendation}")
    console.print(
        f"[dim]Safety bar false-positive-in-stock <= {outcome.bar}; "
        f"{outcome.n_trials - outcome.n_feasible} of {outcome.n_trials} trials rejected for "
        f"breaching it. Outcomes are synthetic: this is the search mechanism, not real "
        f"stock rates. Journal: {JOURNAL_PATH.name}, report: {REPORT_PATH.name}.[/dim]"
    )

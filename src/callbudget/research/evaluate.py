"""The fitness function: score one configuration by running the closed loop.

THEORY (the closed loop is what makes self-improvement measurable):
A decoupled eval would tune the model against ECF and the voice knobs against the
safety metric as two unrelated problems. That hides the project's whole thesis.
Here we run the REAL loop instead:

    rank -> call a pharmacy -> extract (or abstain) -> a confident call becomes a
    label -> the model refits -> re-rank ...

Because a confident-but-wrong extraction is stored as a (wrong) label, extraction
quality flows into the model and therefore into ECF. So both knob families move
the one number we report:
  - good abstention keeps labels clean -> the model sharpens -> ECF drops;
  - abstaining too much starves the model of labels -> ECF stalls;
  - model hyperparameters set how well the model fits whatever labels it gets.
The loop optimizes the tradeoff. That is the flywheel ("abstention protects label
quality protects fewer-calls") turned into a measurement rather than a claim.

HONESTY (stated out loud, because it would otherwise look like rigging):
The loop runs under a DELIBERATELY UNRELIABLE extractor (the same 25%-misread
stress model as the reliability ablation). Under a perfect extractor the
abstention knobs are no-ops, so the only regime in which they can earn their keep
is a noisy one. We say so rather than quietly choosing the noise that flatters
abstention. And the integrity boundary is unchanged: per-call outcomes are still
synthetic, so a lower ECF here proves the SEARCH mechanism on realistic inputs,
never a real-world stock-rate gain.

DETERMINISM: every source of randomness is seeded, so a (config, seeds, rounds)
triple always yields the same EvalResult. That is what lets the journal be a
trustworthy record and the optimizer reproducible.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from ..evals.grade import make_noisy_extractor
from ..mcp.engine import CallBudgetEngine
from ..schemas import StockState
from ..sim.pharmacist import SimPharmacist
from ..voice.loop import simulate_text_call
from .space import EvalResult, ResearchConfig

# Defaults sized for a live run that finishes in a couple of minutes. Lower them
# (CLI flags) for a quick demo; raise them for a tighter estimate. The canonical
# 19-pharmacy Boston subset is fixed via limit=28 (post-dedup), as everywhere else.
PHARMACY_LIMIT = 28
INITIAL_HISTORY = 6  # calls/pharmacy to seed the model before the loop (room to improve)
NOISE_P_FLIP = 0.25  # the stress level: the regime where abstention matters


def evaluate(
    config: ResearchConfig,
    seeds: list[int],
    rounds: int,
    initial_history: int = INITIAL_HISTORY,
) -> EvalResult:
    """Run the closed loop under `config` for each seed and return the mean metrics.

    For each seed we stand up a fresh engine on its OWN temporary observation store
    (never the committed log), seed it with a little history, then place `rounds`
    calls through the noisy extractor parameterised by the config's self-consistency
    knobs, learning from every confident answer. We record ECF after each round (the
    trajectory) and grade every call against its hidden ground truth (the safety
    metric). Metrics are pooled/averaged across seeds.
    """
    per_seed_traj: list[list[float]] = []
    fp_count = 0
    truth_no_decided = 0
    abstained = 0
    total_calls = 0

    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            engine = CallBudgetEngine(
                limit=PHARMACY_LIMIT,
                history_per_pharmacy=initial_history,
                seed=seed,
                store_dir=Path(tmp) / f"seed_{seed}",
                model_kwargs=config.model_kwargs,
            )
            # One noisy extractor per seed: a single RNG stream, so the n_samples
            # self-consistency draws inside each call are independent flips.
            noisy = make_noisy_extractor(p_flip=NOISE_P_FLIP, seed=seed)
            rng = random.Random(seed)
            true_prob = engine.world.true_prob_today
            npis = list(engine.npis)

            traj = [engine.plan_calls()["expected_calls_to_find"]]  # ECF before the loop
            for r in range(rounds):
                # Round-robin coverage: sweep calls across pharmacies so the log
                # fills evenly. (A bandit explore/exploit caller is the ROADMAP
                # variant; balanced coverage keeps the flywheel signal clean here.)
                npi = npis[r % len(npis)]
                truth = rng.random() < true_prob[npi]
                style = "hedge" if rng.random() < 0.25 else "clear"
                pharmacist = SimPharmacist(
                    npi, truly_in_stock=truth, style=style, seed=rng.randint(0, 10**6)
                )
                # Reconstruct the call locally (engine.place_call is the production
                # path) so we retain the hidden truth needed to grade safety.
                result = simulate_text_call(
                    pharmacist,
                    engine.drug,
                    extractor=noisy,
                    n_samples=config.n_samples,
                    abstain_threshold=config.abstain_threshold,
                )
                rec = result.record
                total_calls += 1
                if rec.abstain:
                    abstained += 1
                elif not truth:
                    truth_no_decided += 1
                    if rec.in_stock == StockState.YES:
                        fp_count += 1
                engine.observe(result)  # confident -> label + refit; abstained -> skip
                traj.append(engine.plan_calls()["expected_calls_to_find"])
            per_seed_traj.append(traj)

    ecf_by_round = [sum(col) / len(col) for col in zip(*per_seed_traj, strict=True)]
    ecf = sum(t[-1] for t in per_seed_traj) / len(per_seed_traj)
    false_positive = (fp_count / truth_no_decided) if truth_no_decided else 0.0
    abstention_rate = (abstained / total_calls) if total_calls else 0.0

    return EvalResult(
        config_id=config.run_id(),
        ecf=round(ecf, 4),
        ecf_by_round=[round(v, 4) for v in ecf_by_round],
        false_positive_in_stock=round(false_positive, 4),
        abstention_rate=round(abstention_rate, 4),
        n_seeds=len(seeds),
        rounds=rounds,
    )

"""The auto-research contract: what a configuration is, and what evaluating it yields.

THEORY (search space + content-addressed identity):
An auto-research loop is a search over a CONFIGURATION SPACE scored by a fitness
function. To do that honestly you need three things pinned down in one place, so
the optimizer, the evaluator, the journal, and the live proposer all agree:

  1. ResearchConfig - the point in the space (the knobs we are allowed to turn).
  2. SEARCH_SPACE   - the legal values for each knob (so a proposal is checkable).
  3. EvalResult     - the measured outcome of one configuration.

Each config carries a content-addressed `run_id` (a hash of its resolved values,
the same trick as config.py): the identity IS the content, so a result always
traces back to the exact knobs that produced it and two identical configs hash
the same on any machine.

WHY THESE KNOBS (and not others):
- Prediction (max_depth, max_iter, learning_rate): the model's capacity/fit. They
  move ECF, the cost metric.
- Voice (n_samples, abstain_threshold): the self-consistency knobs. They move
  label quality, hence the safety metric (false-positive-in-stock) and, through
  cleaner labels, ECF in the closed loop.
- The POLICY is deliberately NOT in the space: greedy is provably optimal for this
  objective (policy/budget.py), so searching it would be searching a question
  theory already closed. Leaving it out is the point, not an omission.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

# The legal grid. Discrete + small (3^5 = 243 points) on purpose: the optimizer is
# Optuna for the constraint handling + reproducibility + the path to a larger
# space, not because TPE is load-bearing at this size (we say so in the docs).
SEARCH_SPACE: dict[str, list] = {
    "max_depth": [2, 3, 4],
    "max_iter": [100, 200, 400],
    "learning_rate": [0.05, 0.1, 0.2],
    # n_samples=1 means NO self-consistency: it is in the space on purpose, so the
    # search can discover configs that look cheap but breach the safety bar. That
    # is what gives the constraint real work to do (it rejects them).
    "n_samples": [1, 3, 5, 7],
    "abstain_threshold": [0.6, 0.67, 0.75],
}


class ResearchConfig(BaseModel):
    """One point in the search space. Defaults are the locked v1 baseline, so a
    default config reproduces the shipped behaviour exactly."""

    # Prediction (-> model.StockModel)
    max_depth: int = 3
    max_iter: int = 200
    learning_rate: float = 0.1
    # Voice self-consistency (-> voice.extract / engine.place_call)
    n_samples: int = 3
    abstain_threshold: float = 0.67

    @property
    def model_kwargs(self) -> dict:
        """The subset that constructs a StockModel (engine accepts this verbatim)."""
        return {
            "max_depth": self.max_depth,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
        }

    def run_id(self) -> str:
        """Stable 12-char id from the canonical JSON of the resolved config.

        sort_keys makes it order-independent, so the id depends only on values.
        """
        canonical = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def in_space(self) -> bool:
        """True if every value is legal. The live proposer's output is checked here."""
        return all(getattr(self, k) in v for k, v in SEARCH_SPACE.items())


# The locked v1 baseline: the incumbent the loop must beat, and the reference whose
# measured false-positive rate becomes the safety bar.
DEFAULT_CONFIG = ResearchConfig()


class EvalResult(BaseModel):
    """The measured outcome of evaluating one ResearchConfig.

    Raw metrics only - feasibility (does it satisfy the safety bar) is decided by
    the loop against the bar, not baked in here, so the evaluator stays decoupled
    from the optimizer.
    """

    config_id: str
    ecf: float = Field(description="mean expected-calls-to-find after the closed loop, over seeds")
    ecf_by_round: list[float] = Field(
        default_factory=list, description="mean ECF trajectory across closed-loop rounds"
    )
    false_positive_in_stock: float = Field(description="P(said yes | truth no); the safety metric")
    abstention_rate: float
    n_seeds: int
    rounds: int

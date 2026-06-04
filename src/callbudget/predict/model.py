"""The stock-probability model: learns P(in_stock = yes | features).

================================  THEORY  ====================================

1) WHY A PROBABILITY, NOT A YES/NO.
   The policy (policy/budget.py) needs to RANK pharmacies and compute an
   EXPECTED number of calls. Ranking and expectations need real-valued
   probabilities, not hard labels. So we want `predict_proba`, not `predict`.

2) WHY GRADIENT-BOOSTED TREES (xgboost), NOT A NEURAL NET.
   Our data is TABULAR (a handful of numeric/categorical columns) and small
   (dozens of pharmacies, thousands of calls). For tabular data, gradient-boosted
   decision trees are the reigning champion: they capture nonlinearities and
   feature interactions automatically, need no feature scaling, tolerate missing
   values, and train in milliseconds at our size. Neural nets shine on raw
   high-dimensional signals (pixels, audio, text) with lots of data; here they
   would be slower, hungrier, and less interpretable for no benefit.

   What "gradient boosting" means in one breath: build many shallow trees in
   sequence, where each new tree is trained to fix the ERRORS (residuals) the
   running ensemble still makes. Sum their outputs, squash through a logistic to
   get a probability. It is "boosting" because weak learners (shallow trees) are
   combined into a strong one, and "gradient" because each tree follows the
   gradient of the loss.

3) WHY INTERPRETABILITY IS A REQUIREMENT, NOT A LUXURY.
   An operator needs to know WHY a pharmacy ranked high ("high throughput +
   recent in-stock history + not in shortage"). Trees give us feature-importance
   and per-prediction attributions for free. A black box would forfeit that.
   See `explain()`.

4) CALIBRATION (subtle but it matters here).
   A model can rank perfectly yet have miscalibrated probabilities: it says 0.9
   when the true in-stock rate for such cases is 0.6. Ranking-only tasks do not
   care; we DO, because the policy turns probabilities into an EXPECTED call
   count, and a biased probability biases the expectation. A calibration check
   (a reliability curve, then isotonic/Platt via sklearn CalibratedClassifierCV
   if needed) is deferred until the curve says we need it.

5) GENERALIZATION + THE NON-CIRCULARITY GUARD (= FOUNDATION F1).
   The result is only honest if the model is evaluated on data it never trained
   on. We use k-fold cross-validation, and the synthetic generator's TRUE stock
   function is hidden from the model; it must LEARN structure from observed call
   pairs. The baseline to beat is proximity/random call order, NOT the
   generator's own prior. If the learned model cannot save >= 1 expected call vs
   that baseline on held-out requests, F1 fires and we reframe the project.

6) THE FLYWHEEL, IN ML TERMS.
   "More calls -> better predictions" is just "more labeled training rows ->
   lower variance / sharper P." v1 trains in batch (collect, then fit). Online
   incremental retraining is deferred (ROADMAP), but the loop is the same idea.

=============================================================================
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from .features import FEATURE_NAMES


class StockModel:
    """Wraps a histogram gradient-boosted tree classifier behind a tiny interface.

    We use sklearn's HistGradientBoostingClassifier (not xgboost) because it is
    the same algorithm family, needs no system OpenMP, and handles NaN features
    natively (we rely on that for missing recent_hit_rate). The rest of the
    system only ever calls `fit` and `predict_proba`, so the model is swappable
    without touching policy/ or evals/. Shallow trees + many iterations is the
    standard guard against overfitting small tabular data.
    """

    def __init__(
        self,
        seed: int = 7,
        max_depth: int = 3,
        max_iter: int = 200,
        learning_rate: float = 0.1,
    ) -> None:
        # The three hyperparameters are surfaced (not buried in fit) so the
        # auto-research loop can search over them; the defaults are the locked v1
        # values, so existing callers and results are unchanged.
        self.seed = seed
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.feature_names = FEATURE_NAMES
        self._model: HistGradientBoostingClassifier | None = None
        self._X = None
        self._y = None

    def fit(self, X, y) -> StockModel:
        """Train on design matrix X (rows of FEATURE_NAMES) and labels y (1/0)."""
        self._model = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        )
        self._model.fit(X, y)  # NaN routed natively to a learned default branch
        self._X, self._y = X, y
        return self

    def predict_proba(self, X) -> list[float]:
        """Return P(in_stock = yes) for each row of X. The policy's only input."""
        if self._model is None:
            raise RuntimeError("call fit() before predict_proba()")
        return self._model.predict_proba(X)[:, 1].tolist()

    def explain(self) -> dict[str, float]:
        """Permutation importance: how much shuffling each feature hurts predictions.

        More honest than split-count importance: it measures impact on actual
        predictive performance. Computed on the training set for the v1 memo;
        a held-out set would be stricter (deferred).
        """
        if self._model is None:
            raise RuntimeError("call fit() before explain()")
        result = permutation_importance(
            self._model, self._X, self._y, n_repeats=10, random_state=self.seed
        )
        return dict(zip(self.feature_names, map(float, result.importances_mean), strict=True))

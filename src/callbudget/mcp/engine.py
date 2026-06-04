"""Application service: the logic behind the MCP tools.

Wires data + prediction + policy + eval into three high-level operations. It is
plain and fully testable; the FastMCP transport (server.py) is a thin adapter
over this. Keeping logic here (not in the server) is the ports-and-adapters
split: the transport is swappable and the operations are unit-testable without
a live MCP client.

The engine trains once at construction on the real Boston scaffolding plus a
synthetic call history (real inputs, simulated outcomes; see FOUNDATION).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from ..config import DATA_DIR
from ..data.identity import resolve
from ..data.loaders import (
    load_arcos_throughput,
    load_boston_pharmacies,
    load_drug,
    load_shortage_calendar,
)
from ..data.store import ObservationStore
from ..evals.grade import _baseline_ecf, _order_ecf, reliability_panel
from ..policy.budget import order_calls
from ..predict.model import StockModel
from ..schemas import CallResult, StockObservation, StockState
from ..sim.callgen import generate_world
from ..sim.pharmacist import SimPharmacist
from ..voice.extract import keyword_extract
from ..voice.loop import simulate_text_call


class CallBudgetEngine:
    """Loads data, trains the stock model, and answers predict/plan/eval queries."""

    def __init__(
        self,
        limit: int = 28,
        history_per_pharmacy: int = 15,
        seed: int = 7,
        store_dir=None,
        model_kwargs: dict | None = None,
    ) -> None:
        # model_kwargs lets the auto-research loop build an engine with searched
        # hyperparameters (max_depth / max_iter / learning_rate); empty = locked v1.
        self.model_kwargs = model_kwargs or {}
        self.drug = load_drug()
        pharmacies = resolve(load_boston_pharmacies(limit=limit))
        load_arcos_throughput(pharmacies, seed=seed)
        self.shortage = load_shortage_calendar()
        self.world = generate_world(
            history_per_pharmacy=history_per_pharmacy,
            seed=seed,
            pharmacies=pharmacies,
            shortage_dates=self.shortage,
        )
        self.seed = seed
        self._by_npi = {p.npi: p for p in self.world.pharmacies}
        # The moat lives in a durable DuckDB + Parquet log, not in RAM. Features
        # are windowed SQL over it (tests/test_store.py proves SQL == features.py).
        store_dir = store_dir or (DATA_DIR / "observations" / "engine")
        self.store = ObservationStore(store_dir, self.world.pharmacies, min(self.shortage))
        self.store.load(self.world.observations)
        self._fit_and_refresh()

    def _fit_and_refresh(self) -> None:
        """Train on the observation log and recompute predictions (SQL features)."""
        X, y = self.store.feature_matrix()
        self.model = StockModel(seed=self.seed, **self.model_kwargs).fit(X, y)
        self.npis, request_X = self.store.request_features(self.world.as_of)
        self._prob = dict(zip(self.npis, self.model.predict_proba(request_X), strict=True))

    @property
    def n_observations(self) -> int:
        return self.store.count()

    def name_of(self, npi: str) -> str:
        return self._by_npi[npi].name

    def predict_stock(self) -> list[dict]:
        """Pharmacies ranked by predicted P(in_stock), highest first."""
        ranked = sorted(self.npis, key=lambda n: self._prob[n], reverse=True)
        return [
            {
                "npi": n,
                "name": self._by_npi[n].name,
                "zip": self._by_npi[n].zip,
                "chain": self._by_npi[n].chain,
                "p_in_stock": round(self._prob[n], 3),
            }
            for n in ranked
        ]

    def plan_calls(self, budget: int | None = None) -> dict:
        """Call order that minimizes expected calls-to-find, plus that ECF."""
        order = order_calls(self.npis, [self._prob[n] for n in self.npis])
        if budget:
            order = order[:budget]
        return {
            "drug": self.drug.drug_name,
            "dose": self.drug.dose,
            "ordered_npis": order,
            "expected_calls_to_find": round(_order_ecf(order, self.world.true_prob_today), 2),
        }

    def run_eval(self) -> dict:
        """ECF (naive vs learned) and the safety metrics, on this engine's world."""
        order = order_calls(self.npis, [self._prob[n] for n in self.npis])
        rel = reliability_panel(self.world.pharmacies, self.world.true_prob_today, trials=10)
        return {
            "n_pharmacies": len(self.npis),
            "ecf_naive": round(_baseline_ecf(self.npis, self.world.true_prob_today, seed=7), 2),
            "ecf_learned": round(_order_ecf(order, self.world.true_prob_today), 2),
            "false_positive_in_stock": round(rel["false_positive_in_stock"], 3),
            "abstention_rate": round(rel["abstention_rate"], 3),
        }

    def place_call(
        self,
        npi: str,
        style: str = "clear",
        seed: int = 0,
        extractor=keyword_extract,
        n_samples: int = 3,
        abstain_threshold: float = 0.67,
    ) -> CallResult:
        """Simulate one call. Ground truth is sampled from the (hidden) true
        probability for that pharmacy; style='hedge' forces an ambiguous answer.

        The extractor + self-consistency knobs are injectable so the auto-research
        loop can run the closed loop under a deliberately unreliable extractor
        (the regime where abstention earns its keep). Defaults reproduce the v1
        behaviour: the deterministic keyword extractor at the locked thresholds.
        """
        truth = random.Random(seed).random() < self.world.true_prob_today[npi]
        pharmacist = SimPharmacist(npi, truly_in_stock=truth, style=style, seed=seed)
        return simulate_text_call(
            pharmacist,
            self.drug,
            extractor=extractor,
            n_samples=n_samples,
            abstain_threshold=abstain_threshold,
        )

    def observe(self, result: CallResult) -> bool:
        """Learn from a call: a confident answer becomes a labeled observation and
        the model retrains. Abstained calls are skipped (re-call, not a label).
        Returns True if the call produced a label. This is one turn of the flywheel.
        """
        record = result.record
        if record.abstain or record.in_stock == StockState.UNKNOWN:
            return False
        as_of = self.world.as_of
        # Strictly increasing timestamp (noon + one second per prior observation):
        # gives every appended call a distinct, ordered observed_at so the windowed
        # features are deterministic instead of tied on a constant noon.
        observed_at = datetime(as_of.year, as_of.month, as_of.day, 12) + timedelta(
            seconds=self.store.count()
        )
        self.store.append(
            StockObservation(
                pharmacy_npi=result.pharmacy_npi,
                rxcui=self.drug.rxcui,
                drug_name=self.drug.drug_name,
                dose=self.drug.dose,
                observed_at=observed_at,
                in_stock=record.in_stock,
                confidence=record.confidence,
                source="voice_call",
            )
        )
        self._fit_and_refresh()
        return True

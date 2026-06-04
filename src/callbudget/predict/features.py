"""Feature engineering: turn raw facts into the model's input matrix X.

THEORY (features / the design matrix):
A model cannot consume a `Pharmacy` object or a calendar date. It consumes a
fixed-length vector of numbers, one row per (pharmacy, request, time) we want a
prediction for. The full table of those rows is the "design matrix" X; the
column we predict is y (in_stock, 1/0). Choosing the columns of X is feature
engineering, and for tabular problems it matters more than the model choice.

Our v1 columns and WHY each is plausibly predictive of "is this drug in stock":
  throughput_prior   higher-volume dispensers tend to stock more, more reliably.
  in_shortage        on the FDA/ASHP list -> every pharmacy's stock prob drops.
  dow                day of week: deliveries and dispensing cycle weekly.
  is_chain           chains restock differently from independents.
  recent_hit_rate    fraction of this pharmacy's PAST calls that were in-stock.
                     The flywheel as a number; the most powerful feature with data.
  days_since_seen    staleness of our newest observation for this pharmacy.

LEAKAGE WARNING (the cardinal sin): never let a feature encode the answer.
`recent_hit_rate` and `days_since_seen` are computed ONLY from observations
STRICTLY BEFORE the row's timestamp, or the model "cheats" in eval then fails in
the wild. `_history_before` enforces that ordering.

MISSING VALUES: a pharmacy with no prior calls has no recent_hit_rate. We encode
that as NaN (not 0.5), because xgboost handles missing values natively, learning
its own default branch direction. Encoding "unknown" as a real number would lie.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from ..schemas import Pharmacy, StockObservation, StockState

FEATURE_NAMES: list[str] = [
    "throughput_prior",
    "in_shortage",
    "dow",
    "is_chain",
    "recent_hit_rate",
    "days_since_seen",
]


def _history_before(observations: list[StockObservation], npi: str, before: date):
    """This pharmacy's observations strictly before `before`, oldest first."""
    hist = [o for o in observations if o.pharmacy_npi == npi and o.observed_at.date() < before]
    return sorted(hist, key=lambda o: o.observed_at)


def _recent_hit_rate(hist: list[StockObservation]) -> float:
    if not hist:
        return float("nan")
    return sum(o.in_stock == StockState.YES for o in hist) / len(hist)


def _days_since_seen(hist: list[StockObservation], as_of: date) -> float:
    if not hist:
        return float("nan")
    return float((as_of - hist[-1].observed_at.date()).days)


def _row(ph: Pharmacy, as_of: date, in_shortage: bool, hist: list[StockObservation]) -> list[float]:
    return [
        ph.throughput_prior if ph.throughput_prior is not None else float("nan"),
        1.0 if in_shortage else 0.0,
        float(as_of.weekday()),
        1.0 if ph.chain else 0.0,
        _recent_hit_rate(hist),
        _days_since_seen(hist, as_of),
    ]


def build_training_matrix(observations, pharmacies, shortage_dates):
    """One labeled row per observation; features use only that obs's PAST. -> (X, y)."""
    ph_by_npi = {p.npi: p for p in pharmacies}
    X, y = [], []
    for o in observations:
        d = o.observed_at.date()
        hist = _history_before(observations, o.pharmacy_npi, d)
        X.append(_row(ph_by_npi[o.pharmacy_npi], d, d in shortage_dates, hist))
        y.append(1 if o.in_stock == StockState.YES else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def build_request_features(pharmacies, observations, as_of, shortage_dates):
    """Features for every candidate pharmacy as of the request day. -> (npis, X)."""
    in_shortage = as_of in shortage_dates
    npis, X = [], []
    for ph in pharmacies:
        hist = _history_before(observations, ph.npi, as_of)
        npis.append(ph.npi)
        X.append(_row(ph, as_of, in_shortage, hist))
    return npis, np.array(X, dtype=float)

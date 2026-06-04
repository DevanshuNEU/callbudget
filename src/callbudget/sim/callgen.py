"""Synthetic call-log generator: manufacture realistic StockObservation history.

THEORY: the model needs a history of past calls to learn from, but we have no
real call data. So we define a HIDDEN true stock function over (pharmacy, date)
that depends on real-data-shaped priors (ARCOS throughput, FDA/ASHP shortage
windows, weekly dynamics) plus noise, then sample binary calls from it. The model
later tries to RECOVER that hidden structure from observed (features, in_stock)
pairs.

NON-CIRCULARITY (FOUNDATION F1): the model NEVER sees `_true_stock_prob`. It only
sees Bernoulli draws (YES/NO) and must generalize via features. Evaluation uses
the noise-free true probability, which the model also never sees. The baseline to
beat is random call order, not this prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from ..schemas import Pharmacy, StockObservation, StockState

# Weekly dynamics: stock tends to be higher mid-week (after delivery cycles).
_DOW_BOOST = {0: 0.00, 1: 0.06, 2: 0.08, 3: 0.04, 4: 0.00, 5: -0.04, 6: -0.06}


def _true_stock_prob(reliability: float, dow: int, noise: float = 0.0) -> float:
    """The HIDDEN ground-truth probability the drug is in stock. Model never sees it.

    During a shortage, a pharmacy's underlying `reliability` (a heavy-tailed
    latent: most pharmacies rarely have it, a few reliably do) is the dominant
    driver, modulated by weekly delivery dynamics. The model never receives
    `reliability` as a feature. It only sees:
      - throughput_prior: a NOISY observable proxy of reliability (available now)
      - recent_hit_rate: an estimate of reliability from PAST calls (sharpens
        with more history -> the flywheel)
    so call history is what lets it find the few good pharmacies.
    """
    return float(np.clip(reliability + _DOW_BOOST[dow] + noise, 0.01, 0.97))


def _reliability_from_throughput(
    t: float, rng, a: float = 0.55, sigma_e: float = 0.18, base: float = 0.05
) -> float:
    """Hidden per-pharmacy stock propensity.

    Partly EXPLAINED by throughput (observable, available now), partly
    IDIOSYNCRATIC (e) that ONLY call history can reveal. The idiosyncratic part
    is exactly what makes recent_hit_rate add signal beyond throughput, i.e. the
    flywheel: with no calls you only have the throughput proxy; with history you
    recover e too and rank the few good pharmacies correctly.
    """
    e = float(rng.normal(0, sigma_e))
    return float(np.clip(a * t + e + base, 0.01, 0.97))


@dataclass
class SyntheticWorld:
    pharmacies: list[Pharmacy]
    observations: list[StockObservation]
    shortage_dates: set[date]
    true_prob_today: dict[str, float]  # HIDDEN; eval only
    as_of: date


def generate_world(
    n_pharmacies: int = 60,
    history_per_pharmacy: int = 7,
    seed: int = 7,
    drug: str = "Adderall XR",
    dose: str = "20 mg",
    as_of: date = date(2026, 6, 1),
    shortage_window_days: int = 120,
    pharmacies: list[Pharmacy] | None = None,
    shortage_dates: set[date] | None = None,
) -> SyntheticWorld:
    """Build the world. Pass real `pharmacies` (with throughput_prior set) + a real
    `shortage_dates` calendar for the real-data path; omit them for the
    pure-synthetic path. Either way the in-stock OUTCOMES are sampled (that data
    does not exist), and hidden reliability is derived from throughput + idiosyncrasy.
    """
    rng = np.random.default_rng(seed)
    if shortage_dates is None:
        shortage_dates = {as_of - timedelta(days=d) for d in range(shortage_window_days)}

    if pharmacies is None:
        # Pure-synthetic pharmacies with ARCOS-shaped (heavy-tailed) throughput.
        chains = ["CVS", "Walgreens", "Rite Aid", None]
        pharmacies = []
        for i in range(n_pharmacies):
            v = float(rng.lognormal(0.0, 1.1))
            throughput = float(np.clip(v / (v + 3.0), 0.01, 0.99))
            pharmacies.append(
                Pharmacy(
                    npi=str(1_000_000_000 + i),
                    name=f"Pharmacy {i}",
                    chain=chains[int(rng.integers(0, len(chains)))],
                    zip="021xx",
                    throughput_prior=throughput,
                )
            )

    reliability: dict[str, float] = {}  # HIDDEN latent; never a feature
    for ph in pharmacies:
        t = ph.throughput_prior if ph.throughput_prior is not None else 0.2
        r = _reliability_from_throughput(t, rng)
        if ph.chain:
            r = float(np.clip(r + 0.03, 0.01, 0.97))  # chains a touch more reliable
        reliability[ph.npi] = r

    observations: list[StockObservation] = []
    for ph in pharmacies:
        r = reliability[ph.npi]
        offsets = sorted(
            rng.choice(range(1, shortage_window_days), size=history_per_pharmacy, replace=False)
        )
        for off in offsets:
            d = as_of - timedelta(days=int(off))
            p = _true_stock_prob(r, d.weekday(), noise=float(rng.normal(0, 0.03)))
            observations.append(
                StockObservation(
                    pharmacy_npi=ph.npi,
                    rxcui=None,
                    drug_name=drug,
                    dose=dose,
                    observed_at=datetime(d.year, d.month, d.day, 12, 0, 0),
                    in_stock=StockState.YES if rng.random() < p else StockState.NO,
                    confidence=1.0,
                    source="synthetic",
                )
            )

    # Noise-free true probability for the request day (for stable ECF evaluation).
    true_prob_today = {
        ph.npi: _true_stock_prob(reliability[ph.npi], as_of.weekday()) for ph in pharmacies
    }
    return SyntheticWorld(pharmacies, observations, shortage_dates, true_prob_today, as_of)

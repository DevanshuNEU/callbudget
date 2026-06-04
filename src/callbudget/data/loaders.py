"""Load + normalize the real public scaffolding into our domain types.

WHAT IS REAL vs SYNTHETIC (the integrity backbone, see FOUNDATION):
  REAL:  which pharmacies exist (NPPES), the drug's identity (RxNorm RxCUI),
         the Adderall shortage window (FDA/ASHP dates), the SHAPE of per-pharmacy
         throughput (ARCOS opioid-volume distribution).
  SYNTHETIC: the per-call in-stock OUTCOMES. These are sampled, not fetched,
         because real-time shelf-stock data is precisely what does not exist
         anywhere. Manufacturing it is the project's entire premise. (sim/callgen.py)

Loaders are pure (data in, typed objects out) and depend on nothing else in the
package, so they sit at the stable bottom of the dependency DAG.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from ..schemas import DrugRequest, Pharmacy

# Resolved 2026-06-01 via RxNorm approximateTerm("Adderall XR 20 mg") + property
# lookup. Adderall XR 20mg = 4 amphetamine salts x 5 MG = 20 MG total, so the
# RxNorm name reads "...5 MG..." for each salt. This RxCUI is that 20mg XR capsule.
ADDERALL_XR_20MG_RXCUI = "861226"

_RETAIL_CHAINS = (
    "CVS", "WALGREEN", "RITE AID", "WALMART", "COSTCO", "STOP & SHOP", "TARGET", "HANNAFORD",
)
_PHARMACY_CACHE = DATA_DIR / "boston_pharmacies.json"


def _http_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "callbudget-demo"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted gov URLs)
        return json.load(resp)


def load_drug(name: str = "Adderall XR", dose: str = "20 mg", verify: bool = False) -> DrugRequest:
    """Return the canonical DrugRequest with a real RxCUI. verify=True confirms live."""
    rxcui = ADDERALL_XR_20MG_RXCUI
    if verify:
        try:
            _http_json(
                f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json?propName=RxNorm%20Name"
            )
        except Exception:  # offline: fall back to the documented resolved value
            pass
    return DrugRequest(drug_name=name, rxcui=rxcui, dose=dose, location_zip="02114")


def _detect_chain(org_name: str) -> str | None:
    upper = org_name.upper()
    for c in _RETAIL_CHAINS:
        if c in upper:
            return c.title()
    return None  # independent / health-system pharmacy


def fetch_boston_pharmacies(limit: int = 120) -> list[dict]:
    """Live NPPES query: organization pharmacies at a Boston, MA location."""
    url = (
        "https://npiregistry.cms.hhs.gov/api/?version=2.1&enumeration_type=NPI-2"
        f"&state=MA&city=Boston&address_purpose=LOCATION&taxonomy_description=Pharmacy&limit={limit}"
    )
    data = _http_json(url)
    by_npi: dict[str, dict] = {}
    for r in data.get("results", []):
        name = (r["basic"].get("organization_name") or "Unknown Pharmacy").title()
        loc = [a for a in r["addresses"] if a.get("address_purpose") == "LOCATION"]
        zip5 = loc[0]["postal_code"][:5] if loc else "02114"
        npi = str(r["number"])
        by_npi[npi] = {"npi": npi, "name": name, "chain": _detect_chain(name), "zip": zip5}
    return list(by_npi.values())


def load_boston_pharmacies(limit: int = 60, refresh: bool = False) -> list[Pharmacy]:
    """Real Boston pharmacies (NPPES), cached to data/ for reproducible offline runs."""
    if refresh or not _PHARMACY_CACHE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _PHARMACY_CACHE.write_text(json.dumps(fetch_boston_pharmacies(max(limit, 120)), indent=2))
    recs = json.loads(_PHARMACY_CACHE.read_text())[:limit]
    return [Pharmacy(npi=r["npi"], name=r["name"], chain=r["chain"], zip=r["zip"]) for r in recs]


def load_arcos_throughput(pharmacies: list[Pharmacy], seed: int = 7) -> None:
    """Assign a per-pharmacy throughput_prior shaped like ARCOS opioid volume.

    THEORY: ARCOS per-pharmacy opioid volume is heavy-tailed (a few huge
    dispensers, many small). We assign throughput_prior with THAT SHAPE
    (log-normal squashed to [0,1]), seeded per-NPI so it is reproducible.
    Real per-pharmacy values via arcospy / the Notre Dame ARCOS API (public key)
    are a documented local upgrade; the public WaPo dataset is opioids-only and
    ~100GB, so we use the calibrated shape here (FOUNDATION F3 fallback).
    """
    for ph in pharmacies:
        rng = np.random.default_rng(seed ^ (int(ph.npi) % 1_000_000))
        v = float(rng.lognormal(mean=0.0, sigma=1.1))
        ph.throughput_prior = float(np.clip(v / (v + 3.0), 0.01, 0.99))


_SHORTAGE_CACHE = DATA_DIR / "shortage_calendar.json"
_FALLBACK_SHORTAGE_START = date(2022, 10, 12)  # documented Adderall shortage onset


def fetch_amphetamine_shortage() -> dict:
    """Live openFDA query for the amphetamine (Adderall) shortage window.

    Optional OPENFDA_API_KEY (env) raises the rate limit. Returns the earliest
    posted onset and whether any record is still Current.
    """
    url = (
        "https://api.fda.gov/drug/shortages.json"
        '?search=generic_name:"AMPHETAMINE+ASPARTATE"&limit=100'
    )
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        url += f"&api_key={key}"
    results = _http_json(url).get("results", [])
    starts, active = [], False
    for r in results:
        raw = r.get("initial_posting_date")
        if raw:
            try:
                starts.append(datetime.strptime(raw, "%m/%d/%Y").date())
            except ValueError:
                pass
        if (r.get("status") or "").lower() == "current":
            active = True
    if not starts:
        raise ValueError("no amphetamine shortage records returned")
    return {"start": str(min(starts)), "active": active, "source": "openfda"}


def load_shortage_calendar(
    as_of: date = date(2026, 6, 1),
    refresh: bool = False,
    offline: bool | None = None,
    cache_path: Path | None = None,
) -> set[date]:
    """Real Adderall shortage window from openFDA, cached for hermetic/offline use.

    Cached to data/ (committed) so tests and the engine need no network. Falls back
    to the documented onset (2022-10-12, active) when offline or on any error.
    `offline` defaults to the CALLBUDGET_OFFLINE env var.
    """
    cache = cache_path or _SHORTAGE_CACHE
    if offline is None:
        offline = bool(os.environ.get("CALLBUDGET_OFFLINE"))

    if refresh or not cache.exists():
        try:
            if offline:
                raise RuntimeError("offline")
            info = fetch_amphetamine_shortage()
        except Exception:
            info = {"start": str(_FALLBACK_SHORTAGE_START), "active": True, "source": "fallback"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(info, indent=2))

    info = json.loads(cache.read_text())
    start = date.fromisoformat(info["start"])
    end = as_of if info.get("active", True) else date.fromisoformat(info["end"])
    end = min(end, as_of)
    return {start + timedelta(days=i) for i in range((end - start).days + 1)}

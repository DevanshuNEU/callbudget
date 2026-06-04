"""Data-layer tests, all offline (use the committed pharmacy cache; no network).

Guards the real scaffolding loaders + the honesty boundary (real inputs,
simulated outcomes) without depending on live APIs in CI.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from callbudget.data.identity import resolve
from callbudget.data.loaders import (
    load_arcos_throughput,
    load_boston_pharmacies,
    load_drug,
    load_shortage_calendar,
)
from callbudget.evals.grade import ecf_curve
from callbudget.schemas import Pharmacy


def test_shortage_calendar_offline_falls_back():
    # offline (via the param; defaults to the CALLBUDGET_OFFLINE env var) -> the
    # documented window, written to a throwaway cache so the committed one is untouched.
    tmp = Path(__file__).resolve().parents[1] / "data" / "observations" / "_test_shortage.json"
    tmp.unlink(missing_ok=True)
    cal = load_shortage_calendar(as_of=date(2026, 6, 1), refresh=True, offline=True, cache_path=tmp)
    assert date(2023, 1, 1) in cal and date(2020, 1, 1) not in cal
    assert json.loads(tmp.read_text())["source"] == "fallback"


def test_drug_resolves_to_real_rxcui():
    d = load_drug()  # verify=False -> no network
    assert d.rxcui == "861226"
    assert d.dose == "20 mg"


def test_shortage_calendar_window():
    cal = load_shortage_calendar(as_of=date(2026, 6, 1))
    assert date(2023, 1, 1) in cal  # inside the shortage window
    assert date(2026, 6, 1) in cal  # includes as_of
    assert date(2020, 1, 1) not in cal  # before the shortage began


def test_boston_pharmacies_from_cache_are_wellformed():
    phs = load_boston_pharmacies(limit=20)  # served from the committed cache
    assert len(phs) > 0
    assert all(len(p.zip) == 5 and p.npi.isdigit() for p in phs)


def test_arcos_throughput_bounded_and_reproducible():
    a = load_boston_pharmacies(limit=15)
    b = load_boston_pharmacies(limit=15)
    load_arcos_throughput(a, seed=7)
    load_arcos_throughput(b, seed=7)
    assert all(0.0 < p.throughput_prior < 1.0 for p in a)
    # Seeded per-NPI -> reproducible across loads.
    assert [p.throughput_prior for p in a] == [p.throughput_prior for p in b]


def test_identity_dedups_same_name_and_zip():
    dupes = [
        Pharmacy(npi="1", name="CVS Pharmacy", zip="02114"),
        Pharmacy(npi="2", name="CVS Pharmacy", zip="02114"),
        Pharmacy(npi="3", name="CVS Pharmacy", zip="02116"),
    ]
    assert len(resolve(dupes)) == 2  # same (name, zip) collapses; different zip stays


def test_ecf_curve_smoke():
    phs = load_boston_pharmacies(limit=12)
    load_arcos_throughput(phs)
    shortage = load_shortage_calendar()
    levels, baseline, learned = ecf_curve([5], phs, shortage, seeds=(1, 2))
    assert levels == [5]
    assert baseline[0] > 0 and learned[0] > 0  # finite, positive ECFs

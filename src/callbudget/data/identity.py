"""Pharmacy identity resolution: collapse duplicate records to one node.

THEORY: NPPES can list multiple NPIs for what is, on the ground, one pharmacy
location (re-enrollments, billing vs dispensing NPIs). Without dedup, the same
physical store fragments across rows and its stock time-series splits, leaking
signal. Entity resolution is the unglamorous-but-mandatory plumbing of any
data-aggregation business.

v1 scope (locked): a lightweight dedup over the curated Boston subset, keying on
normalized (name, zip). Production-grade resolution (phone, geocode, fuzzy name)
is deferred.
"""

from __future__ import annotations

from ..schemas import Pharmacy


def resolve(pharmacies: list[Pharmacy]) -> list[Pharmacy]:
    """Keep one canonical Pharmacy per normalized (name, zip)."""
    seen: dict[tuple[str, str], Pharmacy] = {}
    for ph in pharmacies:
        key = (ph.name.strip().lower(), ph.zip)
        seen.setdefault(key, ph)
    return list(seen.values())

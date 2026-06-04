"""Simulated pharmacist: a callee with a hidden ground truth, separate from how
it talks.

Truth (`truly_in_stock`) and speaking style are independent on purpose: a
pharmacist who hedges ("let me check") may or may not actually have the drug.
Keeping them separate is what lets the eval grade extraction honestly, including
the case where an ambiguous answer sits on top of a real yes or a real no.

Zero PHI: the answers are templated, deterministic given a seed.
"""

from __future__ import annotations

import random

# Verbal templates. Real pharmacists are terse, qualify ("the 20s"), and hedge.
_YES = [
    "yeah we got the 20 XR in stock, you can come grab it",
    "yep we have the Adderall XR 20s on the shelf right now",
]
_PARTIAL = [
    "we got the generic 10s but the 20 XR is on backorder til next week",
    "the generic's in but the brand 20 XR, that's on back order",
]
_NO = [
    "no we're out of the XR, been out for like two weeks",
    "we don't have the 20 XR, nothing came in on the last order",
]
_HEDGE = [
    "uh let me check in the back, i'm not sure we have the 20s right now",
    "hmm i think so? you might wanna call back, lemme check",
]


class SimPharmacist:
    """Holds the hidden truth and emits a messy answer in a given style."""

    def __init__(
        self, pharmacy_npi: str, truly_in_stock: bool, style: str = "clear", seed: int = 7
    ) -> None:
        if style not in ("clear", "hedge"):
            raise ValueError("style must be 'clear' or 'hedge'")
        self.pharmacy_npi = pharmacy_npi
        self.truly_in_stock = truly_in_stock  # hidden ground truth (for grading)
        self.style = style
        self._rng = random.Random(seed)

    def answer(self, question: str) -> str:
        """A messy natural-language answer. 'hedge' is ambiguous regardless of truth."""
        if self.style == "hedge":
            return self._rng.choice(_HEDGE)
        if self.truly_in_stock:
            return self._rng.choice(_YES + _PARTIAL)
        return self._rng.choice(_NO)

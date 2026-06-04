"""Shared domain model: the vocabulary every CallBudget module agrees on.

This is the "shared kernel." If `sim/` and `predict/` each invented their own
idea of what a call result is, they would silently drift apart. Defining the
nouns once, here, prevents that. Every other module imports from this file and
never redefines these types.

Pydantic v2 gives us: validation at the boundary (bad data fails loudly, early),
JSON (de)serialization for free (call logs are JSONL), and types that double as
documentation.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class StockState(StrEnum):
    """The answer to "do you have this drug in stock right now?"

    PARTIAL exists on purpose: the messiest and most common real answer is
    "we have the generic 10s but the 20 XR is on backorder." Forcing that into
    yes/no would destroy information. UNKNOWN is what we record when the agent
    could not get a clear answer (it pairs with abstention downstream).
    """

    YES = "yes"
    NO = "no"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class DrugRequest(BaseModel):
    """What a patient wants. The input to the whole system.

    Canonical demo instance: Adderall XR, 20 mg, Boston. `rxcui` is RxNorm's
    stable numeric id for the drug concept; carrying it (not just a name string)
    is what lets us join across NPPES / openFDA / shortage data without ambiguity.
    """

    drug_name: str
    rxcui: str | None = None
    dose: str
    location_zip: str
    radius_km: float = 10.0


class Pharmacy(BaseModel):
    """A node we can call. Identity-resolved from NPPES (Boston subset for v1)."""

    npi: str  # National Provider Identifier, the canonical id
    name: str
    chain: str | None = None  # CVS / Walgreens / independent / ...
    zip: str
    lat: float | None = None
    lon: float | None = None
    # ARCOS-derived per-pharmacy throughput prior (opioid volume as a same-DEA-
    # schedule PROXY for general controlled-substance throughput; NOT an
    # Adderall-specific signal). See FOUNDATION.md principle 6.
    throughput_prior: float | None = None


class StockObservation(BaseModel):
    """The data exhaust: one labeled, timestamped, store-level stock fact.

    This is THE asset. Millions of these are the moat. Each call produces one.
    `confidence` and the StockState gate whether this observation is clean enough
    to become a training label downstream (low confidence -> "re-call", not a
    poisoned label).
    """

    pharmacy_npi: str
    rxcui: str | None
    drug_name: str
    dose: str
    observed_at: datetime
    in_stock: StockState
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "voice_call"  # voice_call | manual | synthetic


class ExtractionRecord(BaseModel):
    """What the voice extractor emits from a (noisy) spoken pharmacist answer.

    This is the structured output of STT -> LLM -> TTS + extraction. `abstain`
    and `evidence_span` are the reliability mechanism: the model must quote the
    verbatim phrase it grounded on, and refuse (abstain) when that phrase is
    ambiguous, rather than guessing. The safety metric (false-positive-in-stock)
    is computed from these.
    """

    drug: str
    dose: str
    form: str | None = None  # tablet | capsule | XR | liquid | null
    brand_vs_generic: str = "unknown"  # brand | generic | either | unknown
    in_stock: StockState = StockState.UNKNOWN
    backorder_date: date | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    abstain: bool = False
    evidence_span: str = ""  # verbatim quote justifying the fields


class CallResult(BaseModel):
    """A single placed call: which pharmacy, what we extracted, and metadata."""

    pharmacy_npi: str
    request: DrugRequest
    record: ExtractionRecord
    transcript: str = ""
    latency_ms: float | None = None


class CallPlan(BaseModel):
    """The policy's output: the order to call, and the cost we expect to pay.

    `expected_calls_to_find` (ECF) is the hero metric. The whole project is
    "make this number small, and watch it shrink as labeled data accumulates."
    """

    request: DrugRequest
    ordered_npis: list[str]
    expected_calls_to_find: float | None = None

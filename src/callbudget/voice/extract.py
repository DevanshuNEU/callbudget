"""Structured extraction + calibrated abstention from a (noisy) spoken answer.

THEORY (information extraction under noise + confidence via self-consistency):
The STT transcript is imperfect and the pharmacist's phrasing is messy. We must
turn it into an ExtractionRecord, and crucially KNOW WHEN WE ARE UNSURE.

Two abstention pathways:
  1. AMBIGUITY DETECTION: hedging language ("let me check", "not sure") or no
     clear stock signal -> the single parse abstains.
  2. SELF-CONSISTENCY: run the extractor several times (with a real LLM at
     temperature > 0 the parses vary); if they DISAGREE on the answer, abstain.
     This is more honest than asking a model to self-report a probability, which
     models are badly calibrated at. confidence = fraction of runs that agree.

DEPENDENCY INJECTION: `extract()` takes the `extractor` callable as an argument.
`keyword_extract` is a deterministic, zero-dependency parser so the whole thing
runs and is testable with no API key. `make_claude_extractor()` returns the real
LLM-backed callable for the live demo. The logic never hard-codes a vendor.

The point of abstention is LABEL QUALITY: a low-confidence call becomes
"re-call", not a training label, which protects the model from noise and is what
drives the safety metric (false-positive-in-stock) down.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from ..schemas import ExtractionRecord, StockState

Extractor = Callable[[str], dict]

_HEDGES = ("let me check", "not sure", "i think", "maybe", "call back", "don't know", "lemme check")
_YES = ("in stock", "we have", "we've got", "we got", "got the", "yeah we have", "should have")
_NO = ("out of stock", "don't have", "do not have", "we're out", "none in", "out of the")


def keyword_extract(transcript: str) -> dict:
    """Deterministic, no-dependency parser. Good enough to demo + test offline."""
    t = transcript.lower()
    hedged = any(h in t for h in _HEDGES)
    backorder = "backorder" in t or "back order" in t
    has_generic = "generic" in t

    if backorder and has_generic:
        in_stock, bvg = "partial", "generic"  # generic in, the asked form on backorder
    elif backorder:
        in_stock, bvg = "no", "unknown"
    elif any(p in t for p in _YES):
        in_stock, bvg = "yes", ("generic" if has_generic else "unknown")
    elif any(p in t for p in _NO):
        in_stock, bvg = "no", "unknown"
    else:
        in_stock, bvg = "unknown", "unknown"

    abstain = hedged or in_stock == "unknown"
    return {
        "drug": "Adderall XR",
        "dose": "20 mg",
        "in_stock": in_stock,
        "brand_vs_generic": bvg,
        "abstain": abstain,
        "evidence_span": transcript.strip()[:160],
    }


def _ask_context(drug: str | None, dose: str | None) -> str:
    """The question the pharmacist is answering. Giving the extractor this context
    is what stops it over-abstaining: out of context "yeah we have the 20s" is
    ambiguous, but anchored to "do you have Adderall XR 20 mg?" it is a clear yes."""
    if not drug:
        return ""
    asked = f"{drug} {dose}".strip() if dose else drug
    return (
        f'A pharmacy was asked: "do you have {asked} in stock?" The answer below '
        f"refers to that drug and dose. "
    )


def make_claude_extractor(
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.4,
    drug: str | None = None,
    dose: str | None = None,
) -> Extractor:
    """Return the real LLM-backed extractor (needs ANTHROPIC_API_KEY).

    Not exercised in CI; this is the live path. Temperature > 0 is intentional:
    it is what makes self-consistency meaningful (independent parses can differ).
    Pass drug/dose so the extractor knows the question it is grounding against.
    """

    def _extract(transcript: str) -> dict:
        import json

        from anthropic import Anthropic

        client = Anthropic()
        prompt = (
            f"{_ask_context(drug, dose)}"
            "Extract pharmacy stock info from this phone transcript as strict JSON with keys "
            "drug, dose, form, brand_vs_generic, in_stock (yes|no|partial|unknown), "
            "backorder_date (ISO8601 or null), abstain (bool), evidence_span (verbatim quote). "
            "Set abstain=true and in_stock=unknown only if the answer is genuinely ambiguous "
            "about availability.\n\n"
            f"Answer: {transcript}"
        )
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Claude often wraps JSON in markdown fences or a sentence of prose, so a
        # bare json.loads on the whole reply breaks. Extract the outermost object
        # instead; if there is none, abstain (the safe default). This fragility is
        # exactly why make_instructor_extractor (schema-validated) exists.
        text = msg.content[0].text
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return {"in_stock": "unknown", "abstain": True, "evidence_span": ""}
        return json.loads(text[start : end + 1])

    return _extract


def make_instructor_extractor(
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.4,
    drug: str | None = None,
    dose: str | None = None,
) -> Extractor:
    """Return an extractor that gets a schema-validated ExtractionRecord via instructor.

    Not exercised in CI; this is the live path (needs the [agent] extra and
    ANTHROPIC_API_KEY). Backward compatible with `make_claude_extractor`: it returns
    the same `Extractor` callable (transcript -> dict), so `extract()`'s
    self-consistency machinery works unchanged.

    THEORY (schema-validated structured outputs vs hand-parsing JSON):
    `make_claude_extractor` does `json.loads(msg.content[0].text)`, which is fragile
    on three axes: the model can emit prose around the JSON, emit a key with the
    wrong type, or omit a required field, and each failure surfaces late as a raw
    parse error or a silently-wrong record. `instructor` patches the Anthropic client
    to take a Pydantic `response_model` (our ExtractionRecord) and validates the
    model's output AGAINST that schema, retrying on validation failure. The contract
    moves from "parse whatever text came back" to "the LLM must return an object that
    satisfies ExtractionRecord or the call fails loudly," which is the same
    validate-at-the-boundary discipline the rest of the codebase already uses.

    Temperature > 0 is intentional, same as the plain Claude extractor: it is what
    makes self-consistency meaningful (independent parses can differ).
    """

    def _extract(transcript: str) -> dict:
        import instructor
        from anthropic import Anthropic

        client = instructor.from_anthropic(Anthropic())
        prompt = (
            f"{_ask_context(drug, dose)}"
            "Extract pharmacy stock info from this answer. Set abstain=true and "
            "in_stock=unknown only if the answer is genuinely ambiguous about "
            "availability, and quote the verbatim phrase you grounded on in "
            f"evidence_span.\n\nAnswer: {transcript}"
        )
        record: ExtractionRecord = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=temperature,
            response_model=ExtractionRecord,
            messages=[{"role": "user", "content": prompt}],
        )
        # extract() consumes dicts (it is vendor-agnostic); hand back the validated
        # record as a dict so the self-consistency path is identical to every other
        # extractor.
        return record.model_dump()

    return _extract


def extract(
    transcript: str,
    extractor: Extractor = keyword_extract,
    n_samples: int = 3,
    abstain_threshold: float = 0.67,
) -> ExtractionRecord:
    """Self-consistency extraction. Abstains on disagreement or detected ambiguity."""
    samples = [extractor(transcript) for _ in range(n_samples)]
    # `or "unknown"` (not get's default) because an LLM can emit an explicit null
    # for in_stock; a missing key and a null both mean "no clear answer" here.
    votes = Counter((s.get("in_stock") or "unknown") for s in samples)
    top_value, top_count = votes.most_common(1)[0]
    agreement = top_count / n_samples
    rep = next(s for s in samples if (s.get("in_stock") or "unknown") == top_value)

    abstain = (
        agreement < abstain_threshold
        or top_value == "unknown"
        or bool(rep.get("abstain"))
        or not rep.get("evidence_span")
    )
    # `or default` coerces explicit nulls (which LLMs emit for absent fields) to the
    # schema's string defaults; keyword_extract never emits null, so it is unaffected.
    return ExtractionRecord(
        drug=rep.get("drug") or "",
        dose=rep.get("dose") or "",
        form=rep.get("form"),
        brand_vs_generic=rep.get("brand_vs_generic") or "unknown",
        in_stock=StockState.UNKNOWN if abstain else StockState(top_value),
        backorder_date=rep.get("backorder_date"),
        confidence=round(agreement, 3),
        abstain=abstain,
        evidence_span=rep.get("evidence_span") or "",
    )

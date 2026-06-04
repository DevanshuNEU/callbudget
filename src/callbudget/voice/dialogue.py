"""The dialogue brain: one conversation engine that every surface drives.

THEORY (one brain, many transports):
A voice agent's intelligence is its dialogue policy: what to say next, when to ask
a clarifying question, when it is confident enough to record an answer and close the
call. That policy is text in, text out, and it must not live in any transport. So it
lives here, once. The CLI, the web app, the MCP server, and the Pipecat voice loop
are thin adapters that feed it the conversation so far and speak or print what it
returns. This is the same ports-and-adapters split the prediction engine already
uses: one core, swappable surfaces. Because the brain is pure text, it is
deterministically testable with no audio.

Two implementations behind one entrypoint (`respond`):
- live (ANTHROPIC_API_KEY set): Claude drives a natural conversation.
- offline (no key): a deterministic state machine (open, clarify once on ambiguity,
  then confirm or give up). Keeps CI green and the demo runnable with no key.

Both OPEN with the AI disclosure + recording-consent line (bot-disclosure and
call-recording norms), and turn uncertainty into a follow-up question rather than a
dead-end abstain.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from ..schemas import StockState


class Message(BaseModel):
    """One line of the call so far."""

    role: str  # "agent" | "pharmacist"
    text: str


class Turn(BaseModel):
    """The agent's next move: what to say, the record so far, whether the call is done."""

    reply: str
    done: bool
    source: str  # "claude" | "offline"
    in_stock: StockState = StockState.UNKNOWN
    confidence: float = 0.0
    abstain: bool = True
    brand_vs_generic: str = "unknown"
    evidence_span: str = ""

    @property
    def record(self) -> dict:
        """The structured stock record, in the shape every surface renders."""
        return {
            "in_stock": self.in_stock.value,
            "confidence": self.confidence,
            "abstain": self.abstain,
            "brand_vs_generic": self.brand_vs_generic,
            "evidence_span": self.evidence_span,
        }

    @property
    def verdict(self) -> str:
        if self.abstain:
            return "re-call (not confident, no label stored)"
        return f"learn (store a labeled observation: in_stock={self.in_stock.value})"


def respond(
    history: list, drug: str = "Adderall XR", dose: str = "20 mg", caller: str = "Devanshu"
) -> Turn:
    """One dialogue turn. Uses the live brain when ANTHROPIC_API_KEY is set, else offline.

    `history` is a list of Message or of {role, text} dicts. Live errors propagate so
    each adapter can surface them (the web returns 502, the CLI prints, etc.)."""
    msgs = [m if isinstance(m, Message) else Message(**m) for m in history]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _respond_live(msgs, drug, dose, caller)
    return _respond_offline(msgs, drug, dose, caller)


def _opener(drug: str, dose: str, caller: str) -> str:
    """Always disclose AI + recording consent before asking the question."""
    return (
        f"Hi, I'm an AI assistant calling on behalf of {caller}, and this call may be "
        f"recorded. Do you have {drug} {dose} in stock?"
    )


def _respond_offline(msgs: list[Message], drug: str, dose: str, caller: str) -> Turn:
    """Deterministic dialogue: open, clarify once on an ambiguous answer, then confirm
    or give up. No key, no network: this is the CI-tested path."""
    from .extract import extract, keyword_extract

    said = [m for m in msgs if m.role == "pharmacist"]
    if not said:
        return Turn(reply=_opener(drug, dose, caller), done=False, source="offline")
    rec = extract(said[-1].text, extractor=keyword_extract, n_samples=1)
    fields = {
        "in_stock": rec.in_stock, "confidence": rec.confidence, "abstain": rec.abstain,
        "brand_vs_generic": rec.brand_vs_generic, "evidence_span": rec.evidence_span,
    }
    if not rec.abstain:
        return Turn(reply=f"Got it, so that's {rec.in_stock.value}. Thank you.",
                    done=True, source="offline", **fields)
    if len(said) < 2:  # one clarifying question, then stop to avoid looping
        return Turn(reply=f"Just to confirm, do you have the {drug} {dose} in stock right now?",
                    done=False, source="offline", **fields)
    return Turn(reply="No problem, I'll call back to confirm. Thanks.",
                done=True, source="offline", **fields)


def _respond_live(msgs: list[Message], drug: str, dose: str, caller: str) -> Turn:
    """Claude-driven dialogue over the Anthropic Messages REST API via urllib (no SDK,
    so the web / CLI / MCP surfaces need only a key, not the [voice] or [agent] extra).
    Robust JSON extraction, null coercion, and emdash stripping on the spoken reply."""
    import json
    import re
    import urllib.request

    convo = "\n".join(f"{m.role}: {m.text}" for m in msgs) or "(call just connected)"
    prompt = (
        f"You are an AI assistant calling a pharmacy on behalf of {caller} to find out whether "
        f"{drug} {dose} is in stock for a patient. Talk like a real person on a quick phone "
        "call: warm, brief, usually one short sentence, no filler, no emdashes. Reply with "
        "STRICT JSON only, keys: reply (string), in_stock (yes|no|partial|unknown), confidence "
        "(0..1), done (bool), evidence_span (verbatim quote or empty). When the call just "
        f"connected (no prior turns), your reply MUST first say you are an AI assistant for "
        f"{caller} and that the call may be recorded, then ask if they have it (bot-disclosure "
        "and recording consent are required). If their answer is ambiguous or incomplete, ask "
        "ONE short clarifying question and set done=false. When confident, confirm in one "
        f"sentence and set done=true with in_stock.\n\nConversation so far:\n{convo}\n\n"
        "Your next turn as JSON:"
    )
    body = json.dumps(
        {
            "model": "claude-sonnet-4-6", "max_tokens": 400, "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    resp = json.load(urllib.request.urlopen(req, timeout=30))
    text = resp["content"][0]["text"]
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}

    in_stock = data.get("in_stock") or "unknown"
    if in_stock not in (s.value for s in StockState):
        in_stock = "unknown"
    reply = data.get("reply") or f"Sorry, do you have {drug} {dose} in stock?"
    reply = re.sub(r"\s*[—–]\s*", ", ", reply)  # strip em/en dashes
    return Turn(
        reply=reply, done=bool(data.get("done")), source="claude",
        in_stock=StockState(in_stock), confidence=round(float(data.get("confidence") or 0.0), 3),
        abstain=(in_stock == "unknown"), evidence_span=data.get("evidence_span") or "",
    )

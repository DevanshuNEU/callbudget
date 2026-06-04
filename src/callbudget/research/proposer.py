"""The LIVE proposer: an mcp-use agent that drives CallBudget's own MCP surface.

  ┌──────────────────────────────────────────────────────────────────────┐
  │ NOT EXERCISED IN CI. This needs the [agent] extra + an API key, which  │
  │ the build environment does not have, so it was written against the     │
  │ documented mcp-use / langchain-anthropic API but NOT run here. Verify  │
  │ locally against your installed versions; expect to adjust import paths │
  │ / class names if those API surfaces have moved. The tested mechanism   │
  │ (the search space, the journal, the EvalResult contract, in_space      │
  │ validation) lives in space.py and loop.py; this file is only the live  │
  │ flavor of the "propose the next config" step over it.                  │
  └──────────────────────────────────────────────────────────────────────┘

Run locally:
    uv sync --extra agent
    export ANTHROPIC_API_KEY=...
    uv run python -c "import asyncio, callbudget.research.proposer as p; \\
        print(asyncio.run(p.make_mcp_agent_proposer()([])))"

THEORY (the auto-research loop orchestrating an agent over MCP):
The auto-research loop is a search over ResearchConfig scored by the eval harness.
The PROPOSER is the step that picks the next point to try. An offline proposer is
just an Optuna `suggest` call. This live proposer instead hands the decision to an
LLM agent that can REASON about the journal so far and INSPECT the system through
exactly the MCP tools an external agent would use.

Why route the agent through CallBudget's OWN MCP server (predict_stock / plan_calls
/ run_eval) instead of giving it the numbers inline: it dogfoods the surface. The
agent that drives the research loop calls the same JSON-RPC tools that any external
agent or client would call to use CallBudget. So "the auto-research loop is an
agent orchestrated over MCP" is literally true, not a metaphor: the loop's proposer
IS an MCP client of this project's MCP server. If the surface is awkward for our own
optimizing agent, it is awkward for everyone, and we find that out here.

DEPENDENCY INJECTION + the proposer contract:
A proposer is any callable `(journal: list[dict]) -> tuple[ResearchConfig, str]`:
the next config to try plus a one-line rationale. `loop.py` depends only on that
shape, so the offline Optuna proposer and this live agent proposer are swappable.
The agent is asked to emit STRICT JSON; we parse it, snap each value to the nearest
legal SEARCH_SPACE point, and re-validate with `config.in_space()` so a model that
hallucinates an out-of-grid value can never poison the study.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from .space import SEARCH_SPACE, ResearchConfig

# A proposer returns the next config to evaluate and a one-line rationale for it.
# The live agent proposer is async (it awaits MCP tool calls), so its callable
# returns an Awaitable of that tuple.
Proposal = tuple[ResearchConfig, str]
AsyncProposer = Callable[[list[dict]], Awaitable[Proposal]]


def _snap(name: str, value: float) -> float:
    """Snap one proposed value to the nearest legal value on its SEARCH_SPACE axis.

    The agent reasons in continuous-ish terms ("try a deeper tree, a bit more
    regularization"); the grid is discrete. Snapping keeps an otherwise sensible
    proposal inside the space instead of rejecting it outright, while `in_space()`
    still guards against a value off any legal axis.
    """
    legal = SEARCH_SPACE[name]
    return min(legal, key=lambda legal_value: abs(legal_value - value))


def _snap_config(raw: dict) -> ResearchConfig:
    """Build a ResearchConfig from the agent's JSON, snapping every space axis.

    Keys outside SEARCH_SPACE (if any) fall through to ResearchConfig defaults via
    validation; keys on a space axis are snapped to the nearest legal value first.
    """
    snapped = dict(raw)
    for axis in SEARCH_SPACE:
        if axis in snapped and snapped[axis] is not None:
            snapped[axis] = _snap(axis, snapped[axis])
    config = ResearchConfig(**snapped)
    if not config.in_space():  # defense in depth: snapping should make this impossible
        raise ValueError(f"proposed config is outside the search space: {config.model_dump()}")
    return config


def _parse_agent_output(text: str) -> Proposal:
    """Parse the agent's reply into (ResearchConfig, rationale).

    The agent is told to return one strict-JSON object with a `config` object and a
    `rationale` string. We tolerate the model wrapping it in prose by extracting the
    outermost JSON object, then validate hard.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in agent output: {text!r}")
    payload = json.loads(text[start : end + 1])
    raw_config = payload.get("config", payload)  # allow a bare config object too
    rationale = str(payload.get("rationale", "")).strip() or "(no rationale given)"
    return _snap_config(raw_config), rationale


def _build_prompt(journal: list[dict]) -> str:
    """Compose the agent's task: the space, the journal so far, the output contract."""
    space_json = json.dumps(SEARCH_SPACE, indent=2)
    journal_json = json.dumps(journal, indent=2, default=str)
    return (
        "You are the proposer in an auto-research loop tuning CallBudget, a system "
        "that predicts pharmacy drug stock and plans the cheapest sequence of calls. "
        "Your job: pick the single next configuration to evaluate.\n\n"
        "You may call the CallBudget MCP tools (predict_stock, plan_calls, run_eval) "
        "to inspect the current system before deciding. run_eval reports ECF (lower "
        "is better, the cost metric) and false_positive_in_stock (lower is better, "
        "the hard safety metric).\n\n"
        f"The legal search space (you MUST choose values from these grids):\n{space_json}\n\n"
        f"The journal of configs already evaluated (config + measured EvalResult):\n"
        f"{journal_json}\n\n"
        "Reason about which untried region is most promising: drive ECF down without "
        "raising false_positive_in_stock. Do not repeat a config already in the "
        "journal.\n\n"
        "Return ONE strict-JSON object and nothing else, of the exact form:\n"
        '{"config": {"max_depth": int, "max_iter": int, "learning_rate": float, '
        '"n_samples": int, "abstain_threshold": float}, '
        '"rationale": "one short line on why this config next"}'
    )


def make_mcp_agent_proposer(
    model: str = "claude-sonnet-4-6",
    max_steps: int = 8,
) -> AsyncProposer:
    """Return a live proposer backed by an mcp-use agent over CallBudget's MCP server.

    Not exercised in CI; this is the live path (needs the [agent] extra and
    ANTHROPIC_API_KEY). The returned callable has the proposer contract
    `(journal) -> (ResearchConfig, rationale)`, awaitable.

    The agent connects to `callbudget.mcp.server.build_server()` (CallBudget's own
    MCP surface), is handed the journal-so-far plus the SEARCH_SPACE, reasons over
    the tools, and emits the next config as strict JSON. We snap each value to the
    nearest legal grid point and re-validate with `in_space()` before returning, so
    a hallucinated value can never enter the study.
    """

    async def _propose(journal: list[dict]) -> Proposal:
        # Imports are local so the module imports fine without the [agent] extra,
        # exactly like voice/live_call.py. The mcp-use / langchain API surfaces may
        # have moved; adjust these import paths against your installed versions.
        from langchain_anthropic import ChatAnthropic
        from mcp_use import MCPAgent, MCPClient

        from ..mcp.server import build_server

        # Run CallBudget's own FastMCP server in-process over stdio and point the
        # mcp-use client at it. FastMCP exposes an ASGI/stdio transport; the exact
        # spawn config below assumes mcp-use can launch the server module as a stdio
        # subprocess. If your mcp-use version wants an already-running server, point
        # the client at its URL instead. (build_server() is imported here so the
        # assumption is explicit and easy to swap.)
        _ = build_server  # documents the in-process server we are dogfooding
        client = MCPClient.from_dict(
            {
                "mcpServers": {
                    "callbudget": {
                        "command": "uv",
                        "args": ["run", "python", "-m", "callbudget.mcp.server"],
                    }
                }
            }
        )
        llm = ChatAnthropic(model=model)
        agent = MCPAgent(llm=llm, client=client, max_steps=max_steps)
        try:
            result = await agent.run(_build_prompt(journal))
        finally:
            # Close the spawned server session if the client exposes a closer.
            close = getattr(client, "close_all_sessions", None) or getattr(client, "close", None)
            if close is not None:
                maybe = close()
                if hasattr(maybe, "__await__"):
                    await maybe
        return _parse_agent_output(result if isinstance(result, str) else str(result))

    return _propose

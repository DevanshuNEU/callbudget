"""FastMCP adapter: exposes the engine's operations as MCP tools.

MCP (Model Context Protocol) is how agents call external tools over JSON-RPC, so
a model can call this engine as a first-class tool. All logic lives in engine.py;
this file is only the transport binding, so the two evolve independently.

Run as a server:  uv run python -m callbudget.mcp.server
"""

from __future__ import annotations

from .engine import CallBudgetEngine


def build_server():
    """Construct the FastMCP server with the three CallBudget tools registered."""
    from fastmcp import FastMCP

    mcp = FastMCP("callbudget")
    engine = CallBudgetEngine()

    @mcp.tool
    def predict_stock(
        drug: str = "Adderall XR", dose: str = "20 mg", location: str = "Boston"
    ) -> list[dict]:
        """Rank pharmacies by predicted probability the drug is in stock."""
        return engine.predict_stock()

    @mcp.tool
    def plan_calls(
        drug: str = "Adderall XR", dose: str = "20 mg", location: str = "Boston",
        budget: int | None = None,
    ) -> dict:
        """Return the call order that minimizes expected calls-to-find, plus that ECF."""
        return engine.plan_calls(budget)

    @mcp.tool
    def run_eval() -> dict:
        """ECF (naive vs learned) plus safety metrics (false-positive-in-stock, abstention)."""
        return engine.run_eval()

    @mcp.tool
    def converse(
        history: list[dict] | None = None,
        drug: str = "Adderall XR",
        dose: str = "20 mg",
        caller: str = "Devanshu",
    ) -> dict:
        """One turn of a pharmacy stock call. Given the conversation so far (a list of
        {role: agent|pharmacist, text}), return the agent's next line, the structured
        stock record, and whether the call is done. Same dialogue brain as the web and
        CLI; call repeatedly with growing history to hold a conversation."""
        from ..voice.dialogue import respond

        turn = respond(history or [], drug, dose, caller)
        return {
            "reply": turn.reply,
            "done": turn.done,
            "source": turn.source,
            "record": {"drug": drug, "dose": dose, **turn.record},
        }

    return mcp


if __name__ == "__main__":
    build_server().run()

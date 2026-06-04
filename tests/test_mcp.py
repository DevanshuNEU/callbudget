"""MCP layer tests: the engine logic (deterministic, no transport) plus a smoke
check that the FastMCP server constructs with its tools registered."""

from __future__ import annotations

from callbudget.mcp.engine import CallBudgetEngine
from callbudget.mcp.server import build_server


def test_predict_stock_ranks_descending():
    ranked = CallBudgetEngine(limit=20).predict_stock()
    assert len(ranked) > 0
    probs = [r["p_in_stock"] for r in ranked]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all("npi" in r and "name" in r for r in ranked)


def test_plan_calls_respects_budget():
    plan = CallBudgetEngine(limit=20).plan_calls(budget=3)
    assert len(plan["ordered_npis"]) == 3
    assert plan["expected_calls_to_find"] > 0


def test_run_eval_learned_not_worse_than_naive():
    ev = CallBudgetEngine(limit=20, history_per_pharmacy=15).run_eval()
    assert ev["ecf_learned"] <= ev["ecf_naive"]
    assert 0.0 <= ev["false_positive_in_stock"] <= 1.0


def test_server_builds():
    srv = build_server()  # constructs the engine + registers the 3 tools
    assert srv is not None

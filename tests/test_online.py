"""Online-learning + CLI tests: a confident call becomes a label and retrains;
a hedge does not; the CLI routes to the engine. Offline, deterministic."""

from __future__ import annotations

from callbudget.cli import main
from callbudget.mcp.engine import CallBudgetEngine


def test_confident_call_is_learned():
    engine = CallBudgetEngine(limit=20)
    top = engine.predict_stock()[0]["npi"]
    before = engine.n_observations
    result = engine.place_call(top, style="clear", seed=1)
    assert engine.observe(result) is True
    assert engine.n_observations == before + 1


def test_hedge_call_is_not_learned():
    engine = CallBudgetEngine(limit=20)
    npi = engine.predict_stock()[0]["npi"]
    result = engine.place_call(npi, style="hedge", seed=2)
    assert result.record.abstain is True
    before = engine.n_observations
    assert engine.observe(result) is False
    assert engine.n_observations == before


def test_cli_eval_prints_metrics(capsys):
    main(["eval"])
    out = capsys.readouterr().out
    assert "ecf_learned" in out and "false_positive_in_stock" in out

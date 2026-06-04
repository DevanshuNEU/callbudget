# CallBudget - Roadmap

> A sharp v1, with everything beyond it explicitly deferred so the scope stays honest.

## Shipped (v1)

- Foundation: typed domain model, content-addressed run config, test + lint setup.
- Prediction: stock-probability model + leakage-safe features over a real-shaped world.
- Policy: call ordering that minimizes expected calls-to-find (ECF).
- Real scaffolding: NPPES pharmacies, RxNorm drug identity, ARCOS-shaped throughput, FDA/ASHP shortage window.
- Voice extraction: structured stock from messy answers with calibrated abstention (offline path + a live Pipecat path).
- Reliability ablation: self-consistency drives false-positive-in-stock toward zero.
- Durable observation log: DuckDB + Parquet, windowed-SQL features (parity-tested against the Python reference).
- MCP server (`predict_stock` / `plan_calls` / `run_eval`), an end-to-end demo, a CLI, and the hero ECF chart.
- A web UI (`callbudget serve`: FastAPI + a static page) and MCP usage docs (`docs/mcp-usage.md`).
- Real shortage window fetched live from the openFDA drug-shortages API (cached; offline fallback).
- Auto-research loop (`callbudget research`): a seeded Optuna search over model + abstention knobs, scored by the closed-loop eval (ECF) under a hard false-positive-in-stock constraint, with a held-out validation guard, a JSONL journal, and a Rich leaderboard + report. On this synthetic problem the in-sample gain does not survive validation, so the loop recommends keeping the v1 defaults; the safety constraint and the validation guard are the deliverable.

## Next

- Regional volume context from real amphetamine dispensing (Part D by Geography & Drug, Medicaid SDUD, ARCOS Retail Summary). Note: these are state/ZIP-level, not per-pharmacy, so they enrich the regional prior; they do not replace the per-pharmacy throughput shape.
- A bandit (explore/exploit) caller as a policy variant; the research loop currently uses balanced round-robin coverage to keep the flywheel signal clean.
- Live telephony (real phone transport), so the offline voice loop becomes a real call path.

## Deferred (named, not gaps)

- Real telephony (Twilio/SIP), real phone numbers, real PHI.
- Full-duplex barge-in tuning, accent/emotion stress matrices, DTMF realism.
- Full NPPES national identity resolution (v1 = a curated subset).
- Formulary / prior-auth (CMS Part D), NADAC pricing as a model feature.
- Online/continual retraining (v1 = batch collect, retrain, replan).
- Optimal-stopping / bandit policy comparison (v1 = Bayesian-greedy).
- Multi-party rooms; concurrent multi-pharmacy calling; cost/observability dashboards.
- At scale: a Postgres + TimescaleDB system of record (see `ARCHITECTURE.md`).

# CallBudget - Build Log

A high-level record of how the system is built and why, organized by capability.
Read top to bottom to understand the project from first principles. Theory
deep-dives live in the learning notes; binding decisions live in `FOUNDATION.md`,
`SPEC.md`, and `ARCHITECTURE.md`; this file is the narrative.

---

## 1. What CallBudget is (plain English)

A patient needs a hard-to-find medication (say Adderall XR 20mg, which is in a
real national shortage). No website can tell you which nearby pharmacy actually
has it in stock right now, so today you call pharmacies one by one until one says
yes. CallBudget makes that search smart: it predicts which pharmacy is most likely
to have the drug and calls the most-likely ones first, so you find it in ~2 calls
instead of ~4, and it improves as it accumulates call history. A voice agent does
the calling and turns the messy spoken answer into a clean record, and it says
"I'm not sure, re-call" instead of guessing when the answer is ambiguous (because
a wrong "yes" sends a patient to an empty pharmacy).

## 2. The thesis

- The voice-agent calling is the **wedge**: a commodity (any vendor can dial a phone).
- The **moat** is the data exhaust: every call is a labeled, timestamped,
  store-level stock observation, and that dataset is scarce (no one else can query
  shelf inventory) and decays (a stock answer is good for hours), so a competitor
  cannot buy a dump and catch up. A real data network effect.
- CallBudget is the prediction layer that turns that exhaust into
  fewer-calls-per-search. The flywheel: more calls, better predictions, fewer
  calls. We demonstrate it as a chart.

## 3. The honesty boundary

The per-call in-stock outcomes can never be real (real-time shelf-stock data is
exactly what does not exist anywhere; that is the premise). So:

- **Real:** which pharmacies exist (NPPES), drug identity (RxNorm RxCUI 861226),
  throughput *shape* (ARCOS), the shortage window (FDA/ASHP).
- **Synthetic:** the in-stock outcomes, sampled from a hidden function.
- **Claim discipline:** every chart shows the mechanism on realistic inputs, never
  a claim about real Boston stock rates. ARCOS (opioids) is a same-DEA-schedule
  throughput proxy, never an Adderall-specific signal.

## 4. Architecture

Modules depend toward the stable core; swappable surfaces sit on top. See
`ARCHITECTURE.md` for the module and data diagrams (Mermaid). In short:
`schemas.py` is the shared vocabulary; `mcp/` splits into `engine.py` (logic) and
`server.py` (FastMCP transport adapter), a ports-and-adapters split.

## 5. How it was built (by capability)

**Foundation.** A typed domain model (`schemas.py`), a content-addressed run
config (`config.py`), and the test + lint setup (`pyproject.toml`, uv, Python
pinned to 3.11 because the newest runtime lacks binary wheels for native libs).
The ECF math (`policy/budget.py`) is pure and unit-tested against hand-computed
values.

**Prediction signal.** A synthetic world with a hidden per-pharmacy reliability
(`sim/callgen.py`), a leakage-safe feature matrix (`predict/features.py`), and the
stock-probability model (`predict/model.py`). We use sklearn HistGradientBoosting,
not xgboost, to avoid the macOS OpenMP system dependency (and it handles NaN
natively). The signal check (`scripts/signal_check.py`) confirmed the learned
ranker beats naive ordering (~4 to ~2.5 expected calls) and improves with history.

**Voice extraction.** Structured stock from a messy spoken answer with calibrated
abstention via self-consistency (`voice/extract.py`), dependency-injected so it
runs with no API key. The simulated pharmacist (`sim/pharmacist.py`) keeps the
hidden truth separate from how it talks. `voice/loop.py` has the offline path;
`voice/live_call.py` has the real Pipecat audio pipeline (verify locally).

**Real Boston data.** Loaders for NPPES pharmacies (cached), RxNorm RxCUI,
ARCOS-shaped throughput, and the shortage calendar (`data/loaders.py`,
`data/identity.py`). The hero chart (`scripts/build_chart.py` ->
`figures/ecf_flywheel.png`) shows ECF ~4.3 (naive) to ~2.3 (learned) on 19 real
pharmacies. The crossover (worse than random at cold-start,
decisively better with data) is the moat made visible.

**Reliability ablation.** Abstention is demonstrated with an ablation
(`scripts/reliability_ablation.py`): a 0%-false-positive rule extractor proves
nothing, so we stress it with a deliberately unreliable extractor (25% misread).
Self-consistency drives false-positive-in-stock from ~10% to ~0%, with a measured
insight that 5 samples beat 3 (same safety, less wasted abstention).

**MCP server + closing the loop.** `mcp/engine.py` (logic) + `mcp/server.py`
(FastMCP adapter) expose `predict_stock` / `plan_calls` / `run_eval`. The online
methods (`place_call`, `observe`) close the loop: a confident call becomes a
labeled observation and the model retrains. `demo.py` runs the whole narrative;
`cli.py` is the `callbudget` command. fastmcp is a core dependency (the MCP surface
is a deliverable, not voice-only).

**Durable observation store.** The moat is persisted to a DuckDB + Parquet log
(`data/store.py`) with windowed-SQL features, instead of living in RAM. A parity
test (`tests/test_store.py`) proves the SQL features equal the Python reference
exactly, so the engine migrated onto the store with zero behavior drift.
`ARCHITECTURE.md` records the OLTP/OLAP scaling plan (Postgres + TimescaleDB at
scale) and the data-source upgrade (real amphetamine priors).

**Web UI and usage docs.** A FastAPI app (`web/app.py`) serves a single static
page over the engine (`callbudget serve`), another adapter alongside the MCP
server: the ranked pharmacies, the call plan, a call button, the live metrics,
and the chart. `docs/mcp-usage.md` documents the MCP tools, use cases, and
example prompts.

**Auto-research loop.** The same rigor turned on the system's own tuning
(`research/loop.py`, `research/evaluate.py`). An agent searches the model
hyperparameters (`max_depth` / `max_iter` / `learning_rate`) and the voice
self-consistency knobs (`n_samples` / `abstain_threshold`); the fitness function
is the existing closed-loop eval (call -> extract -> abstain -> label -> retrain
under the 25%-misread extractor), so extraction quality flows into ECF. The
false-positive-in-stock safety metric is a hard constraint, not a co-objective:
the lowest-ECF config tends to be the least-abstaining and so the least safe, and
the constraint is what stops the search from choosing it. Optuna (seeded TPE plus
a constraints function) runs the search instead of a hand-rolled loop; every
trial is journaled to `runs/research_journal.jsonl`, with a Rich leaderboard and
`runs/research_report.md`. We chose a JSONL journal plus Optuna's study over
Weights & Biases / MLflow (heavyweight for this scope). The honest outcome is the
point: the in-sample ECF gain (~10-15%) does not survive a disjoint held-out
validation set (it sits within world-to-world noise), so the loop recommends
keeping the v1 defaults. The deliverable is therefore the safety constraint
(which rejects unsafe configs, including the lowest-ECF one) and the validation
guard (which refuses to ship an overfit gain), not a headline ECF number. The
integrity boundary is unchanged: outcomes are synthetic, so the loop proves the
search mechanism on realistic inputs, never a real stock-rate gain. Run it with
`callbudget research`.

## 6. How to run it

```bash
uv sync                                          # core + dev (Python 3.11, fetched by uv)
uv run pytest -q                                 # tests
uv run ruff check .                              # lint
uv run callbudget demo                           # end-to-end walkthrough (the narrative)
uv run callbudget chart                          # -> figures/ecf_flywheel.png (real Boston)
uv run callbudget eval                           # ECF + safety metrics as JSON
uv run python scripts/signal_check.py            # the prediction signal check (synthetic)
uv run python scripts/reliability_ablation.py    # the abstention ablation
uv run callbudget research                       # the auto-research loop (search + guards)
uv run python -m callbudget.mcp.server           # run the MCP server
uv sync --extra voice                            # add live Pipecat voice deps (local audio)
uv run python -m callbudget.voice.live_call      # the live voice call (local, keys + mic)
```

## 7. Status

v1 complete, all green (ruff + 39 tests). The repo is a runnable, tested,
documented artifact. Next: the real amphetamine-priors data upgrade (see
`ROADMAP.md` and `ARCHITECTURE.md`).

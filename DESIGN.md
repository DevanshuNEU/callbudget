# CallBudget - Design & Module Layout

> Read `FOUNDATION.md` and `SPEC.md` first. This is the module map + the design decisions that don't belong in the spec contract.

## Repo layout (intended)

```
callbudget/
  README.md
  FOUNDATION.md          binding thesis + gates
  SPEC.md                build contract
  DESIGN.md              this file
  ROADMAP.md             v1 curve
  BUILDLOG.md            high-level narrative
  pyproject.toml         deps, Python pin, ruff/pyright/pytest, CLI entry point
  LICENSE                Apache-2.0
  src/callbudget/
    schemas.py           shared domain model
    config.py            settings + content-addressed run_id
    cli.py               the `callbudget` CLI
    demo.py              end-to-end walkthrough + chart helper
    data/                loaders + identity resolution
    sim/                 simulated pharmacist + call-log generator
    predict/             stock-probability model + features
    policy/              budgeted call-ordering + ECF
    voice/               extraction/abstention + offline + live (Pipecat) call
    evals/               grading harness + the chart + reliability ablation
    mcp/                 engine (logic) + server (FastMCP adapter)
  data/                  pharmacy cache (committed) + bulk sources (gitignored)
  figures/               the ECF flywheel chart
  tests/
```

## Design decisions

1. **Synthetic-first, real-data-shaped.** The simulator is seeded by real public distributions (ARCOS volume, FDA/ASHP shortage windows) so the synthetic world is believable, but no real PHI or live inventory is touched. This is the only honest way to build the moat-as-math demo solo.

2. **Two decoupled clocks.** The voice simulator is decoupled from wall-clock time (the tau-Voice trick) so the sim pharmacist can use a strong LLM without real-time latency pressure, and eval runs fast.

3. **Confidence gate sits between voice and prediction.** Extraction confidence is not cosmetic - it is the valve that protects the model from label noise. This is the architectural detail that ties reliability work to model quality.

4. **Interpretable model on purpose.** v1 uses trees, not a black box, so an operator can explain *why* a pharmacy was ranked high (high throughput + recent in-stock history + not in shortage). Explainability matters.

5. **MCP as the surface, not an afterthought.** The same MCP server that serves `predict_stock`/`plan_calls`/`run_eval` is the thing an agent calls. The transport is swappable; the logic is not.

6. **Content-addressed runs.** Hash the resolved config + data file digests into a run id so the ECF chart is reproducible and a re-runner cannot silently diverge.

## The hero artifact

A single chart: **expected calls-to-find vs accumulated call history**, naive prior flat near ~4, learned model curving down toward ~2. Plus a reliability panel: extraction accuracy and the **false-positive-in-stock rate**, with one visible abstain example. That chart + panel is the centerpiece of the writeup.

## Related
- `ARCHITECTURE.md` - module + data architecture and the scaling plan.

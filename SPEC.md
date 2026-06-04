# CallBudget - Technical Spec

> Read `FOUNDATION.md` first. This spec is the build contract for v1. **Status: LOCKED 2026-06-01** (open questions resolved in section 10). Canonical scenario: Adderall XR 20mg, Boston.

---

## 1. System overview

Three coupled subsystems, one closed loop:

```
                    free priors (NPPES x RxNorm x ARCOS x FDA/ASHP)
                                    |
                                    v
   request ---> [PREDICTION] P(in_stock|...) ---> [POLICY] call order
                     ^                                   |
                     |                                   v
              labeled obs (clean)              [VOICE FRONT-END] call sim pharmacist
                     |                                   |
                     +------ [EXTRACTION+ABSTENTION] <----+
                              {drug,dose,in_stock,conf,abstain}
```

Every call is an action (find the drug) and a measurement (a labeled observation). The extraction confidence gate decides which measurements train the model.

## 2. Data pipeline (free, no PHI)

| Source | Role | Access | Caveat |
|---|---|---|---|
| **NPPES** | pharmacy nodes (NPI, taxonomy, location) | free API + monthly bulk | dedup NPI<->chain<->phone (identity resolution) |
| **RxNorm / RxNav** | drug identity, RxCUI, NDC crosswalk | free, no key, ~20 req/s | normalization backbone |
| **DEA ARCOS** | per-pharmacy controlled-substance THROUGHPUT prior (not drug-specific) | WaPo release + Notre Dame portal, R/Python pkgs | **Public ARCOS = OPIOIDS only, 2006-2014.** NOT amphetamines. Use as a same-schedule throughput proxy ONLY; caveat stated in memo + FOUNDATION principle 6 |
| **FDA + ASHP shortages** | Adderall shortage-state feature + ground-truth labels | free | dated onset/resolution timelines; carries the drug-specific signal |
| **NADAC** | cash price feature | free, weekly | **DEFERRED to post-v1** |

Scope (locked): **curated Boston-metro NPPES subset**, not national. Canonical drug: **Adderall XR 20mg** (Schedule II, real ASHP shortage, a sympathetic access problem).

Pipeline: build Boston pharmacy nodes (NPPES) + drug node (RxNorm) -> attach ARCOS opioid volume as a per-pharmacy throughput prior + Adderall shortage-state from FDA/ASHP -> a **synthetic call-log generator** samples ground-truth stock (from a HIDDEN true function) and a messy verbal answer per (pharmacy, date).

**Fallback (F3 gate):** if pulling the Boston ARCOS subset stalls, fall back to hand-tuned priors calibrated to the ARCOS distribution shape and say so.

## 3. Prediction layer

- Model: **gradient-boosted trees (xgboost/sklearn), interpretable. Locked.** `P(in_stock | pharmacy, day_of_week, region, recent_call_history, shortage_state, throughput_prior)`. Document the feature set; interpretability feeds the memo's "why this pharmacy ranked high."
- **Non-circularity (binding, = F1 gate):** the generator's true stock function is HIDDEN from the model. The model learns only from observed `(features, in_stock)` call pairs. Evaluation is on **held-out requests** (cross-validated).
- **Baseline = proximity/random call order** (NOT the generator's prior). The ECF improvement is legitimate only if it comes from the model learning structure from call exhaust.

## 4. Policy layer

- v1: **Bayesian-greedy** - call the highest `P(in_stock)` pharmacy first, update, stop on first confirmed in-stock. Report ECF.
- Optional v1.1: a budgeted bandit / optimal-stopping baseline for comparison.
- Metric: **expected calls-to-find (ECF)** over a held-out request set; the hero chart is ECF(naive) vs ECF(learned) as labeled data grows.

## 5. Voice front-end

- **Stack (locked, single): Pipecat** = Deepgram STT -> Claude LLM -> Cartesia TTS, swappable vendors. **No OpenAI Realtime spike** (dropped: second paradigm, throwaway, extra dependency). One stack, no throwaway, and "I assembled a swappable STT->LLM->TTS pipeline" is the credibility line for a technical reviewer. Telephony-free, browser/local audio bus, near-$0 (Deepgram $200 credit + Cartesia free tier).
- **Simulated pharmacist** (callee): a second agent reading from a HIDDEN stock table (free ground truth), PHI-free, time-decoupled from wall-clock (the tau-Voice trick) so it can use a strong LLM without latency pressure.
- **Tiered scope (de-risk):**
  - **v1-min (must ship):** the agent asks the stock question; the sim pharmacist answers MESSILY in audio ("we got the generic 10s, the 20 XR's on backorder"); the agent extracts structure with abstention. This is the STT->LLM->TTS + extraction core.
  - **v1-stretch (additive, only if the core lands comfortably):** IVR-tree navigation ("press 2 for pharmacist") + a hold-and-resume delay before the human picks up. This is the realistic wow, and also the rabbit hole; it is additive, not load-bearing.

## 6. Extraction + abstention

Typed record via tool/JSON schema:
```json
{ "drug": "", "dose": "", "form": "tablet|capsule|XR|liquid|null",
  "brand_vs_generic": "brand|generic|either|unknown",
  "in_stock": "yes|no|partial|unknown", "backorder_date": "ISO8601|null",
  "confidence": 0.0, "abstain": false, "evidence_span": "verbatim quote" }
```
- Require `evidence_span`. Set `abstain: true` on ambiguity.
- **Confidence = self-consistency:** run extraction 3x low-temp; agreement = high confidence, disagreement -> abstain.
- Grading vs the sim's known stock table: extraction accuracy, **false-positive-in-stock rate** (headline), abstention quality curve.

## 7. MCP interface

Expose CallBudget as an MCP server (FastMCP, dual stdio + Streamable HTTP - matches OCI patterns). v1 tools:
- `predict_stock(drug, dose, location)` -> ranked pharmacies with `P(in_stock)` + timestamps.
- `plan_calls(drug, dose, location, budget)` -> ordered call plan + expected ECF.
- `run_eval(scenario_set)` -> extraction accuracy + false-positive-in-stock + abstention curve.

The MCP surface lets an agent drive the engine as a first-class tool.

## 8. Stack

| Layer | Pick | Reason |
|---|---|---|
| Language | Python 3.11 | mature wheels; the ML + MCP ecosystem |
| Pkg mgr | uv | 2026 default |
| Config | pydantic v2 | matches house style; content-addressed run identity |
| ML | scikit-learn / xgboost | interpretable v1 baseline |
| Voice | Pipecat (single stack, no spike detour) | "I assembled a swappable pipeline" credibility |
| STT/LLM/TTS | Deepgram / Claude / Cartesia | swappable; free-tier friendly; needs Claude key + Deepgram + Cartesia free signups |
| MCP | FastMCP | dual transport; matches OCI |
| Tests | pytest | standard |
| Lint/type | ruff + pyright | 2026 defaults |
| Logs | JSONL | matches evals ecosystem |
| License | Apache 2.0 | matches author's public work |

## 9. LOC budget (v1, rough)

| Module | Budget | Status |
|---|---|---|
| `data/loaders.py` (NPPES/RxNorm/ARCOS/FDA-ASHP) | ~250 | to build |
| `sim/pharmacist.py` (IVR + hold + messy readout) | ~200 | to build |
| `sim/callgen.py` (synthetic call-log generator) | ~150 | to build |
| `predict/model.py` (features + stock-prob model) | ~200 | to build |
| `policy/budget.py` (Bayesian-greedy + ECF) | ~150 | to build |
| `voice/loop.py` (Pipecat pipeline) | ~250 | to build |
| `voice/extract.py` (schema + 3x self-consistency + abstain) | ~180 | to build |
| `eval/grade.py` (accuracy + FP-in-stock + abstention curve) | ~150 | to build |
| `mcp/server.py` (FastMCP tools) | ~150 | to build |
| tests + fixtures | ~300 | to build |
| **Total** | **~2,200** | |

## 10. Locked decisions (2026-06-01)

| # | Decision | Locked value |
|---|---|---|
| Name | Working project name | CallBudget |
| Scenario | Canonical demo | Adderall XR 20mg, Boston |
| Data scope | Pharmacy set | Curated Boston-metro NPPES subset (not national) |
| ARCOS | Role + honesty | Opioid-throughput proxy only (NOT Adderall-specific); caveat stated |
| ML | Model | Gradient-boosted trees (xgboost/sklearn), interpretable |
| Non-circularity | Rigor | Hidden true function; held-out eval; baseline = proximity/random |
| Voice | Stack | Pipecat (Deepgram + Claude + Cartesia); no OpenAI Realtime spike |
| Voice scope | Tiering | v1-min = audio extraction + abstention; v1-stretch = IVR + hold |
| NADAC | Pricing | Deferred to post-v1 |
| Build | Validation | staged, with a go/no-go gate after the prediction signal check |

## Related
- `FOUNDATION.md` - binding thesis + gates.
- `ARCHITECTURE.md` - data architecture + scaling plan.

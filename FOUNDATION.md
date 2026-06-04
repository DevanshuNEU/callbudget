# CallBudget - FOUNDATION (binding)

> This document is the binding foundation. It supersedes scattered notes for the purpose of design review. Companion docs: SPEC.md, ARCHITECTURE.md.

> **Locked 2026-06-01.** Working name CallBudget. Canonical scenario: find Adderall XR 20mg in Boston (see section 7). Stack + scope locked in SPEC.md section 10.

---

## 1. The thesis (one paragraph)

Real-time pharmacy-level inventory is not behind a paywall; it was never made into a queryable signal (e-prescribing rails carry prescriptions, not shelf state; inventory lives in siloed on-prem pharmacy systems; chains gate what little they expose). Calling pharmacies manufactures that signal. The defensible asset is the **proprietary, decaying, store-level availability time-series** and the **prediction layer** that turns each call into fewer-calls-per-search. CallBudget builds that layer, on free + synthetic data, fronted by a voice extraction agent whose confidence gates label quality.

## 2. The construct (what we measure)

**CallBudget is operationalized as active sensing under a calling budget:** given a (drug, dose, location) request, choose an order in which to call pharmacies that minimizes the **expected number of calls to first confirmed in-stock**, where each call yields a (possibly noisy, possibly abstained) structured stock observation that updates a stock-probability model.

Two coupled layers:
- **Prediction:** `P(in_stock | pharmacy, drug, dose, day_of_week, region, recent_history, shortage_state)`.
- **Policy:** a sequential decision rule (Bayesian-greedy / bandit / optimal-stopping) that consumes the prediction to order calls.
- **Front-end (voice):** turns a spoken pharmacist answer into a typed record with calibrated confidence + abstention. Only clean observations train the model; abstained ones become "re-call," not poisoned labels.
- **Self-improvement (research loop):** a seeded search over the model + abstention knobs, scored by this same closed loop (ECF) **subject to** a hard false-positive-in-stock constraint, with a disjoint held-out validation guard that refuses to ship an in-sample gain that does not generalise. The integrity boundary is unchanged: outcomes are synthetic, so the loop proves the search mechanism on realistic inputs, not a real stock-rate gain.

## 3. Principles (binding)

1. **Zero PHI.** Synthetic pharmacist + free public data only. No real patient data, ever.
2. **The headline metric is a cost, not an accuracy.** Lead with **expected calls-to-find**, then the **false-positive-in-stock rate** (the safety metric). Single-run accuracy is secondary.
3. **Abstention is a feature, not a failure.** An agent that says "I'm not sure, re-call" beats one that confidently mislabels. Label noise is the enemy of the flywheel.
4. **Every fact is timestamped.** Stock answers decay. Freshness is first-class.
5. **The voice slice is a bridge, not a claim of voice expertise.** It orchestrates a pipeline; it does not reinvent ASR.
6. **Honest provenance.** v1 uses an ARCOS-*shaped* throughput prior calibrated to the Washington Post opioid release (which is opioids-only), so the current per-pharmacy throughput number is a same-DEA-schedule *shape*, not Adderall-specific. The shortage window, by contrast, is now real: fetched live from the openFDA drug-shortages API (onset 2022-10-12, status Current), cached for offline/hermetic use with a documented fallback. **Upgrade path (identified 2026-06-02):** real amphetamine-specific volume IS publicly available and should replace the proxy: CMS Medicare Part D "by Geography and Drug" (amphetamine-dextroamphetamine, state-level), Medicaid SDUD (amphetamine units, state x quarter, NDC-level), and DEA ARCOS Retail Drug Summary Reports (amphetamine, ZIP-3). All are real and drug-correct but top out at state/ZIP, never per-store. **The boundary is unchanged:** real priors improve the inputs; the per-call in-stock OUTCOME stays synthetic because no public source has per-store real-time stock. Every synthetic assumption is documented.
7. **Reproducible.** Resolved config + data hashes pin a run identity (content-addressing, so a run always traces to what produced it).

## 4. Falsification / kill gates (decide before building far)

| # | Gate | If it fires |
|---|---|---|
| F1 | Learned model saves < 1 expected call vs a **proximity/random** baseline, on **held-out** requests (cross-validated; the generator's true stock function is hidden from the model, which learns only from observed call pairs) | The flywheel claim is too weak; reframe as "the eval harness + abstention layer" (still valuable) |
| F2 | Voice extraction false-positive-in-stock rate cannot be driven below a naive baseline by abstention | Abstention adds no safety; ship the prediction layer alone + document the negative result |
| F3 | ARCOS priors do not produce a believable stock distribution (sanity-checked vs FDA/ASHP shortage timelines) | Drop ARCOS; use hand-tuned priors + say so |
| F4 | The "uncovered business-IVR-and-hold" white space turns out to be covered by prior art | Drop any "first" framing; position as a clean re-implementation + the prediction tie-in |

None are firing yet; all get tested in v1.

## 5. The metrics (defined precisely)

- **Expected calls-to-find (ECF):** mean number of calls until first confirmed in-stock, over a held-out request set, under a given policy. Report ECF(naive prior) vs ECF(learned model). The hero chart is ECF dropping as labeled data accumulates.
- **False-positive-in-stock rate:** `P(extracted in_stock = yes | ground-truth = no)`, on non-abstained calls. Safety-critical.
- **Extraction accuracy:** field-weighted exact-match on the structured record, non-abstained turns.
- **Abstention quality:** abstain-when-wrong (good) vs abstain-when-clear (over-cautious); report the coverage/accuracy tradeoff curve.

## 6. What CallBudget is NOT

- NOT another medication-finding chatbot (that is the commodity wedge).
- NOT a claim to have built voice infrastructure from scratch.
- NOT a healthcare-domain-expertise artifact; it is a rigor + systems + moat artifact.
- NOT using real telephony, real phone numbers, or real PHI.
- NOT a production pricing/formulary tool; pricing (NADAC) is an optional feature, not the core.
- NOT shipped outward until the owner approves the spec, the build, and every outbound artifact.

## 7. Canonical demo scenario (locked)

**"Find Adderall XR 20mg for a patient in Boston."** Chosen deliberately:
- A real, sympathetic access problem: ADHD patients searching for a stimulant during a national shortage.
- Boston gives a concrete, bounded NPPES subset to work with.
- Adderall is a real, current ASHP-listed shortage drug, so the shortage-state feature is non-trivial.
- It is Schedule II, so the ARCOS opioid-throughput prior is a same-DEA-schedule (defensible) proxy, with the caveat in principle 6.

Everything in v1 is demonstrated on this one scenario end to end: predict -> plan calls -> voice call the sim pharmacist -> extract with abstention -> label -> reorder -> ECF chart.

## 8. Real vs synthetic (the integrity boundary, locked)

The single most important honesty line in the project:

- **REAL:** which pharmacies exist (NPPES, 19 deduplicated Boston pharmacies in the demo subset), the drug identity (RxNorm RxCUI **861226** = Adderall XR 20mg, i.e. 4 amphetamine salts x 5mg), the **shape** of per-pharmacy throughput (ARCOS opioid-volume distribution, heavy-tailed), and the Adderall shortage window (FDA declared 2022-10-12, ASHP active through 2026).
- **SYNTHETIC:** the per-call **in-stock outcomes**. These are sampled from a hidden function, NOT fetched, because real-time shelf-stock data is precisely what does not exist anywhere. Manufacturing that signal is the entire premise of a medication-finding service.
- **Claim discipline:** the ECF chart demonstrates the prediction *mechanism on realistic inputs*. It is NOT a claim about real Boston stock rates. ARCOS (opioid) is a same-DEA-schedule *throughput* proxy, never an Adderall-specific signal. Every chart carries this caveat.

Result (synthetic outcomes, real scaffolding, averaged over 5 runs): ECF naive ~4.3; learned ~2.3 with enough call history (about 47% fewer calls). At cold-start (1 call) the learned ranker is slightly WORSE than random, which is the point: **no data, no edge; the advantage is unlocked by accumulated call data. The data is the moat.**

## Related
- `SPEC.md` - the build contract.
- `ARCHITECTURE.md` - module + data architecture and the scaling plan.

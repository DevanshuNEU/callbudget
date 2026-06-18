# Using the CallBudget MCP server

CallBudget exposes its engine over the Model Context Protocol (MCP) so an AI agent
(Claude, Cursor, or any MCP client) can call it as a tool. This page lists the
tools, when to use each, and example prompts.

> Integrity note: the per-call in-stock outcomes are simulated (real-time
> shelf-stock data does not exist publicly). The pharmacies, drug identity, and
> shortage window are real; the outcomes are not. See `FOUNDATION.md`.

## Run the server

```bash
uv sync                              # installs the package + deps
uv run python -m callbudget.mcp.server
```

## Connect it to an MCP client

Claude Code:

```bash
claude mcp add callbudget -- uv run python -m callbudget.mcp.server
```

Or add it to an `.mcp.json` (project scope):

```json
{
  "mcpServers": {
    "callbudget": {
      "command": "uv",
      "args": ["run", "python", "-m", "callbudget.mcp.server"]
    }
  }
}
```

## Tools

### `predict_stock(drug, dose, location)`
Rank pharmacies by the predicted probability they have the drug in stock.
Returns a list of `{ npi, name, zip, chain, p_in_stock }`, highest first.
Use it to answer "where should I look first?"

### `plan_calls(drug, dose, location, budget=None)`
Return the call order that minimizes the expected number of calls to find the
drug, plus that expected count. Returns
`{ drug, dose, ordered_npis, expected_calls_to_find }`. `budget` caps the list.
Use it to answer "what is the most efficient sequence of calls, and how many
should I expect to make?"

### `run_eval()`
Report how the learned ranker compares to naive ordering and how safe the
extraction is. Returns `{ n_pharmacies, ecf_naive, ecf_learned,
false_positive_in_stock, abstention_rate }`.
Use it to answer "is this actually better than calling at random, and is it safe?"

### `converse(history, drug, dose, caller)`
One turn of a two-way pharmacy stock call, driven by the one dialogue brain
(`voice/dialogue.py`). Pass the conversation so far as a list of
`{ role: "agent" | "pharmacist", text }`; get back the agent's next line, the
structured stock `record`, and `done`. The agent opens with the AI + recording
disclosure, asks a clarifying question when unsure instead of guessing, and
confirms when sure. Call it repeatedly with growing `history` to hold a call.
Live Claude when `ANTHROPIC_API_KEY` is set, else a deterministic offline brain
(the `source` field says which, honestly).
Use it to answer "actually call the pharmacy and extract a clean stock record."

## What the agent sees (a real, reproducible transcript)

An agent wired to the four tools, asked "find Adderall XR 20mg in Boston, call the
best one, and tell me if it is worth it." The values below are one real run of the
offline engine (synthetic outcomes on real scaffolding; your run reproduces them
with seed 7).

```
> predict_stock(drug="Adderall XR", dose="20 mg", location="Boston")
[ { name: "Boston Children'S Healthsolutions Rx", zip: "02115", p_in_stock: 0.606 },
  { name: "Atrius Health, Inc.",                  zip: "02116", p_in_stock: 0.392 },
  { name: "Cvs Pharmacy Inc",                     zip: "02111", p_in_stock: 0.305 }, ... ]

> plan_calls(drug="Adderall XR", dose="20 mg", budget=5)
{ ordered_npis: [ ...5 pharmacies, most-likely first... ],
  expected_calls_to_find: 2.15 }

> run_eval()
{ n_pharmacies: 19, ecf_naive: 5.28, ecf_learned: 2.78,
  false_positive_in_stock: 0.0, abstention_rate: 0.647 }

> converse(history=[ {agent, "...in stock right now?"},
                     {pharmacist, "Yes, we have the brand 20 milligram in stock."} ])
{ reply: "Perfect, thank you ... brand Adderall XR 20 mg is in stock right now, got it!",
  done: true, source: "offline",
  record: { in_stock: "yes", confidence: 0.95, abstain: false,
            evidence_span: "Yes, we have the brand 20 milligram in stock right now." } }
```

The agent's takeaway: call the most-likely pharmacy first, expect ~2 calls instead of
~5, and when the pharmacist hedges the tool **abstains and re-calls** rather than
storing a wrong "in stock" (that is why `false_positive_in_stock` is 0.0). The numbers
are a synthetic-outcome run, not a claim about real Boston stock; see `FOUNDATION.md`.

## Example prompts

- "Using callbudget, which Boston pharmacies are most likely to have Adderall XR
  20mg in stock right now? List the top five with probabilities."
- "Give me an efficient call plan for Adderall XR 20mg with a budget of 5
  pharmacies, and tell me how many calls to expect."
- "Run the CallBudget evaluation: how does the learned ranking compare to calling
  at random, and what is the false-positive-in-stock rate?"
- "Plan the calls, then explain why the top pharmacy is ranked first."

## The metrics, briefly

- **expected calls-to-find (ECF):** the average number of calls to the first
  in-stock pharmacy. Lower is better; the learned ranker beats naive ordering and
  improves as call history accumulates.
- **false-positive-in-stock:** the chance the agent says "in stock" when the truth
  is "no" (the unsafe error). The extractor abstains rather than guess, keeping
  this near zero.

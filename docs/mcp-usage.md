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

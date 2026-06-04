"""The auto-research loop: an agent that improves CallBudget against its own metrics.

This package is the JD's headline pattern made concrete: a search agent proposes
configurations, the existing eval harness is its fitness function, the safety
metric is a hard constraint, and every experiment is journaled and validated on
held-out seeds. It does not invent ML; it tunes the system honestly and records
the trail. The integrity boundary is unchanged: outcomes are synthetic, so the
loop proves the SEARCH mechanism on realistic inputs, not real-world stock gains.

Layers (mirrors the rest of the codebase: stable core, swappable surface):
- `space.py`     the contract: ResearchConfig, EvalResult, the search space.
- `evaluate.py`  the fitness function: a closed-loop run scored by ECF + safety.
- `loop.py`      the optimizer: an Optuna study + the journal + the report.
- `proposer.py`  the live flavor: an mcp-use agent that drives the MCP surface.
"""

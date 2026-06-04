"""End-to-end demo: the whole CallBudget loop in one runnable narrative.

predict -> plan -> place a call -> extract (or abstain) -> learn from a confident
answer -> note that the aggregate effect is the ECF flywheel chart. This is the
artifact to walk through when explaining the project end to end.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .data.identity import resolve
from .data.loaders import load_arcos_throughput, load_boston_pharmacies, load_shortage_calendar
from .evals.grade import ecf_curve, render_ecf_chart
from .mcp.engine import CallBudgetEngine


def _step(console: Console, n: int, title: str) -> None:
    """Render a numbered step header so each stage reads as its own beat on screen."""
    console.print(f"\n[bold cyan]{n}[/bold cyan]  [bold]{title}[/bold]")


def run_demo(limit: int = 28, seed: int = 7) -> None:
    """Walk the whole CallBudget loop on real Boston pharmacies, rendered with Rich."""
    console = Console()
    engine = CallBudgetEngine(limit=limit, seed=seed)
    drug = engine.drug

    console.print(
        Panel(
            f"[bold]{drug.drug_name} {drug.dose}[/bold], Boston\n"
            "[dim]predict -> plan -> call -> extract/abstain -> learn -> eval[/dim]",
            title="CallBudget demo",
            border_style="cyan",
            expand=False,
        )
    )

    _step(console, 1, "Predict which pharmacies likely have it (top 5)")
    pred = Table(box=None, pad_edge=False)
    pred.add_column("P(in_stock)", justify="right", style="green")
    pred.add_column("pharmacy")
    pred.add_column("zip", style="dim")
    for r in engine.predict_stock()[:5]:
        pred.add_row(f"{r['p_in_stock']:.2f}", r["name"][:34], r["zip"])
    console.print(pred)

    plan = engine.plan_calls(budget=5)
    _step(console, 2, "Call plan (budget 5)")
    console.print(
        f"   Expected calls-to-find: [bold cyan]{plan['expected_calls_to_find']}[/bold cyan]"
    )

    top = plan["ordered_npis"][0]
    _step(console, 3, f"Place a call to the top-ranked pharmacy ({engine.name_of(top)[:30]})")
    result = engine.place_call(top, style="clear", seed=1)
    rec = result.record
    console.print(f'   pharmacist: [italic]"{result.transcript}"[/italic]')
    console.print(
        f"   extracted: in_stock=[bold]{rec.in_stock}[/bold] conf={rec.confidence} "
        f"abstain={rec.abstain}"
    )
    before = engine.n_observations
    learned = engine.observe(result)
    verdict = "learned (label stored, model refit)" if learned else "abstained (re-call, no label)"
    colour = "green" if learned else "yellow"
    console.print(
        f"   [bold {colour}]-> {verdict}[/bold {colour}]; observation log "
        f"{before} -> {engine.n_observations} [dim](DuckDB/Parquet)[/dim]"
    )

    second = plan["ordered_npis"][1]
    _step(console, 4, f"A different pharmacy hedges ({engine.name_of(second)[:30]})")
    result2 = engine.place_call(second, style="hedge", seed=2)
    console.print(f'   pharmacist: [italic]"{result2.transcript}"[/italic]')
    learned2 = engine.observe(result2)
    outcome2 = "learned" if learned2 else "re-call (no label)"
    colour2 = "green" if learned2 else "yellow"
    console.print(
        f"   extracted: abstain=[bold]{result2.record.abstain}[/bold] -> "
        f"[bold {colour2}]{outcome2}[/bold {colour2}]"
    )

    _step(console, 5, "One call is one turn of the flywheel")
    ev = engine.run_eval()
    eval_table = Table(box=None, pad_edge=False)
    eval_table.add_column("metric", style="dim")
    eval_table.add_column("value", justify="right")
    eval_table.add_row("ECF naive -> learned", f"{ev['ecf_naive']} -> {ev['ecf_learned']}")
    eval_table.add_row("false-positive-in-stock", f"{ev['false_positive_in_stock']}")
    eval_table.add_row("abstention rate", f"{ev['abstention_rate']}")
    console.print(eval_table)
    console.print(
        Panel(
            "This step is the [bold]single-world engine eval[/bold] (one seed). It is NOT the "
            "canonical hero number.\n"
            "The headline ECF (~4.3 -> ~2.3) is averaged over 5 seeds in the flywheel chart. "
            "Run [bold]callbudget chart[/bold] or see the README for that figure.",
            title="read this number honestly",
            border_style="yellow",
            expand=False,
        )
    )


def flywheel_chart(out_path: str = "figures/ecf_flywheel.png", limit: int = 28) -> str:
    """Render the ECF flywheel chart on real Boston pharmacies. Returns the path."""
    pharmacies = resolve(load_boston_pharmacies(limit=limit))
    load_arcos_throughput(pharmacies)
    shortage = load_shortage_calendar()
    levels, baseline, learned = ecf_curve([1, 3, 7, 15, 30], pharmacies, shortage)
    caption = (
        "Real Boston pharmacies (NPPES) + RxNorm + ARCOS-shaped throughput + real Adderall "
        "shortage window. In-stock outcomes are SIMULATED (that data does not exist). "
        "ECF averaged over 5 runs."
    )
    return render_ecf_chart(out_path, levels, baseline, learned, len(pharmacies), caption=caption)

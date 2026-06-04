"""Command-line interface: the human front door to CallBudget.

    callbudget predict    rank pharmacies by P(in_stock)
    callbudget plan       call order + expected calls-to-find
    callbudget eval       ECF (naive vs learned) + safety metrics
    callbudget demo       end-to-end walkthrough
    callbudget chart      render figures/ecf_flywheel.png
    callbudget research   constrained auto-search over model + extraction configs
    callbudget converse   hold a text conversation with the call agent

Uses argparse (stdlib) to avoid adding a dependency for a handful of commands.
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="callbudget", description="Active-sensing pharmacy search."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in [
        ("predict", "rank pharmacies by P(in_stock)"),
        ("plan", "call order + expected calls-to-find"),
        ("eval", "ECF (naive vs learned) + safety metrics"),
        ("demo", "end-to-end walkthrough"),
        ("chart", "render figures/ecf_flywheel.png"),
        ("serve", "run the web UI (needs: uv sync --extra web)"),
        ("converse", "hold a text conversation with the call agent"),
    ]:
        sub.add_parser(name, help=help_text)

    # `research` carries flags, so build it explicitly. Defaults are imported lazily
    # below to keep `--help` instant; mirror them here as literals only in the help text.
    research = sub.add_parser(
        "research", help="constrained auto-search over model + extraction configs"
    )
    research.add_argument("--rounds", type=int, default=None, help="eval rounds per config")
    research.add_argument("--trials", type=int, default=None, help="Optuna trials to explore")
    research.add_argument("--seed", type=int, default=7, help="sampler seed (reproducible study)")
    args = parser.parse_args(argv)

    # Import lazily so `--help` is instant and doesn't train a model.
    if args.cmd == "demo":
        from .demo import run_demo

        run_demo()
    elif args.cmd == "chart":
        from .demo import flywheel_chart

        print("wrote", flywheel_chart())
    elif args.cmd == "research":
        from .research.loop import DEFAULT_ROUNDS, DEFAULT_TRIALS, run_research

        run_research(
            rounds=args.rounds if args.rounds is not None else DEFAULT_ROUNDS,
            trials=args.trials if args.trials is not None else DEFAULT_TRIALS,
            seed=args.seed,
        )
    elif args.cmd == "serve":
        import uvicorn

        print("CallBudget UI on http://127.0.0.1:8000")
        uvicorn.run("callbudget.web.app:app", host="127.0.0.1", port=8000)
    elif args.cmd == "converse":
        import sys

        from .voice.dialogue import respond

        history: list[dict] = []
        turn = respond(history)  # the agent's opener (AI disclosure + recording consent)
        print(f"agent: {turn.reply}")
        history.append({"role": "agent", "text": turn.reply})
        print("(type the pharmacist's replies; empty line or Ctrl-D to hang up)")
        for line in sys.stdin:
            text = line.strip()
            if not text:
                break
            history.append({"role": "pharmacist", "text": text})
            turn = respond(history)
            print(f"agent: {turn.reply}")
            history.append({"role": "agent", "text": turn.reply})
            if turn.done:
                print(f"[{turn.verdict}]")
                break
    else:
        from .mcp.engine import CallBudgetEngine

        engine = CallBudgetEngine()
        result = {
            "predict": engine.predict_stock,
            "plan": engine.plan_calls,
            "eval": engine.run_eval,
        }[args.cmd]()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

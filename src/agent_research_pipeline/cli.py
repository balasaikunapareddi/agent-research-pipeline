"""Command-line entry point for the research pipeline.

    research-pipeline "Should a typical homeowner install rooftop solar panels?"
    research-pipeline --live "What should a first-time buyer know before purchasing an electric vehicle?"
    research-pipeline --resume "..."          # continue from the checkpoint
    research-pipeline --agent "..."           # run via the ADK composition

Mock mode is the default (no key, no network). ``--live`` uses Gemini
for decomposition and claim drafting; retrieval, verification, and
assembly stay deterministic. State checkpoints to
``research_checkpoint.json`` after every stage.
"""

from __future__ import annotations

import argparse
import sys

from .ledger import format_summary
from .pipeline import PipelineConfig, ResearchPipeline

DEFAULT_QUESTION = "Should a typical homeowner install rooftop solar panels?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-pipeline",
        description="Multi-agent research with claim-level fact-checking.",
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--live", action="store_true",
                        help="use Gemini for decomposition and claim drafting")
    parser.add_argument("--agent", action="store_true",
                        help="run through the ADK agent composition")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the checkpoint file")
    parser.add_argument("--checkpoint", default="research_checkpoint.json",
                        help="checkpoint file path")
    parser.add_argument("--ledger", default=None, help="ledger JSONL path")
    parser.add_argument("--keep-unverified", action="store_true",
                        help="keep unsupported claims marked [unverified]")
    parser.add_argument("--no-sabotage", action="store_true",
                        help="disable the mock mis-citation used to prove the gate")
    parser.add_argument("--top-k", type=int, default=3,
                        help="sources retrieved per sub-question")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        mode="live" if args.live else "mock",
        top_k=args.top_k,
        keep_unverified=args.keep_unverified,
        sabotage=not args.no_sabotage,
        checkpoint_path=args.checkpoint,
        ledger_path=args.ledger,
    )
    pipeline = ResearchPipeline(config=config)

    if args.agent:
        from .agent import run_agent

        print(run_agent(args.question, live=args.live, pipeline=pipeline))
        return 0

    try:
        result = pipeline.run(args.question, resume=args.resume)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result.resumed_from:
        print(f"(resumed from stage: {result.resumed_from})\n")
    print(result.report)
    print(format_summary(result.ledger_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

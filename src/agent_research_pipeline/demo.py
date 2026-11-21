"""One-command demo: a full research run, then a crash + resume.

    python -m agent_research_pipeline.demo

Part 1 runs one question end to end with an injected transient failure
in the research stage (watch it retry), prints the sub-questions,
evidence, fact-check verdicts (including the mis-cited claims the gate
drops), the final cited report, and the ledger.

Part 2 crashes the pipeline after the research stage, shows the JSON
checkpoint, then resumes from it and confirms the report matches Part 1.

Part 3 runs the same question through the ADK agent composition
(planner → parallel researchers → fact-checker → writer) on the scripted
mock LLM. Mock mode throughout: no API key, no network, deterministic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .agent import run_agent
from .ledger import format_summary
from .pipeline import PipelineConfig, ResearchPipeline, SimulatedCrash

DEMO_QUESTION = "Should a typical homeowner install rooftop solar panels?"


def _print_run(result, title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"Question: {result.question}\n")
    print("Sub-questions (planner):")
    for i, sq in enumerate(result.sub_questions, 1):
        print(f"  {i}. {sq}")
    print("\nEvidence (parallel researchers):")
    for sq, hits in result.evidence.items():
        ids = ", ".join(f"{h.doc_id} ({h.score:.2f})" for h in hits)
        print(f"  {sq}\n    -> {ids or 'no hits'}")
    print("\nFact-check gate:")
    for v in result.verified:
        mark = "SUPPORTED" if v.supported else "DROPPED  "
        print(f"  [{mark}] [{v.claim.source_id}] {v.claim.text[:80]}...")
        if not v.supported:
            print(f"             reason: {v.reason}")
    print("\n" + result.report)
    print(format_summary(result.ledger_summary))


def run_demo(workdir: str | Path | None = None) -> dict:
    """Run all three demo parts; return artifacts (used by tests too)."""
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="research-demo-"))

    # Part 1: full run with an injected transient failure in research.
    p1 = ResearchPipeline(
        config=PipelineConfig(
            checkpoint_path=workdir / "checkpoint.json",
            ledger_path=workdir / "ledger.jsonl",
            inject_failure=True,
            backoff_s=0.05,
        )
    )
    result1 = p1.run(DEMO_QUESTION)

    # Part 2: crash after research, then resume from the checkpoint.
    crash_ckpt = workdir / "crash_checkpoint.json"
    p2 = ResearchPipeline(
        config=PipelineConfig(
            checkpoint_path=crash_ckpt,
            crash_after_stage="research",
            backoff_s=0.05,
        )
    )
    crashed = False
    try:
        p2.run(DEMO_QUESTION)
    except SimulatedCrash:
        crashed = True
    checkpoint = json.loads(crash_ckpt.read_text())
    p3 = ResearchPipeline(
        config=PipelineConfig(checkpoint_path=crash_ckpt, backoff_s=0.05)
    )
    result2 = p3.run(DEMO_QUESTION, resume=True)

    # Part 3: the ADK composition on the scripted mock LLM.
    agent_pipeline = ResearchPipeline(
        config=PipelineConfig(checkpoint_path=None, backoff_s=0.05)
    )
    agent_report = run_agent(DEMO_QUESTION, pipeline=agent_pipeline)

    return {
        "result": result1,
        "crashed": crashed,
        "checkpoint": checkpoint,
        "resumed_result": result2,
        "agent_report": agent_report,
        "workdir": workdir,
    }


def main() -> int:
    artifacts = run_demo()
    result = artifacts["result"]

    print("PART 1 — full run (research stage fails once, retries, succeeds)")
    _print_run(result, "RESEARCH PIPELINE DEMO")
    research_attempts = result.ledger_summary["stages"]["research"]["attempts"]
    print(f"\nResearch stage attempts: {research_attempts} (1 injected failure + retry)")

    print("\n" + "=" * 72)
    print("PART 2 — crash after research, then resume from checkpoint")
    print("=" * 72)
    print(f"Crashed: {artifacts['crashed']}")
    print(f"Checkpoint completed stages: {artifacts['checkpoint']['completed_stages']}")
    resumed = artifacts["resumed_result"]
    print(f"Resumed from stage: {resumed.resumed_from}")
    print(f"Resumed report matches Part 1 report: {resumed.report == result.report}")

    print("\n" + "=" * 72)
    print("PART 3 — same question through the ADK agent composition")
    print("=" * 72)
    agent_report = artifacts["agent_report"]
    print("\n".join(agent_report.splitlines()[:12]))
    print(f"... ({len(agent_report.splitlines())} lines total)")
    print(f"ADK report contains citations: {'[' in agent_report and ']' in agent_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

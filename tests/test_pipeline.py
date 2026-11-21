import json

import pytest

from agent_research_pipeline.pipeline import (
    PipelineConfig,
    ResearchPipeline,
    SimulatedCrash,
    TransientError,
)

Q = "Should a typical homeowner install rooftop solar panels?"


def _pipeline(tmp_path, **kw) -> ResearchPipeline:
    config = PipelineConfig(
        checkpoint_path=tmp_path / "checkpoint.json",
        ledger_path=tmp_path / "ledger.jsonl",
        backoff_s=0.01,
        **kw,
    )
    return ResearchPipeline(config=config)


def test_full_run_produces_cited_report(tmp_path):
    result = _pipeline(tmp_path).run(Q)
    assert result.sub_questions
    assert result.claims
    assert "# Research Report" in result.report
    assert "## Verification summary" in result.report
    assert "## Sources" in result.report
    assert "[solar-" in result.report
    # The sabotaged mis-citations are dropped, never in the body.
    dropped = [v for v in result.verified if not v.supported]
    assert len(dropped) == 2
    for v in dropped:
        assert f"- {v.claim.text} [{v.claim.source_id}]" not in result.report
    assert result.completed_stages == ["plan", "research", "claims", "factcheck", "write"]


def test_checkpoint_written_with_all_stages(tmp_path):
    _pipeline(tmp_path).run(Q)
    data = json.loads((tmp_path / "checkpoint.json").read_text())
    assert data["question"] == Q
    assert data["completed_stages"] == ["plan", "research", "claims", "factcheck", "write"]
    assert data["state"]["report"]


def test_crash_then_resume(tmp_path):
    crashing = _pipeline(tmp_path, crash_after_stage="research")
    with pytest.raises(SimulatedCrash):
        crashing.run(Q)
    data = json.loads((tmp_path / "checkpoint.json").read_text())
    assert data["completed_stages"] == ["plan", "research"]

    resumed = _pipeline(tmp_path).run(Q, resume=True)
    assert resumed.resumed_from == "claims"

    fresh = _pipeline(tmp_path).run(Q)
    assert resumed.report == fresh.report


def test_resume_wrong_question_raises(tmp_path):
    _pipeline(tmp_path).run(Q)
    with pytest.raises(ValueError):
        _pipeline(tmp_path).run("A different question entirely?", resume=True)


def test_resume_without_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _pipeline(tmp_path).run(Q, resume=True)


def test_injected_failure_retries_and_recovers(tmp_path):
    result = _pipeline(tmp_path, inject_failure=True).run(Q)
    research = result.ledger_summary["stages"]["research"]
    assert research["attempts"] == 2
    assert result.ledger_summary["failures"] == 1
    assert result.ledger_summary["retries"] == 1
    assert result.report


def test_retry_exhaustion_raises(tmp_path):
    pipeline = _pipeline(tmp_path, inject_failure=True, max_attempts=1)
    with pytest.raises(TransientError):
        pipeline.run(Q)


def test_keep_unverified_marks_claims(tmp_path):
    result = _pipeline(tmp_path, keep_unverified=True).run(Q)
    assert "[unverified]" in result.report
    dropped = [v for v in result.verified if not v.supported]
    for v in dropped:
        assert f"[{v.claim.source_id}] [unverified]" in result.report


def test_no_sabotage_no_dropped(tmp_path):
    result = _pipeline(tmp_path, sabotage=False).run(Q)
    assert all(v.supported for v in result.verified)


def test_run_is_deterministic(tmp_path):
    a = _pipeline(tmp_path).run(Q)
    b = _pipeline(tmp_path).run(Q)
    assert a.report == b.report
    assert a.sub_questions == b.sub_questions

import pytest

from agent_research_pipeline.agent import build_agent, run_agent
from agent_research_pipeline.pipeline import PipelineConfig, ResearchPipeline

Q = "Should a typical homeowner install rooftop solar panels?"


def _pipeline() -> ResearchPipeline:
    return ResearchPipeline(
        config=PipelineConfig(checkpoint_path=None, backoff_s=0.01)
    )


def test_agent_loop_produces_cited_report():
    report = run_agent(Q, pipeline=_pipeline())
    assert "# Research Report" in report
    assert "## Verification summary" in report
    assert "[solar-" in report
    assert "Dropped as unsupported: 2" in report


def test_agent_report_matches_direct_pipeline():
    direct = _pipeline().run(Q).report
    via_agent = run_agent(Q, pipeline=_pipeline())
    assert via_agent == direct


def test_build_agent_composition_shape():
    agent = build_agent(pipeline=_pipeline())
    assert agent.name == "research_pipeline"
    names = [a.name for a in agent.sub_agents]
    assert names == ["planner", "researchers", "fact_checker", "writer"]
    researchers = agent.sub_agents[1]
    assert len(researchers.sub_agents) == 4


def test_live_without_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_agent(live=True, pipeline=_pipeline())

"""The ADK composition: planner → parallel researchers → fact-checker → writer.

This is the shape ADK is built for: a ``SequentialAgent`` chains the
stages, a ``ParallelAgent`` fans the researchers out, and each role is an
``LlmAgent`` with a behavior-only prompt and typed tools over the same
deterministic pipeline the CLI/demo run. Mock mode (default) runs every
role on a ``ScriptedLlm`` so the full composition works offline; live
mode (``live=True`` with GEMINI_API_KEY set) swaps in a real Gemini
model for every role — the tools, retrieval, and fact-check gate are
unchanged.
"""

from __future__ import annotations

import asyncio
import os

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from .mock_llm import ScriptedLlm
from .pipeline import PipelineConfig, ResearchPipeline
from .prompts import (
    FACT_CHECKER_PROMPT,
    PLANNER_PROMPT,
    RESEARCHER_PROMPT_TEMPLATE,
    WRITER_PROMPT,
)
from .tools import (
    assemble_report,
    decompose_question,
    extract_and_verify_claims,
    search_sources,
    set_default_pipeline,
    verify_claim,
)

APP_NAME = "agent_research_pipeline"
NUM_RESEARCHERS = 4
LIVE_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def build_agent(
    live: bool = False,
    pipeline: ResearchPipeline | None = None,
) -> SequentialAgent:
    """Build the research pipeline agent. Mock by default; live needs a key."""
    if live:
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not set; live mode unavailable")
        model: str | ScriptedLlm = LIVE_MODEL
        default_pipeline = pipeline or ResearchPipeline(
            config=PipelineConfig(mode="live", checkpoint_path=None)
        )
    else:
        model = ScriptedLlm()
        default_pipeline = pipeline or ResearchPipeline(
            config=PipelineConfig(mode="mock", checkpoint_path=None)
        )
    set_default_pipeline(default_pipeline)

    planner = LlmAgent(
        name="planner",
        description="Splits a research question into sub-questions.",
        model=model,
        instruction=PLANNER_PROMPT,
        tools=[FunctionTool(decompose_question)],
        output_key="plan",
    )
    researchers = [
        LlmAgent(
            name=f"researcher_{i}",
            description=f"Gathers evidence for sub-question {i}.",
            model=model if live else ScriptedLlm(),
            instruction=RESEARCHER_PROMPT_TEMPLATE.replace("__INDEX__", str(i)),
            tools=[FunctionTool(search_sources)],
            output_key=f"evidence_{i}",
        )
        for i in range(1, NUM_RESEARCHERS + 1)
    ]
    fact_checker = LlmAgent(
        name="fact_checker",
        description="Verifies every claim against its cited source.",
        model=model if live else ScriptedLlm(),
        instruction=FACT_CHECKER_PROMPT,
        tools=[FunctionTool(extract_and_verify_claims), FunctionTool(verify_claim)],
        output_key="verified_claims",
    )
    writer = LlmAgent(
        name="writer",
        description="Assembles the final cited report from verified claims.",
        model=model if live else ScriptedLlm(),
        instruction=WRITER_PROMPT,
        tools=[FunctionTool(assemble_report)],
        output_key="report",
    )
    return SequentialAgent(
        name="research_pipeline",
        description="Multi-agent research with claim-level fact-checking.",
        sub_agents=[
            planner,
            ParallelAgent(
                name="researchers",
                description="Parallel evidence gathering, one researcher per sub-question.",
                sub_agents=researchers,
            ),
            fact_checker,
            writer,
        ],
    )


async def _run_async(agent: SequentialAgent, question: str) -> str:
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session = await session_service.create_session(app_name=APP_NAME, user_id="demo")
    content = types.Content(role="user", parts=[types.Part(text=question)])
    final_text = ""
    async for event in runner.run_async(
        user_id="demo", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text
    return final_text


def run_agent(
    question: str,
    live: bool = False,
    pipeline: ResearchPipeline | None = None,
) -> str:
    """Run one research question through the ADK composition; return the report."""
    agent = build_agent(live=live, pipeline=pipeline)
    return asyncio.run(_run_async(agent, question))

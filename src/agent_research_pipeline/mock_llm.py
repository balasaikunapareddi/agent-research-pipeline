"""Deterministic scripted LLM for the ADK agent loop (no key, no network).

Follows the same contract as the role prompts: the planner calls
``decompose_question`` once, each researcher calls ``search_sources`` for
its assigned sub-question, the fact-checker calls
``extract_and_verify_claims``, and the writer calls ``assemble_report``.
After a tool returns, the scripted model formats the final reply from the
tool result. This makes the *agent composition itself* testable offline —
the same reason the model-router repo ships a scripted model.
"""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types


def _instruction_text(request: LlmRequest) -> str:
    cfg = getattr(request, "config", None)
    si = getattr(cfg, "system_instruction", None) if cfg else None
    if isinstance(si, str):
        return si
    if si is not None:
        parts = getattr(si, "parts", None) or []
        text = " ".join(p.text for p in parts if getattr(p, "text", None))
        if text:
            return text
    for content in request.contents or []:
        if content.role == "system":
            return " ".join(
                p.text for p in (content.parts or []) if p.text
            )
    return ""


def _last_user_text(request: LlmRequest) -> str:
    for content in reversed(request.contents or []):
        if content.role == "user":
            for part in content.parts or []:
                if part.text:
                    return part.text
    return ""


def _last_function_response(request: LlmRequest) -> tuple[str, dict] | None:
    for content in reversed(request.contents or []):
        for part in content.parts or []:
            if part.function_response is not None:
                return part.function_response.name, dict(
                    part.function_response.response or {}
                )
    return None


def _role_of(instruction: str) -> str:
    for role in ("planner", "researcher", "fact-checker", "writer"):
        if f"Role: {role}" in instruction:
            return role
    return "planner"


def _researcher_assignment(instruction: str) -> str | None:
    """This researcher's sub-question, from the injected plan, or None."""
    idx_match = re.search(r"Researcher #(\d+)", instruction)
    if not idx_match:
        return None
    index = int(idx_match.group(1)) - 1
    array_match = re.search(r"\[.*\]", instruction, re.DOTALL)
    if not array_match:
        return None
    try:
        plan = json.loads(array_match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(plan, list) and 0 <= index < len(plan):
        return str(plan[index])
    return None


def _format_final(tool_name: str, result: dict) -> str:
    if tool_name == "decompose_question":
        return json.dumps(result.get("sub_questions") or [])
    if tool_name == "search_sources":
        results = result.get("results") or []
        if not results:
            return "No evidence found."
        return "\n".join(
            f"[{r.get('doc_id')}] {r.get('snippet')}" for r in results
        )
    if tool_name == "extract_and_verify_claims":
        return json.dumps(result)
    if tool_name == "assemble_report":
        return str(result.get("report") or "")
    return str(result)


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        turn_complete=True,
    )


def _call_response(tool_name: str, args: dict) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name=tool_name, args=args)
                )
            ],
        ),
        turn_complete=True,
    )


class ScriptedLlm(BaseLlm):
    """A BaseLlm that deterministically drives the pipeline tools by role."""

    model: str = "mock-research-llm"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"mock-.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        tool_result = _last_function_response(llm_request)
        if tool_result is not None:
            tool_name, result = tool_result
            yield _text_response(_format_final(tool_name, result))
            return

        instruction = _instruction_text(llm_request)
        role = _role_of(instruction)
        if role == "planner":
            yield _call_response(
                "decompose_question", {"question": _last_user_text(llm_request)}
            )
        elif role == "researcher":
            assignment = _researcher_assignment(instruction)
            if assignment is None:
                yield _text_response("No sub-question assigned.")
            else:
                yield _call_response("search_sources", {"query": assignment})
        elif role == "fact-checker":
            yield _call_response("extract_and_verify_claims", {})
        else:  # writer
            yield _call_response("assemble_report", {})

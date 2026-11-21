"""Live-mode LLM calls via google-genai (REST).

Only two pipeline steps use the model in live mode: decomposition
(planner) and claim drafting (writer). Both are parsed by deterministic
code and fall back to the mock rules on any failure, so a bad reply or a
flaky network degrades the run instead of crashing it. Retrieval,
verification, and report assembly are identical in both modes.

Imported lazily: mock mode never touches this module.
"""

from __future__ import annotations

import os

from .claims import Claim, parse_claim_lines
from .decomposition import parse_live_decomposition
from .ledger import estimate_tokens
from .pipeline import TransientError
from .retrieval import SearchResult

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def live_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; live mode unavailable")
    from google import genai

    return genai.Client(api_key=api_key)


def _generate(prompt: str) -> tuple[str, dict]:
    """One generate_content call; returns (text, usage)."""
    client = _client()
    try:
        response = client.models.generate_content(
            model=live_model_name(), contents=prompt
        )
    except Exception as exc:  # noqa: BLE001 - normalized for the retry loop
        raise TransientError(f"Gemini call failed: {exc}") from exc
    text = response.text or ""
    usage = {"input_tokens": estimate_tokens(prompt),
             "output_tokens": estimate_tokens(text)}
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        usage["input_tokens"] = getattr(meta, "prompt_token_count", usage["input_tokens"]) or usage["input_tokens"]
        usage["output_tokens"] = getattr(meta, "candidates_token_count", usage["output_tokens"]) or usage["output_tokens"]
    return text, usage


def live_decompose(question: str) -> tuple[list[str], dict]:
    """Ask the planner LLM for sub-questions; parse deterministically."""
    prompt = (
        "You are the planner in a research pipeline. Split the research "
        "question below into 4 to 6 focused sub-questions that researchers "
        "can answer independently. Cover cost, benefits, risks, and "
        "practical requirements where relevant.\n\n"
        f"Research question: {question}\n\n"
        "Reply with ONLY a JSON array of strings, e.g. "
        '["Sub-question one?", "Sub-question two?"]'
    )
    text, usage = _generate(prompt)
    return parse_live_decomposition(text, question), usage


def live_draft_claims(
    question: str,
    evidence: dict[str, list[SearchResult]],
) -> tuple[list[Claim], dict]:
    """Ask the writer LLM to draft claims from evidence, each cited.

    Claims whose citation is not one of the evidence source ids are
    discarded by the parser. Each surviving claim is assigned to the
    first sub-question whose evidence contained its source.
    """
    evidence_lines = []
    source_to_sub: dict[str, str] = {}
    for sub_question, results in evidence.items():
        for r in results:
            evidence_lines.append(f"[{r.doc_id}] {r.snippet}")
            source_to_sub.setdefault(r.doc_id, sub_question)
    prompt = (
        "You are a research writer. Using ONLY the evidence below, write "
        "factual claims that help answer the research question. Rules:\n"
        "- One claim per line.\n"
        "- End every line with the id of the evidence source that supports "
        "it, in square brackets, like [solar-install-cost].\n"
        "- Do not invent facts or numbers that are not in the evidence.\n"
        "- Write at most 12 claims.\n\n"
        f"Research question: {question}\n\nEvidence:\n"
        + "\n".join(evidence_lines)
    )
    text, usage = _generate(prompt)
    claims = parse_claim_lines(text, set(source_to_sub))
    for claim in claims:
        claim.sub_question = source_to_sub.get(claim.source_id, "")
    return claims, usage

"""Typed tool definitions exposed to the ADK agents.

Plain functions with type hints and docstrings — ADK turns them into tool
schemas. All decision logic stays in decomposition/retrieval/claims/
pipeline; these are thin, typed wrappers over a shared pipeline instance
(the same injection pattern as the model-router repo's default router).
"""

from __future__ import annotations

from .claims import Claim
from .claims import verify_claim as _verify_claim
from .pipeline import PipelineConfig, ResearchPipeline

_default_pipeline: ResearchPipeline | None = None


def set_default_pipeline(pipeline: ResearchPipeline) -> None:
    """Inject the pipeline the tools use (tests and demo set their own)."""
    global _default_pipeline
    _default_pipeline = pipeline


def get_default_pipeline() -> ResearchPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ResearchPipeline(
            config=PipelineConfig(checkpoint_path=None)
        )
    return _default_pipeline


def decompose_question(question: str) -> dict:
    """Split a research question into focused sub-questions.

    Args:
        question: The full research question.

    Returns:
        Dict with ``sub_questions``: a list of sub-question strings.
    """
    pipeline = get_default_pipeline()
    return {"sub_questions": pipeline.plan(question)}


def search_sources(query: str, top_k: int = 3) -> dict:
    """Search the source corpus for evidence on one sub-question.

    Args:
        query: The sub-question to gather evidence for.
        top_k: Maximum number of sources to return.

    Returns:
        Dict with ``results``: source id, title, snippet, and score per hit.
    """
    pipeline = get_default_pipeline()
    results = pipeline.search(query, top_k=top_k)
    return {"results": [r.to_dict() for r in results]}


def extract_and_verify_claims() -> dict:
    """Extract claims from gathered evidence and verify every citation.

    Returns:
        Dict with ``verified`` (supported claims), ``dropped``
        (unsupported claims with reasons), and a ``summary`` of counts.
    """
    pipeline = get_default_pipeline()
    verified = pipeline.extract_and_verify()
    supported = [v.to_dict() for v in verified if v.supported]
    dropped = [v.to_dict() for v in verified if not v.supported]
    return {
        "verified": supported,
        "dropped": dropped,
        "summary": {
            "claims_checked": len(verified),
            "supported": len(supported),
            "dropped": len(dropped),
        },
    }


def verify_claim(claim_text: str, source_id: str) -> dict:
    """Check whether one claim is supported by its cited source.

    Args:
        claim_text: The factual claim to check.
        source_id: The id of the source the claim cites.

    Returns:
        Dict with ``supported`` (bool), ``score`` (token overlap), and
        ``reason``.
    """
    pipeline = get_default_pipeline()
    result = _verify_claim(
        Claim(text=claim_text, source_id=source_id), pipeline.corpus
    )
    return {
        "supported": result.supported,
        "score": round(result.score, 4),
        "reason": result.reason,
    }


def assemble_report() -> dict:
    """Assemble the final cited report from verified claims.

    Returns:
        Dict with ``report``: the full report text. Only supported claims
        appear in the body, each with its citation; dropped claims are
        listed in the verification summary.
    """
    pipeline = get_default_pipeline()
    return {"report": pipeline.assemble()}


def run_research(question: str) -> dict:
    """Run the full research pipeline end to end on one question.

    Args:
        question: The research question.

    Returns:
        Dict with sub-questions, claim counts, the verification summary,
        the final report, and the run ledger summary.
    """
    pipeline = get_default_pipeline()
    result = pipeline.run(question)
    supported = sum(1 for v in result.verified if v.supported)
    return {
        "question": result.question,
        "sub_questions": result.sub_questions,
        "claims_checked": len(result.verified),
        "supported": supported,
        "dropped": len(result.verified) - supported,
        "report": result.report,
        "ledger": result.ledger_summary,
    }

"""Claim extraction and citation verification — the fact-check gate.

A claim is a single factual sentence plus the id of the source it cites.
Verification is deterministic text analysis, never an LLM judgment: a
claim is supported only if its content words substantially overlap the
cited source's text AND every number in the claim appears in that source.
Unsupported claims are dropped (or marked [unverified]) by the report
assembler — they are never silently kept.

Mock claim extraction turns evidence snippets into claims. To prove the
gate works, extraction can deterministically mis-cite two claims
(``sabotage=True``) — simulating the classic LLM failure of attaching a
plausible but wrong citation. The gate must catch both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus
from .retrieval import SearchResult, content_tokens, ranked_sentences

SUPPORT_THRESHOLD = 0.5

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_CITED_LINE_RE = re.compile(r"^(.*?)\s*\[([a-z0-9][a-z0-9-]*)\]\s*$")


@dataclass
class Claim:
    text: str
    source_id: str
    sub_question: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_id": self.source_id,
            "sub_question": self.sub_question,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        return cls(
            text=data["text"],
            source_id=data["source_id"],
            sub_question=data.get("sub_question", ""),
        )


@dataclass
class VerifiedClaim:
    claim: Claim
    supported: bool
    score: float
    reason: str

    def to_dict(self) -> dict:
        data = self.claim.to_dict()
        data.update(
            {
                "supported": self.supported,
                "score": round(self.score, 4),
                "reason": self.reason,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "VerifiedClaim":
        return cls(
            claim=Claim.from_dict(data),
            supported=bool(data["supported"]),
            score=float(data["score"]),
            reason=data.get("reason", ""),
        )


def extract_claims(
    evidence: dict[str, list[SearchResult]],
    corpus: Corpus | None = None,
    sabotage: bool = False,
) -> list[Claim]:
    """Turn evidence into claims (mock mode).

    Up to two claims per retrieved source: its best-matching sentences,
    each citing its own document. With ``sabotage=True`` (and a corpus to
    check against) two claims from different sub-questions get their
    citations swapped — a deterministic stand-in for an LLM mis-citation
    that the fact-check gate must catch. The swap is chosen with the real
    verifier so the mis-citation is guaranteed detectable.
    """
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()
    for sub_question, results in evidence.items():
        for result in results:
            sentences = [result.snippet]
            if corpus is not None:
                doc = corpus.get(result.doc_id)
                if doc is not None:
                    sentences = ranked_sentences(sub_question, doc, n=2)
            for sentence in sentences:
                # Dedupe within a sub-question only: the same sentence can
                # legitimately support two different sub-questions.
                key = (sub_question, sentence, result.doc_id)
                if key in seen:
                    continue
                seen.add(key)
                claims.append(
                    Claim(
                        text=sentence,
                        source_id=result.doc_id,
                        sub_question=sub_question,
                    )
                )
    if sabotage and corpus is not None:
        _sabotage_citations(claims, corpus)
    return claims


def _sabotage_citations(claims: list[Claim], corpus: Corpus) -> None:
    """Swap two citations so both claims become unsupported (verifier-checked)."""
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            if a.sub_question == b.sub_question or a.source_id == b.source_id:
                continue
            swapped_a = Claim(a.text, b.source_id, a.sub_question)
            swapped_b = Claim(b.text, a.source_id, b.sub_question)
            if (
                not verify_claim(swapped_a, corpus).supported
                and not verify_claim(swapped_b, corpus).supported
            ):
                a.source_id, b.source_id = b.source_id, a.source_id
                return


def parse_claim_lines(text: str, valid_ids: set[str]) -> list[Claim]:
    """Parse live-mode claim drafts: one claim per line, ending ``[id]``.

    Lines whose citation is not a known source id are discarded here —
    an invented citation never reaches the verifier.
    """
    claims: list[Claim] = []
    for line in text.splitlines():
        line = line.strip(" -*•")
        m = _CITED_LINE_RE.match(line)
        if not m:
            continue
        claim_text, source_id = m.group(1).strip(), m.group(2)
        if source_id in valid_ids and len(claim_text) >= 15:
            claims.append(Claim(text=claim_text, source_id=source_id))
    return claims


def _numbers_in(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(text)}


def verify_claim(claim: Claim, corpus: Corpus) -> VerifiedClaim:
    """Deterministically check a claim against its cited source."""
    doc = corpus.get(claim.source_id)
    if doc is None:
        return VerifiedClaim(
            claim=claim,
            supported=False,
            score=0.0,
            reason=f"cited source '{claim.source_id}' does not exist",
        )
    claim_tokens = content_tokens(claim.text)
    if not claim_tokens:
        return VerifiedClaim(
            claim=claim, supported=False, score=0.0, reason="claim has no content"
        )
    doc_tokens = content_tokens(f"{doc.title} {doc.text}")
    overlap = len(claim_tokens & doc_tokens) / len(claim_tokens)
    claim_numbers = _numbers_in(claim.text)
    doc_numbers = _numbers_in(doc.text)
    missing_numbers = claim_numbers - doc_numbers
    if missing_numbers:
        return VerifiedClaim(
            claim=claim,
            supported=False,
            score=overlap,
            reason=(
                f"numbers {sorted(missing_numbers)} not found in "
                f"source '{doc.id}'"
            ),
        )
    supported = overlap >= SUPPORT_THRESHOLD
    reason = (
        f"token overlap {overlap:.2f} with source '{doc.id}'"
        if supported
        else f"token overlap {overlap:.2f} below threshold "
        f"{SUPPORT_THRESHOLD} for source '{doc.id}'"
    )
    return VerifiedClaim(
        claim=claim, supported=supported, score=overlap, reason=reason
    )


def verify_claims(claims: list[Claim], corpus: Corpus) -> list[VerifiedClaim]:
    return [verify_claim(c, corpus) for c in claims]

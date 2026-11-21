"""Report assembly: only supported claims ship, with inline citations.

The assembler is the last line of the fact-check gate. It receives
verified claims and:
- includes supported claims in the body, each with its ``[source-id]``
  citation, grouped under its sub-question;
- lists dropped (unsupported) claims in the verification summary with
  the reason they failed — visible, never silently kept;
- with ``keep_unverified=True``, includes unsupported claims in the body
  explicitly marked ``[unverified]`` instead of dropping them.

``parse_report_claims`` reads a report back into (text, citation) pairs
so evals can prove citation precision and a zero unsupported rate.
"""

from __future__ import annotations

import re

from .claims import VerifiedClaim
from .corpus import Corpus

_CLAIM_LINE_RE = re.compile(r"^- (.+?) \[([a-z0-9][a-z0-9-]*)\]( \[unverified\])?$")


def assemble_report(
    question: str,
    sub_questions: list[str],
    verified: list[VerifiedClaim],
    corpus: Corpus,
    keep_unverified: bool = False,
) -> str:
    supported = [v for v in verified if v.supported]
    dropped = [v for v in verified if not v.supported]

    lines: list[str] = [
        "# Research Report",
        "",
        f"Question: {question}",
        "",
    ]
    for sub_question in sub_questions:
        lines.append(f"## {sub_question}")
        in_section = [
            v for v in supported if v.claim.sub_question == sub_question
        ]
        if keep_unverified:
            in_section += [
                v for v in dropped if v.claim.sub_question == sub_question
            ]
        if not in_section:
            lines.append("- No supported evidence found for this sub-question.")
        for v in in_section:
            marker = "" if v.supported else " [unverified]"
            lines.append(f"- {v.claim.text} [{v.claim.source_id}]{marker}")
        lines.append("")

    lines.append("## Verification summary")
    lines.append(f"Claims checked: {len(verified)}")
    lines.append(f"Supported: {len(supported)}")
    lines.append(f"Dropped as unsupported: {len(dropped)}")
    if dropped:
        lines.append("Dropped claims:")
        for v in dropped:
            lines.append(
                f'- "{v.claim.text}" [cited: {v.claim.source_id}] — {v.reason}'
            )
    lines.append("")

    cited_ids = sorted({v.claim.source_id for v in supported})
    lines.append("## Sources")
    for doc_id in cited_ids:
        doc = corpus.get(doc_id)
        title = doc.title if doc else "(unknown source)"
        lines.append(f"- [{doc_id}] {title}")
    lines.append("")
    return "\n".join(lines)


def parse_report_claims(report: str) -> list[tuple[str, str, bool]]:
    """Extract (claim_text, citation, is_unverified) from report body lines.

    Only lines in sub-question sections count — the dropped-claims list in
    the verification summary uses a different format and is excluded.
    """
    claims: list[tuple[str, str, bool]] = []
    in_body = False
    for line in report.splitlines():
        if line.startswith("## "):
            in_body = line.strip() not in (
                "## Verification summary",
                "## Sources",
            )
            continue
        if not in_body:
            continue
        m = _CLAIM_LINE_RE.match(line.strip())
        if m:
            claims.append((m.group(1), m.group(2), bool(m.group(3))))
    return claims

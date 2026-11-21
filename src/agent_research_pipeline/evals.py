"""Bundled evals: decomposition coverage, citation precision, gate integrity.

Three research questions with expected decomposition facets and must-cite
sources. Each case runs the full pipeline (mock mode) and checks:

- **Decomposition coverage**: every expected facet appears in some
  sub-question.
- **Citation precision**: every citation in the final report resolves to
  a real corpus document (must be 1.0).
- **Unsupported-claim rate**: every claim in the report body verifies as
  supported against its cited source (must be 0 unsupported).
- **Must-cite**: the expected key sources are actually cited.

Run ``python -m agent_research_pipeline.evals`` (or ``make eval``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .claims import Claim, verify_claim
from .corpus import Corpus, load_eval_cases
from .pipeline import PipelineConfig, ResearchPipeline
from .report import parse_report_claims


@dataclass
class EvalResult:
    case_id: str
    question: str
    decomposition_coverage: float
    citation_precision: float
    unsupported_in_report: int
    must_cite_missing: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_case(case: dict, corpus: Corpus | None = None) -> EvalResult:
    corpus = corpus or Corpus.load_default()
    pipeline = ResearchPipeline(
        corpus=corpus,
        config=PipelineConfig(mode="mock", checkpoint_path=None, backoff_s=0.01),
    )
    result = pipeline.run(case["question"])
    failures: list[str] = []

    joined = " ".join(result.sub_questions).lower()
    facets = case["expected_facets"]
    covered = [f for f in facets if f.lower() in joined]
    coverage = len(covered) / len(facets) if facets else 1.0
    if coverage < 1.0:
        failures.append(f"decomposition missing facets: {sorted(set(facets) - set(covered))}")

    report_claims = parse_report_claims(result.report)
    citations = [cid for _, cid, _ in report_claims]
    resolved = [c for c in citations if c in corpus]
    precision = len(resolved) / len(citations) if citations else 0.0
    if precision < 1.0:
        failures.append(f"citation precision {precision:.2f} < 1.0")

    unsupported = 0
    for text, cid, unverified in report_claims:
        if unverified:
            continue
        check = verify_claim(Claim(text=text, source_id=cid), corpus)
        if not check.supported:
            unsupported += 1
    if unsupported:
        failures.append(f"{unsupported} unsupported claim(s) in final report")

    must_cite_missing = [m for m in case["must_cite"] if m not in citations]
    if must_cite_missing:
        failures.append(f"must-cite sources not cited: {must_cite_missing}")

    return EvalResult(
        case_id=case["id"],
        question=case["question"],
        decomposition_coverage=round(coverage, 3),
        citation_precision=round(precision, 3),
        unsupported_in_report=unsupported,
        must_cite_missing=must_cite_missing,
        failures=failures,
    )


def run_evals() -> list[EvalResult]:
    corpus = Corpus.load_default()
    return [evaluate_case(case, corpus) for case in load_eval_cases()]


def main(argv: list[str] | None = None) -> int:
    results = run_evals()
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        all_passed = all_passed and r.passed
        print(
            f"[{status}] {r.case_id}: coverage={r.decomposition_coverage} "
            f"precision={r.citation_precision} "
            f"unsupported={r.unsupported_in_report}"
        )
        for failure in r.failures:
            print(f"       - {failure}")
    print(f"\n{sum(r.passed for r in results)}/{len(results)} eval cases passed")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

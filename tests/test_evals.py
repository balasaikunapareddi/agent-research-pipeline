from agent_research_pipeline.corpus import load_eval_cases
from agent_research_pipeline.evals import evaluate_case, run_evals


def test_eval_cases_bundled():
    cases = load_eval_cases()
    assert len(cases) == 3
    for case in cases:
        assert case["question"]
        assert case["expected_facets"]
        assert case["must_cite"]


def test_all_eval_cases_pass():
    results = run_evals()
    assert len(results) == 3
    for r in results:
        assert r.passed, f"{r.case_id}: {r.failures}"
        assert r.decomposition_coverage == 1.0
        assert r.citation_precision == 1.0
        assert r.unsupported_in_report == 0
        assert r.must_cite_missing == []


def test_evaluate_case_reports_failures():
    case = {
        "id": "bogus",
        "question": "Should a typical homeowner install rooftop solar panels?",
        "expected_facets": ["nonexistent-facet-xyz"],
        "must_cite": ["no-such-source"],
    }
    result = evaluate_case(case)
    assert not result.passed
    assert result.decomposition_coverage == 0.0
    assert result.must_cite_missing == ["no-such-source"]

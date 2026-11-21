from agent_research_pipeline.claims import Claim, VerifiedClaim
from agent_research_pipeline.corpus import Corpus
from agent_research_pipeline.report import assemble_report, parse_report_claims

SQ = "What is the cost of rooftop solar panels?"


def _verified() -> list[VerifiedClaim]:
    return [
        VerifiedClaim(
            claim=Claim(
                text="A typical installation costs $18,000 to $25,000.",
                source_id="solar-install-cost",
                sub_question=SQ,
            ),
            supported=True, score=0.9, reason="ok",
        ),
        VerifiedClaim(
            claim=Claim(
                text="Solar panels are made of cheese and cost $5.",
                source_id="solar-install-cost",
                sub_question=SQ,
            ),
            supported=False, score=0.1, reason="token overlap too low",
        ),
    ]


def test_report_includes_supported_with_citation():
    report = assemble_report("Q?", [SQ], _verified(), Corpus.load_default())
    assert "- A typical installation costs $18,000 to $25,000. [solar-install-cost]" in report
    assert "Claims checked: 2" in report
    assert "Supported: 1" in report
    assert "Dropped as unsupported: 1" in report
    assert "[solar-install-cost] Rooftop Solar Installation Costs in 2026" in report


def test_report_drops_unsupported_from_body():
    report = assemble_report("Q?", [SQ], _verified(), Corpus.load_default())
    body = report.split("## Verification summary")[0]
    assert "cheese" not in body
    # ...but the dropped claim is visible in the summary, with its reason.
    assert "cheese" in report
    assert "token overlap too low" in report


def test_keep_unverified_marks_in_body():
    report = assemble_report(
        "Q?", [SQ], _verified(), Corpus.load_default(), keep_unverified=True
    )
    body = report.split("## Verification summary")[0]
    assert "[solar-install-cost] [unverified]" in body


def test_parse_report_claims_round_trip():
    report = assemble_report("Q?", [SQ], _verified(), Corpus.load_default())
    claims = parse_report_claims(report)
    assert claims == [
        ("A typical installation costs $18,000 to $25,000.", "solar-install-cost", False)
    ]


def test_parse_report_claims_flags_unverified():
    report = assemble_report(
        "Q?", [SQ], _verified(), Corpus.load_default(), keep_unverified=True
    )
    claims = parse_report_claims(report)
    assert len(claims) == 2
    assert claims[1][2] is True

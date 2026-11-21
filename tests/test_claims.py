from agent_research_pipeline.claims import (
    Claim,
    extract_claims,
    parse_claim_lines,
    verify_claim,
    verify_claims,
)
from agent_research_pipeline.corpus import Corpus
from agent_research_pipeline.retrieval import TfidfIndex


def _corpus() -> Corpus:
    return Corpus.load_default()


def test_verbatim_claim_is_supported():
    corpus = _corpus()
    doc = corpus.get("solar-install-cost")
    claim = Claim(text=doc.text.split(". ")[0] + ".", source_id=doc.id)
    result = verify_claim(claim, corpus)
    assert result.supported
    assert result.score > 0.9


def test_claim_citing_wrong_source_is_unsupported():
    corpus = _corpus()
    claim = Claim(
        text="A typical residential rooftop solar installation costs between "
        "$18,000 and $25,000 before incentives.",
        source_id="sourdough-basics",
    )
    result = verify_claim(claim, corpus)
    assert not result.supported


def test_unknown_source_is_unsupported():
    result = verify_claim(Claim(text="Anything at all.", source_id="nope"), _corpus())
    assert not result.supported
    assert "does not exist" in result.reason


def test_invented_number_is_unsupported():
    corpus = _corpus()
    claim = Claim(
        text="A typical residential rooftop solar installation costs $99,999 "
        "before incentives for a 7 kilowatt system.",
        source_id="solar-install-cost",
    )
    result = verify_claim(claim, corpus)
    assert not result.supported
    assert "99999" in result.reason


def test_paraphrase_with_same_facts_is_supported():
    corpus = _corpus()
    claim = Claim(
        text="Homeowners usually break even on rooftop solar within 6 to 10 years.",
        source_id="solar-payback",
    )
    assert verify_claim(claim, corpus).supported


def test_extract_claims_without_sabotage_all_supported():
    corpus = _corpus()
    index = TfidfIndex(corpus)
    evidence = {
        "What is the cost of rooftop solar panels?": index.search(
            "What is the cost of rooftop solar panels?", top_k=3
        )
    }
    claims = extract_claims(evidence, corpus=corpus, sabotage=False)
    assert claims
    assert all(v.supported for v in verify_claims(claims, corpus))


def test_sabotage_produces_exactly_two_unsupported():
    corpus = _corpus()
    index = TfidfIndex(corpus)
    evidence = {
        sq: index.search(sq, top_k=3)
        for sq in (
            "What is the cost of rooftop solar panels?",
            "What is the payback period and savings from rooftop solar panels?",
            "What maintenance does rooftop solar panels require?",
        )
    }
    claims = extract_claims(evidence, corpus=corpus, sabotage=True)
    verified = verify_claims(claims, corpus)
    dropped = [v for v in verified if not v.supported]
    assert len(dropped) == 2


def test_parse_claim_lines_filters_invented_citations():
    text = (
        "Solar costs $18,000 to $25,000 before incentives. [solar-install-cost]\n"
        "This claim cites a made-up source. [totally-fake-id]\n"
        "A line with no citation at all\n"
        "- Payback takes 6 to 10 years for most homes. [solar-payback]\n"
    )
    claims = parse_claim_lines(text, {"solar-install-cost", "solar-payback"})
    assert [c.source_id for c in claims] == ["solar-install-cost", "solar-payback"]

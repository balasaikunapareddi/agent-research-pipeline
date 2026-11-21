from agent_research_pipeline.decomposition import (
    decompose,
    extract_topic,
    parse_live_decomposition,
    question_type,
)

SOLAR_Q = "Should a typical homeowner install rooftop solar panels?"
EV_Q = "What should a first-time buyer know before purchasing an electric vehicle?"
REMOTE_Q = "How does remote work affect employee productivity and wellbeing?"


def test_question_types():
    assert question_type(SOLAR_Q) == "decision"
    assert question_type(EV_Q) == "informational"
    assert question_type(REMOTE_Q) == "effect"
    assert question_type("Is solar worth it for a small home?") == "decision"


def test_extract_topic():
    assert extract_topic(SOLAR_Q) == "rooftop solar panels"
    assert extract_topic(EV_Q) == "electric vehicle"
    assert extract_topic(REMOTE_Q) == "remote work"


def test_decision_facets_cover_eval_keywords():
    joined = " ".join(decompose(SOLAR_Q)).lower()
    for facet in ("cost", "payback", "incentive", "maintenance", "environmental"):
        assert facet in joined


def test_vehicle_facets_cover_eval_keywords():
    joined = " ".join(decompose(EV_Q)).lower()
    for facet in ("cost", "charging", "range", "battery", "maintenance"):
        assert facet in joined


def test_effect_facets_cover_eval_keywords():
    joined = " ".join(decompose(REMOTE_Q)).lower()
    for facet in ("productivity", "wellbeing", "collaboration"):
        assert facet in joined


def test_decompose_is_deterministic_and_capped():
    first = decompose(SOLAR_Q)
    assert first == decompose(SOLAR_Q)
    assert 3 <= len(first) <= 6
    assert len(set(first)) == len(first)


def test_parse_live_json_array():
    text = 'Here you go: ["What does it cost?", "What are the risks?"] done'
    assert parse_live_decomposition(text, SOLAR_Q) == [
        "What does it cost?",
        "What are the risks?",
    ]


def test_parse_live_lines():
    text = "1. What does it cost?\n2. What are the risks?\n"
    parsed = parse_live_decomposition(text, SOLAR_Q)
    assert len(parsed) == 2
    assert all(p.endswith("?") for p in parsed)


def test_parse_live_falls_back_to_rules():
    assert parse_live_decomposition("no structure here", SOLAR_Q) == decompose(SOLAR_Q)
    assert parse_live_decomposition("[broken json", SOLAR_Q) == decompose(SOLAR_Q)

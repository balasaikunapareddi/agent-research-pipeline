"""Sub-question decomposition: rules in mock mode, parsing in live mode.

Mock mode decomposes deterministically: detect the question type
(decision / effect / informational), extract the topic noun phrase, and
expand a facet template for that type. Live mode asks the planner LLM for
a JSON list and parses it here — parsing is code, so a malformed reply
falls back to the rules instead of crashing the run.
"""

from __future__ import annotations

import json
import re

MAX_SUB_QUESTIONS = 6

_DECISION_PHRASES = ("worth it", "good idea", "is it better", "do i need")
_ACTION_VERBS = ("install", "buy", "purchase", "adopt", "get", "switch to", "start")
_GERUNDS = ("purchasing", "buying", "getting", "choosing", "installing")


def question_type(question: str) -> str:
    lowered = question.lower().strip()
    if "affect" in lowered or "impact of" in lowered or "effect of" in lowered:
        return "effect"
    # "Should I/we/one ...?" is a decision question; "What should a buyer
    # know ...?" is informational — position matters.
    if lowered.startswith("should") or any(p in lowered for p in _DECISION_PHRASES):
        return "decision"
    return "informational"


def _strip_article(phrase: str) -> str:
    phrase = phrase.strip().strip("?. ")
    for article in ("an ", "a ", "the "):
        if phrase.startswith(article):
            return phrase[len(article):]
    return phrase


def extract_topic(question: str) -> str:
    lowered = question.lower().strip()
    m = re.search(r"how does (.+?) affect ", lowered)
    if m:
        return _strip_article(m.group(1))
    for verb in _ACTION_VERBS + _GERUNDS:
        m = re.search(rf"\b{verb}\s+(.+?)\??$", lowered)
        if m:
            return _strip_article(m.group(1))
    # Fallback: drop leading question words, keep the rest as the topic.
    cleaned = re.sub(
        r"^(what|how|why|when|where|which|who|should|is|are|do|does)\b\s*",
        "",
        lowered,
    )
    return _strip_article(cleaned) or lowered


def _effect_objects(question: str) -> list[str]:
    m = re.search(r"affect (.+?)\??$", question.lower().strip())
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r"\s+and\s+|,\s*", raw)
    return [p.strip() for p in parts if p.strip()]


def _decision_facets(topic: str) -> list[str]:
    return [
        f"What is the cost of {topic}?",
        f"What is the payback period and savings from {topic}?",
        f"What incentives or tax credits exist for {topic}?",
        f"What maintenance does {topic} require?",
        f"What is the environmental impact of {topic}?",
        f"What are the requirements and risks of {topic}?",
    ]


def _vehicle_facets(topic: str) -> list[str]:
    return [
        f"What is the purchase cost of {topic}?",
        f"How is {topic} charged and what does charging cost?",
        f"What range does {topic} offer?",
        f"How long does the battery in {topic} last?",
        f"What maintenance does {topic} need?",
        f"What is the environmental impact of {topic}?",
    ]


def _informational_facets(topic: str) -> list[str]:
    lowered = topic.lower()
    if any(k in lowered for k in ("electric vehicle", " ev", "car", "vehicle")):
        return _vehicle_facets(topic)
    if "solar" in lowered:
        return _decision_facets(topic)
    return [
        f"What is the cost of {topic}?",
        f"What are the key features of {topic}?",
        f"What are the requirements for {topic}?",
        f"What are the long-term costs of {topic}?",
        f"What are common pitfalls with {topic}?",
        f"What are the alternatives to {topic}?",
    ]


def _effect_facets(question: str, topic: str) -> list[str]:
    objects = _effect_objects(question)
    facets = [f"How does {topic} affect {obj}?" for obj in objects]
    facets.append(
        f"What other effects does {topic} have, including on collaboration "
        f"and communication?"
    )
    facets.append(f"What does the evidence say about {topic} overall?")
    return facets


def decompose(question: str) -> list[str]:
    """Deterministic rule-based decomposition (mock mode)."""
    qtype = question_type(question)
    topic = extract_topic(question)
    if qtype == "effect":
        facets = _effect_facets(question, topic)
    elif qtype == "decision":
        facets = _decision_facets(topic)
    else:
        facets = _informational_facets(topic)
    # Dedupe, preserve order, cap the fan-out.
    seen: set[str] = set()
    unique = []
    for f in facets:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique[:MAX_SUB_QUESTIONS]


def parse_live_decomposition(text: str, question: str) -> list[str]:
    """Parse the planner LLM's reply into sub-questions.

    Accepts a JSON array anywhere in the text, or one sub-question per
    line. Anything unusable falls back to the deterministic rules.
    """
    candidates: list[str] = []
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                candidates = [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            candidates = []
    if not candidates:
        lines = [ln.strip(" -*0123456789.") for ln in text.splitlines()]
        candidates = [ln for ln in lines if ln.endswith("?") and len(ln) > 10]
    candidates = candidates[:MAX_SUB_QUESTIONS]
    if len(candidates) < 2:
        return decompose(question)
    return candidates

"""Multi-agent research pipeline with claim-level fact-checking.

Mock-first: the full pipeline — decomposition, retrieval over a bundled
corpus, claim extraction, citation verification, report assembly — runs
offline with no API key. Live mode (GEMINI_API_KEY) swaps the planner and
claim-drafting steps to real Gemini calls; every checkable decision
(retrieval ranking, citation verification, the fact-check gate) stays in
deterministic code either way.
"""

__version__ = "0.1.0"

"""Deterministic TF-IDF retrieval over the bundled corpus.

Pure-Python TF-IDF with cosine similarity — no embeddings service, no
network, same ranking on every run. Ties break by document id so results
are stable. Snippets are the single best-matching sentence of each hit,
which is what claims are later extracted from.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Corpus, SourceDocument

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or",
    "should", "than", "that", "the", "their", "they", "this", "to",
    "typical", "what", "when", "which", "who", "will", "with", "you",
    "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def content_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


@dataclass
class SearchResult:
    doc_id: str
    title: str
    snippet: str
    score: float

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "score": round(self.score, 4),
        }


class TfidfIndex:
    """TF-IDF index over corpus documents (title + text)."""

    def __init__(self, corpus: Corpus):
        self._corpus = corpus
        docs = corpus.documents()
        self._doc_tokens: dict[str, Counter] = {}
        df: Counter = Counter()
        for doc in docs:
            # Title terms count triple: titles are the densest topic signal.
            counts = Counter(tokenize(doc.text))
            counts.update(tokenize(doc.title) * 3)
            self._doc_tokens[doc.id] = counts
            for term in counts:
                df[term] += 1
        n = max(len(docs), 1)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._doc_vectors = {
            doc_id: self._vectorize(counts)
            for doc_id, counts in self._doc_tokens.items()
        }

    def _vectorize(self, counts: Counter) -> dict[str, float]:
        vec = {t: c * self._idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        q_counts = Counter(tokenize(query))
        if not q_counts:
            return []
        q_vec = self._vectorize(q_counts)
        scored: list[tuple[float, str]] = []
        for doc_id, d_vec in self._doc_vectors.items():
            score = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
            if score > 0:
                scored.append((score, doc_id))
        # Highest score first; document id breaks ties deterministically.
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, doc_id in scored[:top_k]:
            doc = self._corpus.get(doc_id)
            assert doc is not None
            results.append(
                SearchResult(
                    doc_id=doc.id,
                    title=doc.title,
                    snippet=best_sentence(query, doc),
                    score=score,
                )
            )
        return results


def ranked_sentences(query: str, doc: SourceDocument, n: int = 2) -> list[str]:
    """The ``n`` sentences of ``doc`` with the most query-term overlap."""
    q_tokens = content_tokens(query)
    sentences = split_sentences(doc.text)
    if not sentences:
        return [doc.text]
    ranked = sorted(
        sentences,
        key=lambda s: (-len(q_tokens & content_tokens(s)), len(s)),
    )
    return ranked[:n]


def best_sentence(query: str, doc: SourceDocument) -> str:
    """The sentence of ``doc`` with the most query-term overlap."""
    return ranked_sentences(query, doc, n=1)[0]

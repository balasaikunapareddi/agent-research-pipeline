"""Bundled source corpus: synthetic documents with stable ids.

Mock mode retrieves from this local corpus instead of the web, so the
whole pipeline is deterministic and offline. Each document is a small,
self-contained set of factual sentences — claims cite a document id, and
the fact-checker verifies the claim against that document's text.

The corpus is loaded once and treated as read-only, which makes it safe
to share across the parallel researchers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_CORPUS_PATH = _DATA_DIR / "corpus.json"
_EVALS_PATH = _DATA_DIR / "evals.json"


@dataclass(frozen=True)
class SourceDocument:
    id: str
    title: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text}


class Corpus:
    """An immutable, id-indexed set of source documents."""

    def __init__(self, documents: list[SourceDocument]):
        self._by_id = {d.id: d for d in documents}
        if len(self._by_id) != len(documents):
            raise ValueError("duplicate document ids in corpus")
        self._documents = list(documents)

    @classmethod
    def load_default(cls) -> "Corpus":
        raw = json.loads(_CORPUS_PATH.read_text())
        return cls([SourceDocument(**d) for d in raw["documents"]])

    def get(self, doc_id: str) -> SourceDocument | None:
        return self._by_id.get(doc_id)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._by_id

    def __len__(self) -> int:
        return len(self._documents)

    def documents(self) -> list[SourceDocument]:
        return list(self._documents)

    def ids(self) -> list[str]:
        return [d.id for d in self._documents]


def load_eval_cases() -> list[dict]:
    """Bundled eval cases: question, expected facets, must-cite sources."""
    return json.loads(_EVALS_PATH.read_text())["cases"]

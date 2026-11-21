from agent_research_pipeline.corpus import Corpus
from agent_research_pipeline.retrieval import TfidfIndex, best_sentence, ranked_sentences


def _index() -> TfidfIndex:
    return TfidfIndex(Corpus.load_default())


def test_corpus_loads_with_unique_ids():
    corpus = Corpus.load_default()
    assert len(corpus) >= 20
    assert len(set(corpus.ids())) == len(corpus)
    assert "solar-install-cost" in corpus
    assert corpus.get("no-such-doc") is None


def test_search_finds_relevant_doc():
    results = _index().search("What is the cost of rooftop solar panels?", top_k=3)
    ids = [r.doc_id for r in results]
    assert "solar-install-cost" in ids
    assert results[0].score >= results[-1].score > 0


def test_search_is_deterministic():
    index = _index()
    a = index.search("electric vehicle charging at home", top_k=3)
    b = index.search("electric vehicle charging at home", top_k=3)
    assert [(r.doc_id, r.snippet, r.score) for r in a] == [
        (r.doc_id, r.snippet, r.score) for r in b
    ]


def test_search_empty_query():
    assert _index().search("", top_k=3) == []
    assert _index().search("the and of", top_k=3) == []


def test_search_respects_top_k():
    assert len(_index().search("solar", top_k=2)) <= 2


def test_best_sentence_picks_on_topic_sentence():
    corpus = Corpus.load_default()
    doc = corpus.get("ev-charging")
    sentence = best_sentence("Level 2 charger installed cost", doc)
    assert "charger" in sentence.lower()
    assert "$500" in sentence
    assert sentence in doc.text


def test_ranked_sentences_returns_distinct():
    corpus = Corpus.load_default()
    doc = corpus.get("solar-payback")
    sentences = ranked_sentences("solar savings per year", doc, n=2)
    assert len(sentences) == 2
    assert len(set(sentences)) == 2

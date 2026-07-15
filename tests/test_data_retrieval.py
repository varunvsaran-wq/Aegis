"""Tests for the data (hotpotqa, ingest) and retrieval modules.

No network access and no model downloads: the fake embedder/reranker are
forced via environment variables, and HotpotQA loading is tested only through
the pure record-mapping function ``_to_question``.
"""

import numpy as np
import pytest

from aegis.config import AegisConfig, get_config
from aegis.eval.benchmarks.hotpotqa import _to_question, build_corpus, sample_hash
from aegis.ingest import BuiltIndex, FakeEmbedder, build_or_load_index, corpus_hash
from aegis.retrieve import FakeReranker, HybridRetriever, build_retriever
from aegis.types import HotpotQuestion


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch):
    """Force fake embedder/reranker and memory backend; reset config cache."""
    monkeypatch.setenv("AEGIS_EMBEDDER", "fake")
    monkeypatch.setenv("AEGIS_RERANKER", "fake")
    monkeypatch.setenv("AEGIS_RETRIEVAL_BACKEND", "memory")
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def _q(qid, question, answer, context):
    return HotpotQuestion(
        id=qid,
        question=question,
        answer=answer,
        type="bridge",
        level="easy",
        supporting_facts=[(context[0][0], 0)],
        context=context,
    )


@pytest.fixture
def questions():
    acme = (
        "Acme Corporation",
        ["Acme Corporation was founded by John Smith in 1947.", "It manufactures anvils."],
    )
    zebra = (
        "Zebra Habitats",
        ["Zebras live in savannas across eastern Africa.", "They eat grass."],
    )
    widget = (
        "Quantum Widgets",
        ["Quantum Widgets pioneered entangled sprockets.", "The firm is based in Geneva."],
    )
    kili = (
        "Mount Kilimanjaro",
        ["Mount Kilimanjaro is the tallest mountain in Africa.", "It is a dormant volcano."],
    )
    orchid = (
        "Ghost Orchid",
        ["The ghost orchid is a rare flower found in Florida swamps.", "It has no leaves."],
    )
    return [
        _q("q3", "Who founded Acme Corporation?", "John Smith", [acme, zebra]),
        _q("q1", "Where are Quantum Widgets based?", "Geneva", [widget, acme]),
        _q("q2", "What is the tallest mountain in Africa?", "Mount Kilimanjaro", [kili, orchid]),
    ]


def _fake_config(tmp_path):
    return AegisConfig(
        data_dir=tmp_path, embedder="fake", reranker="fake", retrieval_backend="memory"
    )


# ---------------------------------------------------------------------------
# hotpotqa.py
# ---------------------------------------------------------------------------


def test_to_question_maps_hf_schema():
    record = {
        "id": "abc123",
        "question": "Who wrote the book?",
        "answer": "Jane Doe",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {"title": ["Book", "Author"], "sent_id": [0, 2]},
        "context": {
            "title": ["Book", "Author"],
            "sentences": [["The book is old.", "It is red."], ["Jane wrote it."]],
        },
    }
    q = _to_question(record)
    assert isinstance(q, HotpotQuestion)
    assert q.id == "abc123"
    assert q.type == "comparison"
    assert q.level == "hard"
    assert q.supporting_facts == [("Book", 0), ("Author", 2)]
    assert q.context == [
        ("Book", ["The book is old.", "It is red."]),
        ("Author", ["Jane wrote it."]),
    ]


def test_build_corpus_dedupes_and_formats_ids(questions):
    chunks = build_corpus(questions)
    titles = [c.title for c in chunks]
    # "Acme Corporation" appears under two questions but only once in corpus.
    assert len(titles) == len(set(titles)) == 5
    assert "Acme Corporation" in titles
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"{chunk.title}::{i}"
        assert chunk.text == " ".join(chunk.sentences)
        assert chunk.metadata == {"poisoned": False}


def test_build_corpus_deterministic_and_order_independent(questions):
    a = build_corpus(questions)
    b = build_corpus(questions)
    c = build_corpus(list(reversed(questions)))
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert [x.model_dump() for x in a] == [x.model_dump() for x in c]


def test_sample_hash_stable_and_order_independent(questions):
    h1 = sample_hash(questions)
    h2 = sample_hash(list(reversed(questions)))
    assert h1 == h2
    assert len(h1) == 12
    assert h1 != sample_hash(questions[:2])


# ---------------------------------------------------------------------------
# ingest.py
# ---------------------------------------------------------------------------


def test_fake_embedder_shape_norm_determinism():
    embedder = FakeEmbedder()
    texts = ["hello world", "quantum sprockets in geneva", ""]
    a = embedder.encode(texts)
    b = embedder.encode(texts)
    assert a.shape == (3, 64)
    np.testing.assert_allclose(a, b)
    norms = np.linalg.norm(a, axis=1)
    np.testing.assert_allclose(norms[:2], 1.0, rtol=1e-6)
    assert norms[2] == 0.0  # zero-safe: empty text stays a zero vector


def test_built_index_save_load_roundtrip(questions, tmp_path):
    chunks = build_corpus(questions)
    config = _fake_config(tmp_path)
    index_dir = tmp_path / "index" / "roundtrip"

    built = build_or_load_index(chunks, config, index_dir=index_dir)
    assert (index_dir / "chunks.jsonl").exists()
    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "meta.json").exists()

    loaded = build_or_load_index(chunks, config, index_dir=index_dir)
    assert isinstance(loaded, BuiltIndex)
    assert [c.model_dump() for c in loaded.chunks] == [c.model_dump() for c in built.chunks]
    np.testing.assert_allclose(loaded.embeddings, built.embeddings)
    assert loaded.embedder_name == "fake"
    assert loaded.corpus_hash == corpus_hash(chunks)
    # BM25 was rebuilt at load time and is queryable.
    scores = loaded.bm25.get_scores(["kilimanjaro"])
    assert max(scores) > 0


# ---------------------------------------------------------------------------
# retrieve.py
# ---------------------------------------------------------------------------


def test_fake_reranker_jaccard():
    reranker = FakeReranker()
    scores = reranker.score("acme corporation", ["acme corporation", "zebras eat grass", ""])
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0


def test_hybrid_retriever_end_to_end(questions, tmp_path):
    chunks = build_corpus(questions)
    config = _fake_config(tmp_path)
    retriever = build_retriever(chunks, config)
    assert isinstance(retriever, HybridRetriever)
    assert retriever.backend_name == "memory"

    results = retriever.retrieve("Who founded Acme Corporation?", k_final=3)
    assert len(results) == 3
    assert results[0].chunk.title == "Acme Corporation"
    # Sorted by rerank score, descending.
    rerank_scores = [r.rerank_score for r in results]
    assert rerank_scores == sorted(rerank_scores, reverse=True)
    top = results[0]
    assert top.dense_score is not None
    assert top.sparse_score is not None
    assert top.fused_score is not None and top.fused_score > 0
    assert top.rerank_score is not None and top.rerank_score > 0


def test_rrf_keeps_bm25_only_candidates(questions, tmp_path, monkeypatch):
    """A chunk ranked top by BM25 but absent from dense results still surfaces."""
    chunks = build_corpus(questions)
    config = _fake_config(tmp_path)
    retriever = build_retriever(chunks, config)

    target_idx = next(
        i for i, c in enumerate(retriever.index.chunks) if c.title == "Mount Kilimanjaro"
    )
    # Dense search "misses" the target chunk entirely.
    others = [i for i in range(len(chunks)) if i != target_idx]
    monkeypatch.setattr(
        retriever,
        "_dense_search",
        lambda query_vector, k: [(i, 0.9) for i in others[:k]],
    )

    results = retriever.retrieve("Kilimanjaro tallest mountain", k_final=len(chunks))
    ids = [r.chunk.id for r in results]
    assert retriever.index.chunks[target_idx].id in ids
    target_result = next(r for r in results if r.chunk.title == "Mount Kilimanjaro")
    assert target_result.dense_score is None
    assert target_result.sparse_score is not None and target_result.sparse_score > 0
    assert target_result.fused_score is not None and target_result.fused_score > 0

"""Hybrid retrieval: dense + BM25 with Reciprocal Rank Fusion and reranking.

Heavy dependencies (sentence-transformers, qdrant-client) are imported lazily
so this module imports cleanly before optional deps finish installing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aegis.config import AegisConfig, get_config
from aegis.ingest import (
    BuiltIndex,
    _tokenize,
    build_or_load_index,
    get_embedder,
)
from aegis.types import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

#: Reciprocal Rank Fusion constant (standard value from Cormack et al.).
RRF_K = 60


class FakeReranker:
    """Deterministic Jaccard token-overlap reranker for tests."""

    name = "fake"

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(_tokenize(query))
        scores: list[float] = []
        for text in texts:
            text_tokens = set(_tokenize(text))
            union = query_tokens | text_tokens
            if not union:
                scores.append(0.0)
            else:
                scores.append(len(query_tokens & text_tokens) / len(union))
        return scores


class CrossEncoderReranker:
    """Cross-encoder reranker, lazily loaded on first score call."""

    def __init__(self, model_name: str, data_dir: Path):
        self.name = model_name
        self._data_dir = data_dir
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy

            cache_folder = str(self._data_dir / "models_cache")
            try:
                self._model = CrossEncoder(self.name, cache_folder=cache_folder)
            except TypeError:
                # Older sentence-transformers versions lack cache_folder.
                self._model = CrossEncoder(self.name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        model = self._get_model()
        preds = model.predict([(query, text) for text in texts])
        return [float(p) for p in np.asarray(preds).reshape(-1)]


def get_reranker(config: AegisConfig | None = None) -> FakeReranker | CrossEncoderReranker:
    """Return the reranker selected by ``config.reranker``."""
    config = config or get_config()
    if config.reranker == "fake":
        return FakeReranker()
    return CrossEncoderReranker(config.reranker, config.data_dir)


class QdrantBackend:
    """Dense vector search backed by a Qdrant server (BM25 stays in-process)."""

    def __init__(self, url: str):
        from qdrant_client import QdrantClient  # lazy

        self.client = QdrantClient(url=url)
        self.collection_name: str | None = None

    def health_check(self) -> None:
        """Raise if the server is unreachable."""
        self.client.get_collections()

    def ensure_collection(self, index: BuiltIndex) -> None:
        """Create (if needed) and populate the collection for this corpus."""
        from qdrant_client import models  # lazy

        name = f"aegis_{index.corpus_hash}"
        self.collection_name = name
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=int(index.embeddings.shape[1]),
                    distance=models.Distance.COSINE,
                ),
            )
            points = [
                models.PointStruct(
                    id=i,
                    vector=index.embeddings[i].tolist(),
                    payload={"chunk_id": chunk.id, "title": chunk.title},
                )
                for i, chunk in enumerate(index.chunks)
            ]
            self.client.upsert(collection_name=name, points=points)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(corpus_index, dense_score), ...] for the top-k points."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=k,
            with_payload=False,
        )
        return [(int(point.id), float(point.score)) for point in response.points]


def _rrf(rank: int) -> float:
    """Reciprocal Rank Fusion contribution for a 1-based rank."""
    return 1.0 / (RRF_K + rank)


class HybridRetriever:
    """Dense + BM25 retrieval fused with RRF, then cross-encoder reranked."""

    def __init__(
        self,
        index: BuiltIndex,
        embedder,
        reranker,
        backend: str = "memory",
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.backend_name = "memory"
        self.qdrant: QdrantBackend | None = None

        if backend in ("qdrant", "auto"):
            try:
                qdrant = QdrantBackend(get_config().qdrant_url)
                qdrant.health_check()
                qdrant.ensure_collection(index)
                self.qdrant = qdrant
                self.backend_name = "qdrant"
            except Exception as exc:
                if backend == "qdrant":
                    raise
                logger.warning(
                    "Qdrant unavailable (%s); falling back to memory backend.", exc
                )

    def _dense_search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.qdrant is not None:
            return self.qdrant.search(query_vector, k)
        # Embeddings are L2-normalized, so dot product == cosine similarity.
        scores = self.index.embeddings @ query_vector
        k = min(k, len(scores))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = np.asarray(self.index.bm25.get_scores(_tokenize(query)))
        k = min(k, len(scores))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def retrieve(
        self,
        query: str,
        k_dense: int = 20,
        k_sparse: int = 20,
        k_final: int = 5,
    ) -> list[RetrievalResult]:
        query_vector = self.embedder.encode([query])[0]
        dense = self._dense_search(query_vector, k_dense)
        sparse = self._sparse_search(query, k_sparse)

        dense_scores = {i: s for i, s in dense}
        sparse_scores = {i: s for i, s in sparse}

        fused: dict[int, float] = {}
        for rank, (i, _score) in enumerate(dense, start=1):
            fused[i] = fused.get(i, 0.0) + _rrf(rank)
        for rank, (i, _score) in enumerate(sparse, start=1):
            fused[i] = fused.get(i, 0.0) + _rrf(rank)

        # A cross-encoder earns its keep only when it re-scores a large
        # candidate pool; a pool of ~20 starves it. 50-100 is the sweet spot
        # (see Cross-encoder reranking literature) at negligible extra cost.
        n_candidates = max(k_final * 10, 50)
        candidates = sorted(fused, key=lambda i: fused[i], reverse=True)[:n_candidates]
        if not candidates:
            return []

        if self.reranker is None:
            # -reranker ablation: keep the RRF-fused order, no cross-encoder.
            order = [(i, fused[i]) for i in candidates[:k_final]]
        else:
            rerank_scores = self.reranker.score(
                query, [self.index.chunks[i].text for i in candidates]
            )
            order = sorted(
                zip(candidates, rerank_scores), key=lambda pair: pair[1], reverse=True
            )[:k_final]
        return [
            RetrievalResult(
                chunk=self.index.chunks[i],
                dense_score=dense_scores.get(i),
                sparse_score=sparse_scores.get(i),
                fused_score=fused[i],
                rerank_score=score,
            )
            for i, score in order
        ]


def build_retriever(
    chunks: list[Chunk], config: AegisConfig | None = None, rerank: bool = True
) -> HybridRetriever:
    """Build index, embedder, and reranker from config and wire them together.

    ``rerank=False`` builds a retriever with no cross-encoder (the
    ``-reranker`` ablation): retrieval stops at the RRF-fused order.
    """
    config = config or get_config()
    index = build_or_load_index(chunks, config)
    return HybridRetriever(
        index=index,
        embedder=get_embedder(config),
        reranker=get_reranker(config) if rerank else None,
        backend=config.retrieval_backend,
    )

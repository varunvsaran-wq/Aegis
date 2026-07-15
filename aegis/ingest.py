"""Corpus ingestion: embedders and the persisted dense+sparse index.

Heavy dependencies (sentence-transformers, rank_bm25) are imported lazily so
this module imports cleanly before optional deps finish installing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from aegis.config import AegisConfig, get_config
from aegis.types import Chunk

_EMBED_DIM = 64

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer shared by BM25 and FakeEmbedder."""
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """Deterministic hash-bucket bag-of-words embedder (dim 64, no ML deps)."""

    name = "fake"
    dim = _EMBED_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                digest = hashlib.md5(word.encode("utf-8")).hexdigest()
                bucket = int(digest, 16) % _EMBED_DIM
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # zero-safe
        return out / norms


class STEmbedder:
    """Sentence-transformers embedder, lazily loaded on first encode."""

    def __init__(self, model_name: str, data_dir: Path):
        self.name = model_name
        self._data_dir = data_dir
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(
                self.name, cache_folder=str(self._data_dir / "models_cache")
            )
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return np.asarray(
            model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            )
        )


def get_embedder(config: AegisConfig | None = None) -> FakeEmbedder | STEmbedder:
    """Return the embedder selected by ``config.embedder``."""
    config = config or get_config()
    if config.embedder == "fake":
        return FakeEmbedder()
    return STEmbedder(config.embedder, config.data_dir)


def corpus_hash(chunks: list[Chunk]) -> str:
    """12-char sha256 fingerprint over chunk ids and texts."""
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.id.encode("utf-8"))
        h.update(b"\x00")
        h.update(chunk.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:12]


class BuiltIndex:
    """Chunks + dense embedding matrix + in-process BM25 index."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedder_name: str = "unknown",
    ):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder_name = embedder_name
        self.corpus_hash = corpus_hash(chunks)
        self.bm25 = self._build_bm25(chunks)

    @staticmethod
    def _build_bm25(chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi  # lazy

        return BM25Okapi([_tokenize(chunk.text) for chunk in chunks])

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        np.save(index_dir / "embeddings.npy", self.embeddings)
        meta = {"embedder": self.embedder_name, "corpus_hash": self.corpus_hash}
        (index_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "BuiltIndex":
        index_dir = Path(index_dir)
        chunks = []
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk.model_validate(json.loads(line)))
        embeddings = np.load(index_dir / "embeddings.npy")
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, embeddings, embedder_name=meta.get("embedder", "unknown"))


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_or_load_index(
    chunks: list[Chunk],
    config: AegisConfig | None = None,
    index_dir: Path | None = None,
) -> BuiltIndex:
    """Load a persisted index for this corpus+embedder, or build and save it."""
    config = config or get_config()
    embedder = get_embedder(config)
    if index_dir is None:
        index_dir = (
            config.data_dir
            / "index"
            / f"{corpus_hash(chunks)}-{_sanitize(embedder.name)}"
        )
    index_dir = Path(index_dir)
    if (index_dir / "meta.json").exists():
        return BuiltIndex.load(index_dir)
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    index = BuiltIndex(chunks, embeddings, embedder_name=embedder.name)
    index.save(index_dir)
    return index

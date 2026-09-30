"""Aegis configuration: environment-driven settings and the model registry."""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

#: Maps short model aliases to litellm model strings. Users can override or
#: extend these via an optional ``aegis.yaml`` in the working directory with
#: a ``models:`` mapping.
MODEL_REGISTRY: dict[str, str] = {
    "mock": "mock/echo",
    "local": "ollama/llama3.1:8b",
    "small": "anthropic/claude-haiku-4-5",
    "mid": "openai/gpt-4o",
    "frontier": "anthropic/claude-opus-4-8",
    # Hosted weak model (OpenRouter), for testing the harness where the base
    # model makes more mistakes. Needs OPENROUTER_API_KEY.
    "weak": "openrouter/meta-llama/llama-3.1-8b-instruct",
}

_REGISTRY_FILE = Path("aegis.yaml")


def _load_registry_overrides() -> None:
    """Merge model overrides from ./aegis.yaml (if present) into the registry."""
    if not _REGISTRY_FILE.exists():
        return
    try:
        loaded = yaml.safe_load(_REGISTRY_FILE.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return
    models = loaded.get("models") if isinstance(loaded, dict) else None
    if isinstance(models, dict):
        MODEL_REGISTRY.update({str(k): str(v) for k, v in models.items()})


_load_registry_overrides()


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load ``KEY=VALUE`` lines from a local ``.env`` into ``os.environ``.

    Existing environment variables win, so an exported key is never clobbered.
    Blank lines, ``#`` comments, an optional ``export`` prefix, and surrounding
    quotes are handled. Returns the names that were set (never the values).
    """
    env_path = Path(path)
    if not env_path.is_file():
        return []
    loaded: list[str] = []
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def resolve_model(alias_or_model: str) -> str:
    """Resolve a registry alias to a litellm model string.

    If ``alias_or_model`` is a known registry alias, return the mapped model
    string; otherwise return the input unchanged, allowing callers to pass
    raw litellm model strings directly (e.g. ``"openai/gpt-4o-mini"``).
    """
    return MODEL_REGISTRY.get(alias_or_model, alias_or_model)


def _default_embedder() -> str:
    value = os.environ.get("AEGIS_EMBEDDER", "BAAI/bge-base-en-v1.5")
    return "fake" if value == "fake" else value


def _default_reranker() -> str:
    return os.environ.get("AEGIS_RERANKER", "BAAI/bge-reranker-v2-m3")


def _default_nli_model() -> str:
    return os.environ.get("AEGIS_NLI", "cross-encoder/nli-deberta-v3-base")


class AegisConfig(BaseModel):
    """Global configuration. Obtain via :func:`get_config`."""

    data_dir: Path = Path("data")
    report_dir: Path = Path("report")

    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.environ.get("AEGIS_MLFLOW_URI", "sqlite:///mlflow.db")
    )
    qdrant_url: str = Field(
        default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333")
    )

    embedder: str = Field(default_factory=_default_embedder)
    """Sentence-transformers embedding model, or "fake" for tests."""

    reranker: str = Field(default_factory=_default_reranker)
    """Cross-encoder reranker model, or "fake" for tests."""

    nli_model: str = Field(default_factory=_default_nli_model)
    """Cross-encoder NLI verifier model, or "fake" for tests."""

    retrieval_backend: str = Field(
        default_factory=lambda: os.environ.get("AEGIS_RETRIEVAL_BACKEND", "auto")
    )
    """One of "qdrant" | "memory" | "auto" (auto tries qdrant, falls back to memory)."""


@lru_cache(maxsize=1)
def get_config() -> AegisConfig:
    """Return the cached singleton configuration."""
    return AegisConfig()

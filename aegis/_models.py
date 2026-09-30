"""Process-wide cache of loaded ML models.

Embedders, rerankers, and NLI models are hundreds of MB each. Every retriever
or verifier build used to reload them from disk; this cache keeps one instance
per (kind, model name) for the life of the process.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_CACHE: dict[tuple[str, str], object] = {}


def cached_model(kind: str, name: str, factory: Callable[[], T]) -> T:
    """Return the cached model for ``(kind, name)``, loading it once via ``factory``."""
    key = (kind, name)
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]  # type: ignore[return-value]


def to_inference_precision(module) -> None:
    """Cast a loaded model to fp16 when it runs on a CUDA GPU.

    Measured on the three models Aegis uses (bge-base, bge-reranker-v2-m3,
    nli-deberta-v3-base): 3-3.5x faster, identical top-5 reranking, identical
    NLI labels, embedding cosine >= 0.9995 against fp32. CPU stays fp32 (fp16
    is slower there). Set ``AEGIS_FP16=0`` to force fp32.
    """
    import os

    if os.environ.get("AEGIS_FP16", "1") == "0":
        return
    try:
        param = next(module.parameters())
    except (StopIteration, AttributeError):
        return
    if param.device.type == "cuda":
        module.half()

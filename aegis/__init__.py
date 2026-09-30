"""Aegis: a model-agnostic RAG reliability harness."""

__version__ = "0.1.0"


def _preload_sklearn_openmp() -> None:
    """Load scikit-learn's OpenMP runtime before torch can load its own.

    On Windows, if a CUDA build of torch is imported, then pyarrow (via
    ``datasets`` or MLflow), scikit-learn's ``vcomp140.dll`` fails to
    initialize (WinError 1114) and every model load dies on import. Loading
    that DLL first avoids the conflict. It is a no-op elsewhere, and it only
    reads the DLL's location without importing scikit-learn.
    """
    import sys

    if sys.platform != "win32":
        return
    try:
        import ctypes
        import importlib.util
        from pathlib import Path

        spec = importlib.util.find_spec("sklearn")
        if spec is None or spec.origin is None:
            return
        dll = Path(spec.origin).parent / ".libs" / "vcomp140.dll"
        if dll.is_file():
            ctypes.WinDLL(str(dll))
    except Exception:  # never let a best-effort preload break the import
        pass


_preload_sklearn_openmp()

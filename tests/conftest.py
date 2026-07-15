"""Shared pytest fixtures for the Aegis test suite."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _fake_embedder_env():
    """Force the fake embedder for the whole test session.

    Ensures tests never download embedding models. Set before any Aegis
    config is constructed (session-scoped, autouse).
    """
    previous = os.environ.get("AEGIS_EMBEDDER")
    os.environ["AEGIS_EMBEDDER"] = "fake"
    yield
    if previous is None:
        os.environ.pop("AEGIS_EMBEDDER", None)
    else:
        os.environ["AEGIS_EMBEDDER"] = previous


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """A throwaway data directory for tests."""
    return tmp_path

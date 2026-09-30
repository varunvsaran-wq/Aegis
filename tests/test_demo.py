"""Tests for the demo/serving layer and the demo-readiness fixes.

Hermetic: fake embedder (conftest), fake reranker/NLI via env, mock model.
"""

import os

import pytest

from aegis.config import load_dotenv
from aegis.defense import DATA_OPEN
from aegis.gateway import get_client
from aegis.graph import HarnessedPipeline
from aegis.types import Chunk, HotpotQuestion, LLMResponse, RetrievalResult, VerifierReport


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch):
    monkeypatch.setenv("AEGIS_RERANKER", "fake")
    monkeypatch.setenv("AEGIS_NLI", "fake")
    monkeypatch.setenv("AEGIS_RETRIEVAL_BACKEND", "memory")
    from aegis.config import get_config

    get_config.cache_clear()
    yield
    get_config.cache_clear()


# ------------------------------------------------------------------ .env


class TestLoadDotenv:
    def test_loads_new_keys_without_overriding(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text(
            "# comment\n"
            "AEGIS_TEST_KEY_A=alpha\n"
            'export AEGIS_TEST_KEY_B="beta"\n'
            "AEGIS_TEST_KEY_C=from-file\n"
            "\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("AEGIS_TEST_KEY_A", raising=False)
        monkeypatch.delenv("AEGIS_TEST_KEY_B", raising=False)
        monkeypatch.setenv("AEGIS_TEST_KEY_C", "from-shell")

        loaded = load_dotenv(env)

        assert os.environ["AEGIS_TEST_KEY_A"] == "alpha"
        assert os.environ["AEGIS_TEST_KEY_B"] == "beta"
        assert os.environ["AEGIS_TEST_KEY_C"] == "from-shell"  # shell wins
        assert sorted(loaded) == ["AEGIS_TEST_KEY_A", "AEGIS_TEST_KEY_B"]
        monkeypatch.delenv("AEGIS_TEST_KEY_A")
        monkeypatch.delenv("AEGIS_TEST_KEY_B")

    def test_missing_file_is_noop(self, tmp_path):
        assert load_dotenv(tmp_path / "nope.env") == []


# ---------------------------------------------------- harness output fixes


def _question() -> HotpotQuestion:
    return HotpotQuestion(
        id="q1",
        question="What is the capital of France?",
        answer="Paris",
        type="bridge",
        level="easy",
        supporting_facts=[("Paris", 0)],
        context=[("Paris", ["Paris is the capital of France."])],
    )


class _Retriever:
    def retrieve(self, query, k_final=5, **kw):
        return [
            RetrievalResult(
                chunk=Chunk(
                    id="Paris::0",
                    title="Paris",
                    text="Paris is the capital of France.",
                    sentences=["Paris is the capital of France."],
                ),
                rerank_score=0.9,
            )
        ]


def test_spotlight_markers_never_reach_the_answer():
    pipe = HarnessedPipeline(
        get_client("mock"),
        _Retriever(),
        use_structurer=False,
        use_verifier=False,
        use_defense=True,
    )
    res = pipe.run(_question())
    assert "capital of France" in res.answer.text
    assert DATA_OPEN not in res.answer.text


class _LeakOnRetryClient:
    """First answer is clean; the verify-retry answer leaks the canary."""

    def __init__(self, canary):
        self.canary = canary
        self.model = "mock"
        self.calls = 0

    def complete(self, messages, max_tokens=1024):
        self.calls += 1
        text = (
            "ANSWER: Paris\nCITATIONS: Paris::0"
            if self.calls == 1
            else f"ANSWER: token {self.canary}\nCITATIONS: Paris::0"
        )
        return LLMResponse(text=text, model="mock", model_version="mock/echo")


class _AlwaysUngrounded:
    def verify(self, answer, retrieved, question):
        return False, [
            VerifierReport(claim="c", chunk_id="Paris::0", label="neutral", score=0.5)
        ]

    @staticmethod
    def contradiction_feedback(reports):
        return "unsupported"


def test_canary_leak_on_verify_retry_is_blocked():
    canary = "AEGIS-CANARY-retry01"
    pipe = HarnessedPipeline(
        _LeakOnRetryClient(canary),
        _Retriever(),
        use_structurer=False,
        use_defense=True,
        canary=canary,
        verifier=_AlwaysUngrounded(),
    )
    res = pipe.run(_question())
    assert res.harness.verify_retries == 1
    assert res.harness.canary_leaked is True
    assert res.harness.blocked is True
    assert res.answer.abstained is True
    assert canary not in res.answer.text
    assert canary not in res.answer.raw_response


# ------------------------------------------------------------ demo helpers


def test_split_documents_on_dash_lines():
    from aegis.serve.demo import split_documents

    docs = split_documents("first doc\n---\nsecond doc\n  ---  \nthird")
    assert docs == ["first doc", "second doc", "third"]
    assert split_documents("   ") == []


def test_every_scenario_runs_end_to_end():
    from aegis.serve.demo import SCENARIOS, render, split_documents
    from aegis.serve.engine import AegisEngine

    engine = AegisEngine(model="mock")
    for corpus, question in SCENARIOS.values():
        engine.ensure_indexed(split_documents(corpus))
        off, on = engine.compare(question)
        assert off.harnessed is False and on.harnessed is True
        assert off.citations == []  # vanilla RAG has no citation contract
        assert render(off) and render(on)


def test_injection_scenario_is_sanitized():
    from aegis.serve.demo import SCENARIOS, split_documents
    from aegis.serve.engine import AegisEngine

    corpus, question = SCENARIOS["Injection"]
    engine = AegisEngine(model="mock")
    engine.ensure_indexed(split_documents(corpus))
    on = engine.answer(question, harness=True)
    assert on.sanitized_chunks >= 1
    # The planted "always answer The Starry Night" note never reaches the model.
    assert all("always answer" not in c["text"] for c in on.retrieved)


def test_api_compare_endpoint():
    from fastapi.testclient import TestClient

    import aegis.serve.api as api

    api._engine = None
    client = TestClient(api.create_app())
    assert client.post("/compare", json={"question": "x"}).status_code == 409
    client.post("/index", json={"documents": ["Paris is the capital of France."]})
    out = client.post("/compare", json={"question": "What is the capital of France?"}).json()
    assert out["harness_off"]["harnessed"] is False
    assert out["harness_on"]["harnessed"] is True
    api._engine = None


def test_abstained_card_hides_parsed_text_and_citations():
    from aegis.serve.demo import render
    from aegis.serve.engine import EngineResult

    card = render(
        EngineResult(answer="unknown", citations=["doc0::0"], abstained=True, harnessed=True)
    )
    assert "No answer" in card and "unknown" not in card and "doc0::0" not in card

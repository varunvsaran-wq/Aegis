"""Tests for the Phase-2 LangGraph harness (aegis/graph.py).

Hermetic by construction: no network, no model downloads, and no module-scope
imports of aegis.structurer / aegis.verify (they are developed in parallel).
The structurer and verifier are injected as test doubles; the LLM is the
real deterministic mock client from aegis.gateway.
"""

import pytest

from aegis.gateway import get_client
from aegis.graph import HARNESS_TEMPLATES, HarnessedPipeline
from aegis.types import (
    Chunk,
    HotpotQuestion,
    LLMResponse,
    RetrievalResult,
    StructuredQuery,
    VerifierReport,
)

# --------------------------------------------------------------- test doubles


class FakeStructurer:
    """Returns a fixed StructuredQuery plus a token-bearing LLMResponse."""

    def __init__(self, sub_questions: list[str] | None = None):
        self.sub_questions = sub_questions
        self.calls = 0

    def structure(self, question: str):
        self.calls += 1
        subs = list(self.sub_questions) if self.sub_questions else [question]
        return (
            StructuredQuery(sub_questions=subs),
            LLMResponse(
                text="",
                model="mock",
                model_version="mock/echo",
                tokens_in=5,
                tokens_out=5,
            ),
        )


def _chunk(cid: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            id=cid,
            title=cid.split("::")[0],
            text=text,
            sentences=[text],
        ),
        rerank_score=score,
    )


DEFAULT_CHUNKS = [
    _chunk("Paris::0", "Paris is the capital of France. It lies on the Seine.", 0.9),
    _chunk("Berlin::0", "Berlin is the capital of Germany. It has museums.", 0.5),
    _chunk("Rome::0", "Rome is the capital of Italy. It is ancient.", 0.3),
]


class FakeRetriever:
    """Serves a fixed chunk set for every query and records the queries."""

    def __init__(self, results: list[RetrievalResult] | None = None):
        self.results = results if results is not None else DEFAULT_CHUNKS
        self.queries: list[str] = []

    def retrieve(self, query: str, k_final: int = 5, **kwargs):
        self.queries.append(query)
        return [r.model_copy(deep=True) for r in self.results[:k_final]]


class FakeVerifier:
    """Replays a configurable sequence of (grounded, reports) verdicts."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def verify(self, answer, retrieved, question):
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return verdict

    def feedback(self, reports):
        lines = [f"- '{r.claim}' vs {r.chunk_id}: {r.label}" for r in reports]
        return "\n".join(lines) or "no verifier reports"


def _report(label: str = "entailment") -> VerifierReport:
    return VerifierReport(
        claim="Paris is the capital of France",
        chunk_id="Paris::0",
        label=label,
        score=0.95,
    )


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


# ---------------------------------------------------------------------- tests


def test_verify_retry_template_contract():
    template = HARNESS_TEMPLATES["verify_retry"]
    assert "{feedback}" in template
    assert "ANSWER:" in template
    assert "CITATIONS:" in template


def test_happy_path_kvote_1():
    client = get_client("mock")
    verifier = FakeVerifier([(True, [_report()])])
    pipe = HarnessedPipeline(
        client,
        FakeRetriever(),
        structurer=FakeStructurer(),
        verifier=verifier,
    )
    res = pipe.run(_question())

    assert res.mode == "harnessed"
    assert res.model == "mock"
    assert res.answer.abstained is False
    assert [c.chunk_id for c in res.answer.citations] == ["Paris::0"]
    assert "capital of France" in res.answer.text
    assert res.harness is not None
    assert res.harness.grounded is True
    assert res.harness.verify_retries == 0
    assert res.harness.agreement is None
    # 1 structurer call + 1 generation.
    assert res.harness.llm_calls == 2
    # Usage aggregated over ALL calls (structurer contributes 5/5 tokens).
    assert res.tokens_in >= 5 and res.tokens_out >= 5
    assert res.cost_usd == 0.0
    assert res.latency_s >= 0.0
    assert verifier.calls == 1


def test_verify_retry_path():
    client = get_client("mock")
    verifier = FakeVerifier([(False, [_report("contradiction")]), (True, [_report()])])
    structurer = FakeStructurer()
    pipe = HarnessedPipeline(
        client, FakeRetriever(), structurer=structurer, verifier=verifier
    )
    res = pipe.run(_question())

    assert res.harness.verify_retries == 1
    assert res.harness.grounded is True
    assert res.answer.abstained is False
    # 1 structurer + 1 generation + 1 verify retry.
    assert structurer.calls == 1
    assert res.harness.llm_calls == 3
    assert verifier.calls == 2
    # Final reports are those of the LAST verification.
    assert res.harness.verifier_reports[0].label == "entailment"


def test_abstain_when_never_grounded():
    client = get_client("mock")
    verifier = FakeVerifier([(False, [_report("contradiction")])])
    pipe = HarnessedPipeline(
        client, FakeRetriever(), structurer=FakeStructurer(), verifier=verifier
    )
    res = pipe.run(_question())

    assert res.answer.abstained is True
    assert res.harness.verify_retries == 1
    assert res.harness.grounded is False
    assert verifier.calls == 2  # initial verify + post-retry verify
    # The parsed answer text is kept even though the answer is abstained.
    assert res.answer.text


def test_multi_query_retrieval_merges_and_dedups():
    client = get_client("mock")
    retriever = FakeRetriever()
    structurer = FakeStructurer(
        sub_questions=[
            "Which city is the capital of France?",
            "What river runs through Paris?",
        ]
    )
    pipe = HarnessedPipeline(
        client,
        retriever,
        k_final=2,
        structurer=structurer,
        verifier=FakeVerifier([(True, [_report()])]),
    )
    res = pipe.run(_question())

    # Original question first, then the two distinct sub-questions.
    assert retriever.queries == [
        "What is the capital of France?",
        "Which city is the capital of France?",
        "What river runs through Paris?",
    ]
    ids = [r.chunk.id for r in res.retrieved]
    assert len(ids) == len(set(ids)), "merged results must be deduped by chunk id"
    assert len(res.retrieved) <= 2
    # Sorted by rerank_score descending.
    scores = [r.rerank_score for r in res.retrieved]
    assert scores == sorted(scores, reverse=True)


def test_kvote_3_unanimous_mock():
    client = get_client("mock")
    pipe = HarnessedPipeline(
        client,
        FakeRetriever(),
        k_vote=3,
        use_structurer=False,
        verifier=FakeVerifier([(True, [_report()])]),
    )
    res = pipe.run(_question())

    assert res.harness.agreement == 1.0
    assert len(res.harness.votes) == 3
    # No structurer; all 3 samples counted.
    assert res.harness.llm_calls == 3
    assert res.answer.abstained is False
    assert [c.chunk_id for c in res.answer.citations] == ["Paris::0"]
    assert res.answer.text in res.harness.votes


class RealShapedVerifier:
    """Verifier double mirroring GroundednessVerifier's API (no ``.feedback``).

    Regression guard: the retry path must fall back to
    ``contradiction_feedback`` without importing a non-existent module symbol.
    """

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def verify(self, answer, retrieved, question):
        v = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return v

    @staticmethod
    def contradiction_feedback(reports):
        return "not supported: " + ", ".join(r.chunk_id for r in reports)


def test_retry_feedback_falls_back_to_contradiction_feedback():
    client = get_client("mock")
    verifier = RealShapedVerifier(
        [(False, [_report("contradiction")]), (True, [_report()])]
    )
    pipe = HarnessedPipeline(
        client, FakeRetriever(), structurer=FakeStructurer(), verifier=verifier
    )
    res = pipe.run(_question())
    # The retry ran (no ImportError) and the answer recovered.
    assert res.harness.verify_retries == 1
    assert res.harness.grounded is True
    assert verifier.calls == 2


def test_structurer_and_verifier_disabled():
    client = get_client("mock")
    retriever = FakeRetriever()
    pipe = HarnessedPipeline(
        client, retriever, use_structurer=False, use_verifier=False
    )
    q = _question()
    res = pipe.run(q)

    assert res.answer.abstained is False
    assert res.harness.grounded is None
    assert res.harness.verifier_reports == []
    assert res.harness.verify_retries == 0
    # Minimal pass-through structuring: just the original question.
    assert res.harness.structured is not None
    assert res.harness.structured.sub_questions == [q.question]
    assert retriever.queries == [q.question]
    # Single generation call, nothing else.
    assert res.harness.llm_calls == 1


class ClaimAwareVerifier:
    """Records the claims the harness passes in."""

    def __init__(self):
        self.seen_claims = []

    def verify(self, answer, retrieved, question, claims=None):
        self.seen_claims.append(claims)
        return True, [_report()]


class _ClaimClient:
    """Answers the contract, then returns a fixed sentence for the claim prompt."""

    model = "mock"

    def __init__(self):
        self.calls = []

    def complete(self, messages, max_tokens=1024):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if prompt.startswith("Rewrite the question and its answer"):
            text = "Paris is the capital of France."
        else:
            text = "ANSWER: Paris\nCITATIONS: Paris::0"
        return LLMResponse(text=text, model="mock", model_version="mock/echo", cost_usd=0.001)


def test_verifier_checks_the_rewritten_claim_and_its_cost_is_counted():
    client = _ClaimClient()
    verifier = ClaimAwareVerifier()
    pipe = HarnessedPipeline(client, FakeRetriever(), use_structurer=False, verifier=verifier)
    res = pipe.run(_question())
    assert verifier.seen_claims == [["Paris is the capital of France."]]
    # generation + claim rewrite, both billed to the question.
    assert res.harness.llm_calls == 2
    assert res.cost_usd == pytest.approx(0.002)


def test_claim_rewrite_can_be_disabled():
    client = _ClaimClient()
    verifier = ClaimAwareVerifier()
    pipe = HarnessedPipeline(
        client, FakeRetriever(), use_structurer=False, verifier=verifier, rewrite_claims=False
    )
    res = pipe.run(_question())
    assert verifier.seen_claims == [None]
    assert res.harness.llm_calls == 1

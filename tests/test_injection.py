"""Integration tests: injection defense in the harness + ASR scoring.

Hermetic: the deterministic mock client, a fixed fake retriever serving a
poisoned chunk, and injected structurer/verifier doubles. No downloads.
"""

from aegis.eval.poison import poison_corpus
from aegis.eval.scorers import injection_outcomes
from aegis.gateway import get_client
from aegis.graph import HarnessedPipeline
from aegis.pipeline import RAGPipeline
from aegis.types import (
    Answer,
    Chunk,
    Citation,
    HotpotQuestion,
    QueryResult,
    RetrievalResult,
)


def _poisoned_result_chunk() -> Chunk:
    corpus = [
        Chunk(
            id="Paris::0",
            title="Paris",
            text="Paris is the capital of France.",
            sentences=["Paris is the capital of France."],
            metadata={"poisoned": False},
        )
    ]
    # Force an override/injection (non-exfiltration) attack by using a seed
    # whose first (only) attack is category-index 0 == instruction_injection.
    poisoned, manifest = poison_corpus(corpus, rate=1.0, seed=0)
    return poisoned[0]


class FakeRetriever:
    def __init__(self, chunk: Chunk):
        self._chunk = chunk

    def retrieve(self, query: str, k_final: int = 5, **kwargs):
        return [RetrievalResult(chunk=self._chunk.model_copy(deep=True), rerank_score=0.9)]


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


def test_raw_pipeline_is_vulnerable_harness_defends():
    chunk = _poisoned_result_chunk()
    sentinel = chunk.metadata["attack_sentinel"]
    assert sentinel  # this seed yields a sentinel-bearing attack

    # Raw RAG: the mock echoes the poisoned first sentence -> sentinel leaks.
    raw = RAGPipeline(get_client("mock"), retriever=FakeRetriever(chunk), mode="raw_rag")
    raw_res = raw.run(_question())
    raw_out = f"{raw_res.answer.text}\n{raw_res.answer.raw_response}"
    assert sentinel in raw_out

    # Harnessed with defense: the injected sentence is stripped before the
    # model sees it, so the sentinel never reaches the output.
    harnessed = HarnessedPipeline(
        get_client("mock"),
        FakeRetriever(chunk),
        use_structurer=False,
        use_verifier=False,
        use_defense=True,
    )
    h_res = harnessed.run(_question())
    h_out = f"{h_res.answer.text}\n{h_res.answer.raw_response}"
    assert sentinel not in h_out
    assert h_res.harness.sanitized_chunks >= 1


class FakeCanaryClient:
    """A client that leaks the canary verbatim in a well-formed answer."""

    def __init__(self, canary: str):
        self.canary = canary
        self.model = "mock"

    def complete(self, messages, max_tokens: int = 1024):
        from aegis.types import LLMResponse

        return LLMResponse(
            text=f"ANSWER: the secret is {self.canary}\nCITATIONS: Paris::0",
            model="mock",
            model_version="mock/echo",
            tokens_in=5,
            tokens_out=5,
        )


def test_canary_leak_is_blocked_and_scrubbed():
    canary = "AEGIS-CANARY-abc123"
    chunk = Chunk(
        id="Paris::0",
        title="Paris",
        text="Paris is the capital of France.",
        sentences=["Paris is the capital of France."],
        metadata={"poisoned": False},
    )
    pipe = HarnessedPipeline(
        FakeCanaryClient(canary),
        FakeRetriever(chunk),
        use_structurer=False,
        use_verifier=False,
        use_defense=True,
        canary=canary,
    )
    res = pipe.run(_question())
    assert res.harness.canary_leaked is True
    assert res.harness.blocked is True
    assert res.answer.abstained is True
    # The secret must not survive anywhere in the returned answer.
    assert canary not in res.answer.text
    assert canary not in res.answer.raw_response


def _result_with(chunk: Chunk, answer_text: str) -> QueryResult:
    return QueryResult(
        question_id="q1",
        question="q?",
        gold_answer="Paris",
        answer=Answer(text=answer_text, citations=[Citation(chunk_id=chunk.id)]),
        retrieved=[RetrievalResult(chunk=chunk, rerank_score=0.9)],
        mode="raw_rag",
        model="mock",
    )


def test_injection_outcomes_counts_success_by_category():
    chunk = _poisoned_result_chunk()
    sentinel = chunk.metadata["attack_sentinel"]
    category = chunk.metadata["attack_category"]

    hit = _result_with(chunk, f"the answer is {sentinel}")
    miss = _result_with(chunk, "Paris")

    out = injection_outcomes([hit, miss], canary="")
    assert out["n"] == 2
    assert out["attacked"] == [True, True]
    assert out["success"] == [True, False]
    assert out["asr"] == 0.5
    assert out["asr_by_category"][category]["n"] == 2
    assert out["asr_by_category"][category]["k"] == 1
    assert out["asr_by_category"][category]["rate"] == 0.5


def test_injection_outcomes_ignores_clean_corpus():
    clean = Chunk(
        id="C::0", title="C", text="clean", sentences=["clean"], metadata={"poisoned": False}
    )
    out = injection_outcomes([_result_with(clean, "anything")], canary="")
    assert out["attacked"] == [False]
    assert out["asr"] == 0.0

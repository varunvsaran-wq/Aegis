"""Tests for aegis.pipeline and aegis.eval.scorers (mock client, no network).

Uses a small in-test fake retriever rather than aegis.retrieve so these tests
stay decoupled from the retrieval module.
"""

import pytest

from aegis.eval.scorers import (
    citation_precision,
    exact_match,
    f1_score,
    judge_correct,
    normalize_answer,
    retrieval_precision_recall,
    score_results,
)
from aegis.gateway import get_client
from aegis.pipeline import PROMPT_TEMPLATES, RAGPipeline, parse_contract
from aegis.types import (
    Answer,
    Chunk,
    Citation,
    HotpotQuestion,
    LLMResponse,
    QueryResult,
    RetrievalResult,
)

# --------------------------------------------------------------------------
# Fixtures / fakes
# --------------------------------------------------------------------------

CHUNK_PARIS = Chunk(
    id="Paris::0",
    title="Paris",
    text="Paris is the capital of France. It is known for the Eiffel Tower.",
    sentences=[
        "Paris is the capital of France.",
        "It is known for the Eiffel Tower.",
    ],
)
CHUNK_BERLIN = Chunk(
    id="Berlin::0",
    title="Berlin",
    text="Berlin is the capital of Germany. It has the Brandenburg Gate.",
    sentences=[
        "Berlin is the capital of Germany.",
        "It has the Brandenburg Gate.",
    ],
)
CHUNK_LONDON = Chunk(
    id="London::0",
    title="London",
    text="London is the capital of England. The Thames flows through it.",
    sentences=[
        "London is the capital of England.",
        "The Thames flows through it.",
    ],
)


class FakeRetriever:
    """Returns a fixed list of RetrievalResults regardless of the query."""

    def __init__(self, chunks: list[Chunk]):
        self._results = [
            RetrievalResult(chunk=c, fused_score=1.0 - 0.1 * i)
            for i, c in enumerate(chunks)
        ]

    def retrieve(self, query, k_dense=20, k_sparse=20, k_final=5):
        return self._results[:k_final]


class ScriptedClient:
    """Client returning canned texts in order and counting calls."""

    model = "scripted"

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.calls = 0

    def complete(self, messages, max_tokens=1024):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return LLMResponse(
            text=text,
            model=self.model,
            model_version="scripted/fake",
            tokens_in=7,
            tokens_out=3,
            cost_usd=0.001,
            latency_s=0.01,
        )


class ExplodingClient:
    """Fails the test if any completion is attempted."""

    model = "exploding"

    def complete(self, messages, max_tokens=1024):
        raise AssertionError("LLM must not be called for this case")


def make_question(**overrides) -> HotpotQuestion:
    base = dict(
        id="q1",
        question="What is the capital of France?",
        answer="Paris is the capital of France",
        type="bridge",
        level="easy",
        supporting_facts=[("Paris", 0)],
        context=[("Paris", CHUNK_PARIS.sentences)],
    )
    base.update(overrides)
    return HotpotQuestion(**base)


# --------------------------------------------------------------------------
# normalize / EM / F1
# --------------------------------------------------------------------------


def test_normalize_answer_official_behavior():
    assert normalize_answer("The Apple") == "apple"
    assert normalize_answer("An  apple, a day!") == "apple day"
    assert normalize_answer("YES") == "yes"


def test_exact_match_ignores_articles_case_punct():
    assert exact_match("The Apple", "apple") is True
    assert exact_match("apples", "apple") is False


def test_f1_partial_overlap():
    # normalized: "cat sat" vs "cat" -> precision 0.5, recall 1.0, f1 2/3
    assert f1_score("the cat sat", "cat") == pytest.approx(2 / 3)


def test_f1_yes_no():
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("no", "no") == 1.0
    assert f1_score("yes", "no") == 0.0
    # Official guard: yes/no must match exactly, no partial credit.
    assert f1_score("yes indeed", "yes") == 0.0


def test_f1_empty_edge_cases():
    assert f1_score("", "") == 1.0
    assert f1_score("the", "a") == 1.0  # both normalize to empty
    assert f1_score("", "Paris") == 0.0
    assert f1_score("Paris", "") == 0.0


# --------------------------------------------------------------------------
# parse_contract
# --------------------------------------------------------------------------


def test_parse_contract_tolerant_case_and_spacing():
    answer, cites = parse_contract(
        "answer: X\ncitations: A::0, B::1", valid_ids={"A::0", "B::1"}
    )
    assert answer == "X"
    assert cites == ["A::0", "B::1"]


def test_parse_contract_drops_hallucinated_ids():
    answer, cites = parse_contract(
        "ANSWER: X\nCITATIONS: A::0, Fake::9", valid_ids={"A::0"}
    )
    assert answer == "X"
    assert cites == ["A::0"]


def test_parse_contract_strips_brackets_and_chunk_prefix():
    _, cites = parse_contract(
        "ANSWER: X\nCITATIONS: [[chunk:A::0]], [B::1]", valid_ids={"A::0", "B::1"}
    )
    assert cites == ["A::0", "B::1"]


def test_parse_contract_missing_citations_line():
    answer, cites = parse_contract("ANSWER: X", valid_ids={"A::0"})
    assert answer == "X"
    assert cites == []


def test_parse_contract_missing_everything():
    assert parse_contract("I refuse.", valid_ids={"A::0"}) == ("", [])


# --------------------------------------------------------------------------
# Pipeline modes end-to-end (mock client)
# --------------------------------------------------------------------------


def test_raw_rag_end_to_end_with_mock():
    client = get_client("mock")
    retriever = FakeRetriever([CHUNK_PARIS, CHUNK_BERLIN])
    pipeline = RAGPipeline(client, retriever=retriever, mode="raw_rag")
    q = make_question()  # gold = first sentence of top chunk (sans period)

    result = pipeline.run(q)

    assert result.mode == "raw_rag"
    assert result.model == "mock"
    assert result.question_id == q.id
    assert result.gold_answer == q.answer
    assert result.answer.abstained is False
    assert result.answer.text == "Paris is the capital of France"
    assert [c.chunk_id for c in result.answer.citations] == [CHUNK_PARIS.id]
    assert len(result.retrieved) == 2
    assert result.answer.raw_response  # last raw LLM text preserved
    assert result.cost_usd >= 0.0
    assert result.latency_s >= 0.0
    assert result.tokens_in > 0
    assert result.tokens_out > 0
    # Downstream: the mock's answer exactly matches gold.
    assert exact_match(result.answer.text, q.answer) is True


def test_raw_rag_abstains_after_failed_retry_with_two_calls():
    client = ScriptedClient(["I refuse to follow the format.", "Still no citations."])
    retriever = FakeRetriever([CHUNK_PARIS, CHUNK_BERLIN])
    pipeline = RAGPipeline(client, retriever=retriever, mode="raw_rag")

    result = pipeline.run(make_question())

    assert client.calls == 2  # one initial + exactly one retry
    assert result.answer.abstained is True
    assert result.answer.citations == []
    assert result.answer.raw_response == "Still no citations."
    # Costs/tokens aggregated across BOTH calls.
    assert result.cost_usd == pytest.approx(0.002)
    assert result.tokens_in == 14
    assert result.tokens_out == 6


def test_raw_rag_retry_recovers_valid_citations():
    client = ScriptedClient(
        ["No format here.", "ANSWER: Paris\nCITATIONS: Paris::0"]
    )
    retriever = FakeRetriever([CHUNK_PARIS])
    pipeline = RAGPipeline(client, retriever=retriever, mode="raw_rag")

    result = pipeline.run(make_question())

    assert client.calls == 2
    assert result.answer.abstained is False
    assert [c.chunk_id for c in result.answer.citations] == ["Paris::0"]
    assert result.answer.text == "Paris"


def test_closed_book_no_citations_no_retrieval():
    client = get_client("mock")
    pipeline = RAGPipeline(client, mode="closed_book")

    result = pipeline.run(make_question())

    assert result.mode == "closed_book"
    assert result.answer.citations == []
    assert result.retrieved == []
    assert result.answer.text == "I don't know."  # mock fallback
    assert result.answer.abstained is False


def test_vanilla_rag_populates_retrieved():
    client = get_client("mock")
    retriever = FakeRetriever([CHUNK_PARIS, CHUNK_BERLIN, CHUNK_LONDON])
    pipeline = RAGPipeline(client, retriever=retriever, mode="vanilla_rag", k_final=2)

    result = pipeline.run(make_question())

    assert result.mode == "vanilla_rag"
    assert [r.chunk.id for r in result.retrieved] == ["Paris::0", "Berlin::0"]
    assert result.answer.citations == []  # no citation contract in vanilla mode
    assert result.answer.text  # raw stripped response


def test_rag_modes_require_retriever():
    client = get_client("mock")
    with pytest.raises(AssertionError):
        RAGPipeline(client, mode="raw_rag")
    with pytest.raises(AssertionError):
        RAGPipeline(client, mode="vanilla_rag")
    with pytest.raises(AssertionError):
        RAGPipeline(client, retriever=FakeRetriever([]), mode="not_a_mode")


def test_prompt_templates_complete_and_marker_compatible():
    assert set(PROMPT_TEMPLATES) == {
        "closed_book",
        "vanilla_rag",
        "raw_rag_system",
        "raw_rag_user",
        "citation_retry",
    }
    assert "Question: {question}" in PROMPT_TEMPLATES["closed_book"]
    assert "ANSWER:" in PROMPT_TEMPLATES["raw_rag_system"]
    assert "CITATIONS:" in PROMPT_TEMPLATES["raw_rag_system"]


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------


def test_judge_correct_with_mock():
    client = get_client("mock")
    q = "What is the capital of France?"
    assert judge_correct(client, q, "Paris", "paris") is True
    assert judge_correct(client, q, "Paris", "London") is False


def test_judge_abstained_candidate_no_llm_call():
    assert judge_correct(ExplodingClient(), "Q?", "Paris", "") is False
    assert judge_correct(ExplodingClient(), "Q?", "Paris", "   ") is False


# --------------------------------------------------------------------------
# Retrieval / citation metrics
# --------------------------------------------------------------------------


def test_retrieval_precision_recall():
    q = make_question(supporting_facts=[("Paris", 0), ("Berlin", 1), ("Berlin", 0)])
    retrieved = [
        RetrievalResult(chunk=CHUNK_PARIS),
        RetrievalResult(chunk=CHUNK_BERLIN),
        RetrievalResult(chunk=CHUNK_LONDON),
    ]
    precision, recall = retrieval_precision_recall(retrieved, q)
    assert precision == pytest.approx(2 / 3)  # gold titles deduped to 2
    assert recall == pytest.approx(1.0)

    # Empty retrieval -> precision 0.0; empty gold -> recall 1.0.
    p, r = retrieval_precision_recall([], q)
    assert (p, r) == (0.0, 0.0)
    p, r = retrieval_precision_recall(retrieved, make_question(supporting_facts=[]))
    assert (p, r) == (0.0, 1.0)


def _query_result(citations: list[str], retrieved: list[Chunk], **overrides):
    base = dict(
        question_id="q1",
        question="What is the capital of France?",
        gold_answer="Paris is the capital of France",
        answer=Answer(
            text="Paris",
            citations=[Citation(chunk_id=c) for c in citations],
        ),
        retrieved=[RetrievalResult(chunk=c) for c in retrieved],
        mode="raw_rag",
        model="mock",
    )
    base.update(overrides)
    return QueryResult(**base)


def test_citation_precision():
    q = make_question(supporting_facts=[("Paris", 0)])
    result = _query_result(
        ["Paris::0", "London::0"], [CHUNK_PARIS, CHUNK_LONDON]
    )
    assert citation_precision(result, q) == pytest.approx(0.5)

    # No citations -> None so it can be averaged over cited answers only.
    assert citation_precision(_query_result([], [CHUNK_PARIS]), q) is None


# --------------------------------------------------------------------------
# score_results aggregation
# --------------------------------------------------------------------------


def test_score_results_end_to_end():
    q = make_question()
    client = get_client("mock")
    pipeline = RAGPipeline(
        client, retriever=FakeRetriever([CHUNK_PARIS, CHUNK_BERLIN]), mode="raw_rag"
    )
    results = [pipeline.run(q)]

    scores = score_results(results, [q], judge_client=client)

    assert scores["em"] == [True]
    assert scores["f1"] == [pytest.approx(1.0)]
    assert scores["judge"] == [True]
    assert scores["retrieval_precision"] == [pytest.approx(0.5)]
    assert scores["retrieval_recall"] == [pytest.approx(1.0)]
    assert scores["citation_precision"] == [pytest.approx(1.0)]
    assert scores["abstain_rate"] == 0.0
    assert scores["em_mean"] == 1.0
    assert scores["f1_mean"] == pytest.approx(1.0)
    assert scores["citation_precision_mean"] == pytest.approx(1.0)


def test_score_results_no_judge_and_mismatch():
    q = make_question()
    result = _query_result(["Paris::0"], [CHUNK_PARIS])
    scores = score_results([result], [q])
    assert scores["judge"] == [None]
    assert scores["judge_mean"] is None

    with pytest.raises(KeyError):
        score_results([_query_result([], [], question_id="missing")], [q])


# ---------------------------------------------------------------------------
# Citation resolution: models abbreviate chunk ids.
# ---------------------------------------------------------------------------


def test_parse_contract_resolves_bare_indices_and_titles():
    from aegis.pipeline import parse_contract

    valid = ["Damon Stoudamire::60", "Terrence Jones::62", "Don Haskins::61"]
    text = "ANSWER: Terrence Jones\nCITATIONS: 60, 62"
    assert parse_contract(text, valid) == (
        "Terrence Jones",
        ["Damon Stoudamire::60", "Terrence Jones::62"],
    )
    assert parse_contract("ANSWER: x\nCITATIONS: Terrence Jones", valid)[1] == [
        "Terrence Jones::62"
    ]
    assert parse_contract("ANSWER: x\nCITATIONS: terrence jones :: 62", valid)[1] == [
        "Terrence Jones::62"
    ]


def test_parse_contract_still_drops_unknown_and_ambiguous():
    from aegis.pipeline import parse_contract

    valid = ["A::1", "B::1", "C::2"]
    # "1" matches two chunks -> ambiguous -> dropped; "99" and "Z" unknown.
    assert parse_contract("ANSWER: x\nCITATIONS: 1, 99, Z, 2", valid)[1] == ["C::2"]


def test_parse_contract_ignores_markdown_emphasis():
    from aegis.pipeline import parse_contract

    text = "**ANSWER:** Tennis\n\n**CITATIONS:** 278, 277"
    assert parse_contract(text, ["Angelique Kerber::278", "Justin Gimelstob::277"]) == (
        "Tennis",
        ["Angelique Kerber::278", "Justin Gimelstob::277"],
    )

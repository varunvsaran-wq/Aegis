"""Tests for the groundedness verifier and consistency vote (aegis.verify).

Uses FakeNLI exclusively — no model downloads, and CrossEncoderNLI is never
constructed here.
"""

import pytest

from aegis.types import Answer, Chunk, Citation, RetrievalResult
from aegis.verify import FakeNLI, GroundednessVerifier, majority_vote


def _rr(chunk_id: str, title: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(id=chunk_id, title=title, text=text, sentences=[text])
    )


@pytest.fixture
def retrieved() -> list[RetrievalResult]:
    return [
        _rr(
            "France::0",
            "France",
            "Paris is the capital of France, which is what every atlas says.",
        ),
        _rr(
            "Banana::0",
            "Banana",
            "Bananas grow in tropical climates and taste sweet.",
        ),
        _rr(
            "Louvre::0",
            "Louvre",
            "The Louvre museum in Paris holds the Mona Lisa.",
        ),
    ]


QUESTION = "What is the capital of France"


# ---------------------------------------------------------------------------
# FakeNLI
# ---------------------------------------------------------------------------


class TestFakeNLI:
    def test_full_overlap_is_entailment(self):
        label, score = FakeNLI().predict(
            "paris is the capital city of france", "The capital is Paris"
        )
        # Content tokens {capital, paris} both appear in the premise.
        assert label == "entailment"
        assert score == 1.0

    def test_disjoint_is_contradiction(self):
        label, score = FakeNLI().predict(
            "bananas grow in tropical climates", "The capital is Paris"
        )
        assert label == "contradiction"
        assert score == 1.0  # round(1 - 0.0, 4)

    def test_partial_overlap_is_neutral(self):
        # Content tokens {capital, berlin}: only "capital" is in the premise.
        label, score = FakeNLI().predict(
            "paris is the capital of france", "The capital is Berlin"
        )
        assert label == "neutral"
        assert score == 0.5

    def test_entailment_boundary_at_0_6(self):
        # 3 of 5 content tokens present -> overlap exactly 0.6 -> entailment.
        label, score = FakeNLI().predict(
            "alpha beta gamma", "alpha beta gamma delta epsilon"
        )
        assert label == "entailment"
        assert score == 0.6

    def test_deterministic(self):
        nli = FakeNLI()
        pair = ("paris is the capital of france", "The capital is Berlin")
        assert nli.predict(*pair) == nli.predict(*pair)
        assert FakeNLI().predict(*pair) == nli.predict(*pair)

    def test_predict_batch_matches_predict(self):
        nli = FakeNLI()
        pairs = [
            ("paris is the capital of france", "The capital is Paris"),
            ("bananas grow in tropical climates", "The capital is Paris"),
        ]
        assert nli.predict_batch(pairs) == [nli.predict(*p) for p in pairs]


# ---------------------------------------------------------------------------
# GroundednessVerifier
# ---------------------------------------------------------------------------


class TestGroundednessVerifier:
    def test_supported_answer_is_grounded(self, retrieved):
        verifier = GroundednessVerifier(FakeNLI())
        answer = Answer(text="Paris", citations=[Citation(chunk_id="France::0")])
        grounded, reports = verifier.verify(answer, retrieved, QUESTION)
        assert grounded is True
        assert len(reports) == 1
        assert reports[0].label == "entailment"
        assert reports[0].chunk_id == "France::0"
        assert reports[0].claim == (
            "The answer to the question 'What is the capital of France' is: Paris."
        )

    def test_unrelated_citation_is_not_grounded(self, retrieved):
        verifier = GroundednessVerifier(FakeNLI())
        answer = Answer(text="Paris", citations=[Citation(chunk_id="Banana::0")])
        grounded, reports = verifier.verify(answer, retrieved, QUESTION)
        assert grounded is False
        assert len(reports) == 1
        assert all(r.label != "entailment" for r in reports)

    def test_one_entailing_chunk_suffices(self, retrieved):
        verifier = GroundednessVerifier(FakeNLI())
        answer = Answer(
            text="Paris",
            citations=[Citation(chunk_id="Banana::0"), Citation(chunk_id="France::0")],
        )
        grounded, reports = verifier.verify(answer, retrieved, QUESTION)
        assert grounded is True
        assert [r.chunk_id for r in reports] == ["Banana::0", "France::0"]

    def test_abstained_answer_returns_false_empty(self, retrieved):
        verifier = GroundednessVerifier(FakeNLI())
        answer = Answer(
            text="Paris", abstained=True, citations=[Citation(chunk_id="France::0")]
        )
        assert verifier.verify(answer, retrieved, QUESTION) == (False, [])

    def test_empty_text_and_no_citations_return_false_empty(self, retrieved):
        verifier = GroundednessVerifier(FakeNLI())
        empty = Answer(text="   ", citations=[Citation(chunk_id="France::0")])
        uncited = Answer(text="Paris", citations=[])
        assert verifier.verify(empty, retrieved, QUESTION) == (False, [])
        assert verifier.verify(uncited, retrieved, QUESTION) == (False, [])

    def test_unknown_chunk_id_skipped_without_crash(self, retrieved):
        verifier = GroundednessVerifier(FakeNLI())
        answer = Answer(
            text="Paris",
            citations=[Citation(chunk_id="Nope::9"), Citation(chunk_id="France::0")],
        )
        grounded, reports = verifier.verify(answer, retrieved, QUESTION)
        assert grounded is True
        assert [r.chunk_id for r in reports] == ["France::0"]

        only_unknown = Answer(text="Paris", citations=[Citation(chunk_id="Nope::9")])
        assert verifier.verify(only_unknown, retrieved, QUESTION) == (False, [])

    def test_contradiction_feedback_mentions_failing_chunk(self, retrieved):
        verifier = GroundednessVerifier(FakeNLI())
        answer = Answer(
            text="Paris",
            citations=[Citation(chunk_id="Banana::0"), Citation(chunk_id="France::0")],
        )
        _, reports = verifier.verify(answer, retrieved, QUESTION)
        feedback = verifier.contradiction_feedback(reports)
        assert "Banana::0" in feedback
        assert "contradiction" in feedback
        # The entailing chunk is not reported as failing.
        assert "France::0" not in feedback


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_case_insensitive_majority(self):
        answers = ["Paris", "paris", "London"]
        winner, agreement, votes = majority_vote(answers)
        assert winner == "Paris"
        assert agreement == pytest.approx(2 / 3)
        assert votes == answers

    def test_all_empty(self):
        assert majority_vote([]) == ("", 0.0, [])
        answers = ["", "   "]
        assert majority_vote(answers) == ("", 0.0, answers)

    def test_tie_broken_by_first_occurrence(self):
        answers = ["London", "Paris", "paris", "london"]
        winner, agreement, _ = majority_vote(answers)
        assert winner == "London"
        assert agreement == pytest.approx(0.5)

    def test_articles_normalized_into_one_cluster(self):
        answers = ["the apple", "Apple"]
        winner, agreement, _ = majority_vote(answers)
        assert winner == "the apple"
        assert agreement == 1.0

    def test_empty_answers_excluded_from_denominator(self):
        winner, agreement, _ = majority_vote(["Paris", "", "Paris"])
        assert winner == "Paris"
        assert agreement == 1.0

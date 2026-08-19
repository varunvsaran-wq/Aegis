"""Unit tests for aegis.structurer (mock model only — no network)."""

from aegis.gateway import get_client
from aegis.structurer import STRUCTURER_TEMPLATES, QueryStructurer, parse_structured
from aegis.types import LLMResponse, StructuredQuery

MULTI_HOP = "Who directed Titanic and who starred in it?"
YES_NO = "Were Scott Derrickson and Ed Wood of the same nationality?"


def _structurer() -> QueryStructurer:
    return QueryStructurer(get_client("mock"))


class TestTemplate:
    def test_required_markers_present(self):
        template = STRUCTURER_TEMPLATES["structure"]
        assert "Return ONLY a JSON object" in template
        for key in ("intent", "sub_questions", "keywords", "answer_type"):
            assert key in template
        assert any(
            line.startswith("Question: {question}")
            for line in template.splitlines()
        )


class TestStructureMultiHop:
    def test_splits_into_two_sub_questions(self):
        structured, _ = _structurer().structure(MULTI_HOP)
        assert isinstance(structured, StructuredQuery)
        assert len(structured.sub_questions) == 2

    def test_keywords_include_titanic(self):
        structured, _ = _structurer().structure(MULTI_HOP)
        assert "titanic" in structured.keywords

    def test_answer_type_short_phrase(self):
        structured, _ = _structurer().structure(MULTI_HOP)
        assert structured.answer_type == "short_phrase"

    def test_llm_response_returned_with_token_counts(self):
        _, response = _structurer().structure(MULTI_HOP)
        assert isinstance(response, LLMResponse)
        assert response.tokens_in > 0
        assert response.tokens_out > 0


class TestStructureYesNo:
    def test_answer_type_yes_no(self):
        structured, _ = _structurer().structure(YES_NO)
        assert structured.answer_type == "yes_no"

    def test_sub_questions_non_empty(self):
        structured, _ = _structurer().structure(YES_NO)
        assert structured.sub_questions


class TestParseStructuredTolerance:
    QUESTION = "Who wrote Dracula?"

    def test_json_fences_parse(self):
        text = (
            "```json\n"
            '{"intent": "find author", "sub_questions": ["Who wrote Dracula?"],'
            ' "keywords": ["dracula"], "answer_type": "entity"}\n'
            "```"
        )
        result = parse_structured(text, self.QUESTION)
        assert result.intent == "find author"
        assert result.sub_questions == ["Who wrote Dracula?"]
        assert result.keywords == ["dracula"]
        assert result.answer_type == "entity"

    def test_unknown_keys_are_dropped(self):
        text = (
            '{"intent": "x", "sub_questions": ["Who wrote Dracula?"],'
            ' "keywords": ["dracula"], "answer_type": "entity",'
            ' "confidence": 0.9, "reasoning": "because"}'
        )
        result = parse_structured(text, self.QUESTION)
        assert isinstance(result, StructuredQuery)
        assert result.answer_type == "entity"
        assert not hasattr(result, "confidence")

    def test_garbage_falls_back(self):
        result = parse_structured("total nonsense, no json here", self.QUESTION)
        assert result.intent == ""
        assert result.sub_questions == [self.QUESTION]
        assert result.keywords == []
        assert result.answer_type == "short_phrase"

    def test_sub_questions_string_coerced_or_fallback(self):
        text = (
            '{"intent": "x", "sub_questions": "Who wrote Dracula?",'
            ' "keywords": ["dracula"], "answer_type": "entity"}'
        )
        result = parse_structured(text, self.QUESTION)
        assert isinstance(result, StructuredQuery)
        assert result.sub_questions

    def test_never_raises_on_wrong_top_level_type(self):
        result = parse_structured('["not", "an", "object"]', self.QUESTION)
        assert result.sub_questions == [self.QUESTION]

    def test_sub_questions_clamped_to_four(self):
        text = (
            '{"intent": "x", "sub_questions": ["a?", "b?", "c?", "d?", "e?", "f?"],'
            ' "keywords": [], "answer_type": "short_phrase"}'
        )
        result = parse_structured(text, self.QUESTION)
        assert len(result.sub_questions) == 4

    def test_invalid_answer_type_degrades(self):
        text = (
            '{"intent": "x", "sub_questions": ["q?"], "keywords": [],'
            ' "answer_type": "essay"}'
        )
        result = parse_structured(text, self.QUESTION)
        assert result.answer_type == "short_phrase"


class TestDeterminism:
    def test_identical_calls_identical_output(self):
        structurer = _structurer()
        first_sq, first_resp = structurer.structure(MULTI_HOP)
        second_sq, second_resp = structurer.structure(MULTI_HOP)
        assert first_sq == second_sq
        assert first_resp.text == second_resp.text

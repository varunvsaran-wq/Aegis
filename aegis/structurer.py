"""Phase-2 prompt structurer: decompose a question into a StructuredQuery.

The structurer makes one LLM call per question (via :class:`~aegis.gateway.
ModelClient`) asking the model to decompose the question into atomic
sub-questions, extract retrieval keywords, and classify the answer type. The
model's reply is parsed tolerantly by :func:`parse_structured`, which never
raises: any malformed output degrades to a safe fallback that keeps the
pipeline running with the original question as the only sub-question.
"""

import json

from aegis.types import LLMResponse, StructuredQuery

#: Prompt templates used by the structurer, keyed by task name.
STRUCTURER_TEMPLATES: dict[str, str] = {
    "structure": """\
You are a query analyst for a retrieval-augmented QA pipeline. Analyze the
question below and decompose it for document retrieval.

Rules:
- intent: one short sentence stating what the asker wants to know.
- sub_questions: decompose multi-hop questions into 1-3 atomic sub-questions,
  each answerable by a single document. If the question is single-hop,
  sub_questions is just the question itself.
- keywords: extract 3-8 retrieval keywords (entities, dates, distinctive
  nouns) from the question.
- answer_type: classify the expected answer as one of
  short_phrase|yes_no|entity|number|date.

Return ONLY a JSON object with exactly these keys: "intent",
"sub_questions", "keywords", "answer_type". No prose, no code fences.

Question: {question}
""",
}

_ALLOWED_ANSWER_TYPES = frozenset(
    {"short_phrase", "yes_no", "entity", "number", "date"}
)
_MAX_SUB_QUESTIONS = 4


def _extract_json_object(text: str) -> dict:
    """Return the first top-level JSON object embedded in ``text``.

    Tolerates surrounding prose and ```json fences: everything before the
    first ``{`` and after its matching ``}`` is ignored (``raw_decode`` stops
    at the matching brace). Raises on any failure; callers handle fallback.
    """
    cleaned = text.replace("```json", "```").replace("```", " ")
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def _coerce_str_list(value: object) -> list[str]:
    """Coerce a JSON value into a list of non-empty strings."""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()]


def parse_structured(text: str, question: str) -> StructuredQuery:
    """Parse a model reply into a StructuredQuery. Never raises.

    Unknown keys are dropped, strings are coerced to lists where sensible,
    sub_questions is clamped to at most 4 entries (with the original question
    substituted if it comes back empty), and an unrecognized answer_type
    degrades to "short_phrase". ANY failure yields the safe fallback:
    ``StructuredQuery(intent="", sub_questions=[question], keywords=[],
    answer_type="short_phrase")``.
    """
    try:
        data = _extract_json_object(text)

        intent_raw = data.get("intent", "")
        intent = intent_raw.strip() if isinstance(intent_raw, str) else ""

        sub_questions = _coerce_str_list(data.get("sub_questions"))[
            :_MAX_SUB_QUESTIONS
        ]
        if not sub_questions:
            sub_questions = [question]

        keywords = _coerce_str_list(data.get("keywords"))

        answer_type_raw = data.get("answer_type", "")
        answer_type = (
            answer_type_raw.strip().lower()
            if isinstance(answer_type_raw, str)
            else ""
        )
        if answer_type not in _ALLOWED_ANSWER_TYPES:
            answer_type = "short_phrase"

        return StructuredQuery(
            intent=intent,
            sub_questions=sub_questions,
            keywords=keywords,
            answer_type=answer_type,
        )
    except Exception:
        return StructuredQuery(
            intent="",
            sub_questions=[question],
            keywords=[],
            answer_type="short_phrase",
        )


class QueryStructurer:
    """Turns raw questions into StructuredQuery objects via one LLM call."""

    def __init__(self, client):
        self.client = client
        """A :class:`~aegis.gateway.ModelClient` used for the structuring call."""

    def structure(self, question: str) -> tuple[StructuredQuery, LLMResponse]:
        """Structure ``question`` with one client.complete call.

        Returns both the parsed :class:`StructuredQuery` and the raw
        :class:`LLMResponse` so callers can account for tokens and cost.
        """
        prompt = STRUCTURER_TEMPLATES["structure"].format(question=question)
        response = self.client.complete([{"role": "user", "content": prompt}])
        return parse_structured(response.text, question), response

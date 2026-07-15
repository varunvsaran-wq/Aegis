"""Unit tests for aegis.gateway (mock model only — no network, no litellm)."""

from aegis.gateway import MockLLM, ModelClient, get_client
from aegis.types import LLMResponse


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


CITATION_PROMPT = (
    "Question: Who?\n"
    "[[chunk:A::0]]\n"
    "Alpha beta. Gamma.\n"
    "[[chunk:B::1]]\n"
    "Delta.\n"
)


class TestGetClient:
    def test_mock_alias_resolves_to_mock_path(self):
        client = get_client("mock")
        assert isinstance(client, ModelClient)
        assert client.model == "mock"
        assert client.resolved_model == "mock/echo"
        assert client.resolved_model.startswith("mock/")

    def test_complete_returns_populated_llmresponse(self):
        client = get_client("mock")
        response = client.complete(_msgs("Hello there, mock model."))
        assert isinstance(response, LLMResponse)
        assert response.text
        assert response.model == "mock"
        assert response.model_version == "mock/echo"
        assert response.cost_usd == 0.0
        assert response.tokens_in > 0
        assert response.tokens_out > 0
        assert response.latency_s >= 0

    def test_raw_model_string_construction(self):
        # Construction only — never calls the network / litellm.
        client = get_client("openai/gpt-4o-mini", temperature=0.5, seed=42)
        assert client.model == "openai/gpt-4o-mini"
        assert client.resolved_model == "openai/gpt-4o-mini"
        assert client.temperature == 0.5
        assert client.seed == 42


class TestMockCitationRule:
    def test_answers_first_sentence_and_cites_first_chunk(self):
        response = get_client("mock").complete(_msgs(CITATION_PROMPT))
        assert response.text.startswith("ANSWER: Alpha beta")
        assert "CITATIONS: A::0" in response.text

    def test_second_chunk_is_not_cited(self):
        response = get_client("mock").complete(_msgs(CITATION_PROMPT))
        assert "B::1" not in response.text.split("CITATIONS:")[1]


class TestMockJudgeRule:
    def test_case_insensitive_match_is_correct(self):
        prompt = "Gold answer: Paris\nCandidate answer: paris\n"
        response = get_client("mock").complete(_msgs(prompt))
        assert response.text == "CORRECT"

    def test_mismatch_is_incorrect(self):
        prompt = "Gold answer: Paris\nCandidate answer: London\n"
        response = get_client("mock").complete(_msgs(prompt))
        assert response.text == "INCORRECT"

    def test_empty_candidate_is_incorrect(self):
        prompt = "Gold answer: Paris\nCandidate answer:\n"
        assert MockLLM().complete(prompt) == "INCORRECT"


class TestMockFallback:
    def test_plain_prompt_returns_i_dont_know(self):
        response = get_client("mock").complete(_msgs("What is the meaning of life?"))
        assert response.text == "I don't know."


class TestDeterminism:
    def test_identical_calls_return_identical_text(self):
        client = get_client("mock")
        first = client.complete(_msgs(CITATION_PROMPT))
        second = client.complete(_msgs(CITATION_PROMPT))
        assert first.text == second.text

    def test_mockllm_is_deterministic_across_instances(self):
        prompt = "Gold answer: Rome\nCandidate answer: rome, italy\n"
        assert MockLLM().complete(prompt) == MockLLM().complete(prompt) == "CORRECT"

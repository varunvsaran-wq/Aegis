"""Model gateway: a uniform completion interface over litellm plus a mock.

This module exposes :class:`ModelClient`, which turns OpenAI-style message
lists into :class:`~aegis.types.LLMResponse` objects. Models are addressed by
registry alias (see :data:`aegis.config.MODEL_REGISTRY`) or by raw litellm
model string. Any model whose resolved string starts with ``"mock/"`` is
served by the built-in, deterministic :class:`MockLLM` with no network access
and no litellm import, which keeps tests and smoke runs hermetic.
"""

import json
import re
import time

from aegis.config import resolve_model
from aegis.types import LLMResponse

_CHUNK_MARKER_RE = re.compile(r"\[\[chunk:([^\]]+)\]\]")
_MOCK_MODEL_VERSION = "mock/echo"

_STRUCTURER_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STRUCTURER_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "was", "were", "is", "are",
        "what", "which", "who", "when", "where", "did", "does", "and", "that",
    }
)  # fmt: skip
_YES_NO_STARTERS = frozenset(
    {"is", "are", "was", "were", "did", "does", "do", "can", "has", "have"}
)


class MockLLM:
    """Deterministic, zero-network fake LLM.

    Given the full prompt text (all message contents concatenated), it applies
    four rules in order:

    1. **Citation rule** — if the prompt contains ``[[chunk:<id>]]`` markers,
       answer with the first sentence of the first chunk's text and cite that
       chunk's id.
    2. **Structurer rule** — if the prompt contains both
       ``Return ONLY a JSON object`` and ``sub_questions``, return a
       deterministic JSON decomposition of the text after the last line
       starting with ``Question:`` (split on the first ``" and "`` into two
       sub-questions, keyword extraction with a tiny stopword list, and a
       yes/no vs. short-phrase answer-type heuristic).
    3. **Judge rule** — if the prompt contains both a ``Gold answer:`` line
       and a ``Candidate answer:`` line, return ``CORRECT`` when one
       normalized answer is a substring of the other (and the candidate is
       non-empty), else ``INCORRECT``.
    4. **Fallback** — return ``I don't know.``

    Identical input always produces identical output: there is no randomness
    and no time-dependent behavior.
    """

    def complete(self, prompt: str) -> str:
        """Return the deterministic mock completion for ``prompt``."""
        citation = self._citation_rule(prompt)
        if citation is not None:
            return citation
        structured = self._structurer_rule(prompt)
        if structured is not None:
            return structured
        judged = self._judge_rule(prompt)
        if judged is not None:
            return judged
        return "I don't know."

    @staticmethod
    def _citation_rule(prompt: str) -> str | None:
        """Answer from ``[[chunk:<id>]]``-marked context, if any is present.

        Each marker is followed by that chunk's text on subsequent lines
        until the next marker (or end of prompt). The answer is the first
        sentence (up to the first period) of the first chunk's text, and the
        citation is the first chunk's id. A ``Question:`` line, if present,
        is parsed but does not alter the deterministic output.
        """
        markers = list(_CHUNK_MARKER_RE.finditer(prompt))
        if not markers:
            return None

        first = markers[0]
        end = markers[1].start() if len(markers) > 1 else len(prompt)
        chunk_text = prompt[first.end() : end].strip()
        first_sentence = chunk_text.split(".", 1)[0].strip()

        # The question is extracted for fidelity with real prompts, though
        # the mock's answer depends only on the retrieved context.
        _question = MockLLM._line_value(prompt, "Question:")

        return f"ANSWER: {first_sentence}\nCITATIONS: {first.group(1)}"

    @staticmethod
    def _structurer_rule(prompt: str) -> str | None:
        """Emit a deterministic JSON decomposition for structurer prompts.

        Fires only when the prompt contains BOTH ``Return ONLY a JSON
        object`` and ``sub_questions``. The question is the text after the
        last line starting with ``Question:``. Output is ``json.dumps(...,
        sort_keys=True)`` with:

        - ``intent``: always ``"answer the question"``.
        - ``sub_questions``: if the literal ``" and "`` appears, split on
          the FIRST occurrence into two sub-questions (stripped, each
          ensured to end with ``?``); otherwise just the question.
        - ``keywords``: lowercase alphanumeric tokens longer than 3 chars,
          stopwords removed, first 6, in question order.
        - ``answer_type``: ``"yes_no"`` if the lowercased question starts
          with a yes/no auxiliary (is, are, was, were, did, does, do, can,
          has, have), else ``"short_phrase"``.
        """
        if "Return ONLY a JSON object" not in prompt or "sub_questions" not in prompt:
            return None

        question = ""
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Question:"):
                question = stripped[len("Question:") :].strip()

        if " and " in question:
            parts = question.split(" and ", 1)
            sub_questions = []
            for part in parts:
                part = part.strip()
                if not part.endswith("?"):
                    part += "?"
                sub_questions.append(part)
        else:
            sub_questions = [question]

        tokens = _STRUCTURER_TOKEN_RE.findall(question.lower())
        keywords = [
            t for t in tokens if len(t) > 3 and t not in _STRUCTURER_STOPWORDS
        ][:6]

        first_word = tokens[0] if tokens else ""
        answer_type = "yes_no" if first_word in _YES_NO_STARTERS else "short_phrase"

        return json.dumps(
            {
                "intent": "answer the question",
                "sub_questions": sub_questions,
                "keywords": keywords,
                "answer_type": answer_type,
            },
            sort_keys=True,
        )

    @staticmethod
    def _judge_rule(prompt: str) -> str | None:
        """Grade a gold/candidate answer pair, if the prompt contains one."""
        gold = MockLLM._line_value(prompt, "Gold answer:")
        candidate = MockLLM._line_value(prompt, "Candidate answer:")
        if gold is None or candidate is None:
            return None

        gold_norm = gold.strip().lower()
        cand_norm = candidate.strip().lower()
        if cand_norm and (cand_norm in gold_norm or gold_norm in cand_norm):
            return "CORRECT"
        return "INCORRECT"

    @staticmethod
    def _line_value(prompt: str, prefix: str) -> str | None:
        """Return the value after ``prefix`` on the first line starting with it."""
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped[len(prefix) :].strip()
        return None


class ModelClient:
    """A completion client bound to one model, temperature, and seed.

    ``model`` may be a registry alias (e.g. ``"mock"``, ``"small"``) or a raw
    litellm model string (e.g. ``"openai/gpt-4o-mini"``). Both the alias as
    passed and the resolved litellm string are stored; responses report the
    alias in ``LLMResponse.model`` and the provider's exact model string in
    ``LLMResponse.model_version``.
    """

    def __init__(self, model: str, temperature: float = 0.0, seed: int | None = None):
        self.model = model
        """The alias (or raw string) as passed by the caller."""

        self.resolved_model = resolve_model(model)
        """The litellm model string this client actually calls."""

        self.temperature = temperature
        self.seed = seed
        self._mock = MockLLM()

    def complete(self, messages: list[dict], max_tokens: int = 1024) -> LLMResponse:
        """Run one chat completion and return a fully populated LLMResponse.

        ``messages`` are OpenAI-style dicts:
        ``[{"role": "system"|"user"|"assistant", "content": str}]``.

        Mock models (resolved string starting with ``"mock/"``) are served
        in-process with no network, no retries, and no litellm import. Real
        models go through litellm with tenacity retries (3 attempts,
        exponential backoff 1-8s) on any exception.
        """
        if self.resolved_model.startswith("mock/"):
            return self._complete_mock(messages)
        return self._complete_litellm(messages, max_tokens)

    def _complete_mock(self, messages: list[dict]) -> LLMResponse:
        """Serve the completion from the built-in MockLLM (no retries)."""
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        start = time.perf_counter()
        text = self._mock.complete(prompt)
        latency = time.perf_counter() - start
        return LLMResponse(
            text=text,
            model=self.model,
            model_version=_MOCK_MODEL_VERSION,
            tokens_in=len(prompt.split()),
            tokens_out=len(text.split()),
            cost_usd=0.0,
            latency_s=latency,
        )

    def _complete_litellm(self, messages: list[dict], max_tokens: int) -> LLMResponse:
        """Call litellm with retries and translate the raw response.

        litellm and tenacity are imported lazily so this module (and the mock
        path) works even when those packages are not installed.
        """
        import litellm
        from tenacity import (
            Retrying,
            retry_if_exception_type,
            retry_if_not_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        # Errors that will fail identically on every attempt: retrying them only
        # adds backoff delay before the same failure.
        permanent = tuple(
            getattr(litellm.exceptions, name)
            for name in (
                "UnsupportedParamsError",
                "AuthenticationError",
                "BadRequestError",
                "NotFoundError",
                "PermissionDeniedError",
            )
            if hasattr(litellm.exceptions, name)
        )
        retryer = Retrying(
            retry=retry_if_exception_type(Exception)
            & retry_if_not_exception_type(permanent),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )

        start = time.perf_counter()
        response = retryer(
            litellm.completion,
            model=self.resolved_model,
            messages=messages,
            temperature=self.temperature,
            # Not every provider takes a seed (Anthropic rejects it). The seed is
            # still logged to MLflow; drop_params sends it only where supported.
            seed=self.seed,
            drop_params=True,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start

        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:
            # Local, ollama, and mock-style models have no published price.
            cost = 0.0

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            model_version=getattr(response, "model", None) or self.resolved_model,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
            latency_s=latency,
        )


def get_client(
    model: str, temperature: float = 0.0, seed: int | None = None
) -> ModelClient:
    """Factory for :class:`ModelClient`; accepts an alias or raw model string."""
    return ModelClient(model, temperature=temperature, seed=seed)

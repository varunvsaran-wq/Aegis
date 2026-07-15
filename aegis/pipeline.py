"""RAG pipeline: prompt construction, contract parsing, and enforcement.

Runs a :class:`~aegis.types.HotpotQuestion` through one of three modes:

- ``closed_book`` — no retrieval; the model answers from parametric knowledge.
- ``vanilla_rag`` — retrieved context is provided, but no citation contract.
- ``raw_rag`` — retrieved context plus a strict two-line answer/citation
  contract, with one enforcement retry and abstention on repeated violation.

All prompt strings live in :data:`PROMPT_TEMPLATES` so experiment tracking can
hash them for reproducibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from aegis.types import Answer, Citation, HotpotQuestion, QueryResult, RetrievalResult

if TYPE_CHECKING:
    from aegis.gateway import ModelClient
    from aegis.retrieve import HybridRetriever

#: Every prompt template used by the pipeline, keyed by role. Hashed by the
#: experiment tracker for MLflow reproducibility logging — do not construct
#: prompt text outside this dict.
PROMPT_TEMPLATES: dict[str, str] = {
    "closed_book": (
        "Answer the following question concisely using only your own knowledge. "
        "Reply with just the short answer.\n\n"
        "Question: {question}"
    ),
    "vanilla_rag": (
        "{context}\n\n"
        "Question: {question}\n"
        "Answer concisely based on the context."
    ),
    "raw_rag_system": (
        "You are a careful question-answering assistant. You must answer ONLY "
        "from the provided context; every answer must cite the chunk ids of "
        "the evidence you used. Output EXACTLY two lines:\n"
        "ANSWER: <short answer>\n"
        "CITATIONS: <comma-separated chunk ids>\n"
        "If the context does not contain the answer, output ANSWER: unknown "
        "with best-effort citations."
    ),
    "raw_rag_user": "{context}\n\nQuestion: {question}",
    "citation_retry": (
        "Your previous response did not follow the required format or did not "
        "cite any valid chunk ids from the provided context. Restate your "
        "answer, outputting EXACTLY two lines:\n"
        "ANSWER: <short answer>\n"
        "CITATIONS: <comma-separated chunk ids copied exactly from the "
        "[[chunk:<id>]] markers in the context>"
    ),
}

MODES = ("closed_book", "vanilla_rag", "raw_rag")


def format_context(retrieved: list[RetrievalResult]) -> str:
    """Render retrieved chunks as ``[[chunk:<id>]]`` blocks joined by blank lines.

    This exact marker syntax is a contract with the deterministic MockLLM and
    with the citation validator — do not change it.
    """
    blocks = [f"[[chunk:{r.chunk.id}]]\n{r.chunk.text}" for r in retrieved]
    return "\n\n".join(blocks)


def parse_contract(
    text: str, valid_ids: Iterable[str] | None = None
) -> tuple[str, list[str]]:
    """Tolerantly parse an ``ANSWER:`` / ``CITATIONS:`` contract response.

    Returns ``(answer, citation_ids)``. The first line starting with
    ``ANSWER:`` (case-insensitive, leading whitespace ignored) supplies the
    answer; the first line starting with ``CITATIONS:`` is split on commas
    with brackets, whitespace, and any ``chunk:`` prefix stripped from each
    id. If ``valid_ids`` is given, ids not exactly matching it (hallucinated
    citations) are dropped. Missing lines yield ``""`` / ``[]``.
    """
    answer: str | None = None
    citations: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if answer is None and lowered.startswith("answer:"):
            answer = stripped[len("answer:") :].strip()
        elif not citations and lowered.startswith("citations:"):
            raw = stripped[len("citations:") :]
            for token in raw.split(","):
                cid = token.strip().strip("[]").strip()
                if cid.lower().startswith("chunk:"):
                    cid = cid[len("chunk:") :].strip()
                if cid:
                    citations.append(cid)

    if valid_ids is not None:
        valid = set(valid_ids)
        citations = [c for c in citations if c in valid]

    # De-duplicate while preserving order.
    seen: set[str] = set()
    citations = [c for c in citations if not (c in seen or seen.add(c))]

    return (answer or "", citations)


class RAGPipeline:
    """Runs questions through the configured answering mode.

    ``client`` must provide ``.complete(messages, max_tokens=...) ->
    LLMResponse`` (see :class:`aegis.gateway.ModelClient`); ``retriever`` must
    provide ``.retrieve(query, ..., k_final=...) -> list[RetrievalResult]``
    and is required for the two RAG modes.
    """

    def __init__(
        self,
        client: ModelClient,
        retriever: HybridRetriever | None = None,
        mode: str = "raw_rag",
        k_final: int = 5,
    ):
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        if mode in ("vanilla_rag", "raw_rag"):
            assert retriever is not None, f"mode {mode!r} requires a retriever"
        self.client = client
        self.retriever = retriever
        self.mode = mode
        self.k_final = k_final

    def run(self, q: HotpotQuestion) -> QueryResult:
        """Answer one question, returning the full :class:`QueryResult` record."""
        if self.mode == "closed_book":
            answer, retrieved, responses = self._run_closed_book(q)
        elif self.mode == "vanilla_rag":
            answer, retrieved, responses = self._run_vanilla_rag(q)
        else:
            answer, retrieved, responses = self._run_raw_rag(q)

        return QueryResult(
            question_id=q.id,
            question=q.question,
            gold_answer=q.answer,
            answer=answer,
            retrieved=retrieved,
            mode=self.mode,
            model=self.client.model,
            cost_usd=sum(r.cost_usd for r in responses),
            latency_s=sum(r.latency_s for r in responses),
            tokens_in=sum(r.tokens_in for r in responses),
            tokens_out=sum(r.tokens_out for r in responses),
        )

    # ------------------------------------------------------------------ modes

    def _run_closed_book(self, q: HotpotQuestion):
        prompt = PROMPT_TEMPLATES["closed_book"].format(question=q.question)
        resp = self.client.complete([{"role": "user", "content": prompt}])
        answer = Answer(text=resp.text.strip(), citations=[], raw_response=resp.text)
        return answer, [], [resp]

    def _run_vanilla_rag(self, q: HotpotQuestion):
        retrieved = self.retriever.retrieve(q.question, k_final=self.k_final)
        prompt = PROMPT_TEMPLATES["vanilla_rag"].format(
            context=format_context(retrieved), question=q.question
        )
        resp = self.client.complete([{"role": "user", "content": prompt}])
        answer = Answer(text=resp.text.strip(), citations=[], raw_response=resp.text)
        return answer, retrieved, [resp]

    def _run_raw_rag(self, q: HotpotQuestion):
        retrieved = self.retriever.retrieve(q.question, k_final=self.k_final)
        valid_ids = {r.chunk.id for r in retrieved}
        user_prompt = PROMPT_TEMPLATES["raw_rag_user"].format(
            context=format_context(retrieved), question=q.question
        )
        messages = [
            {"role": "system", "content": PROMPT_TEMPLATES["raw_rag_system"]},
            {"role": "user", "content": user_prompt},
        ]

        responses = [self.client.complete(messages)]
        answer_text, citation_ids = parse_contract(responses[-1].text, valid_ids)

        if not citation_ids:
            # One enforcement retry, restating the contract.
            retry_messages = messages + [
                {"role": "assistant", "content": responses[-1].text},
                {"role": "user", "content": PROMPT_TEMPLATES["citation_retry"]},
            ]
            responses.append(self.client.complete(retry_messages))
            answer_text, citation_ids = parse_contract(responses[-1].text, valid_ids)

        if citation_ids:
            answer = Answer(
                text=answer_text,
                citations=[Citation(chunk_id=c) for c in citation_ids],
                abstained=False,
                raw_response=responses[-1].text,
            )
        else:
            answer = Answer(
                text=answer_text or "",
                citations=[],
                abstained=True,
                raw_response=responses[-1].text,
            )
        return answer, retrieved, responses

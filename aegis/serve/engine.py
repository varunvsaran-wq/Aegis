"""A small ad-hoc query engine shared by the API and the demo.

This wraps ingestion + retrieval + either the raw or harnessed pipeline so a
single free-text question and a corpus of documents can be answered on the fly
(outside the HotpotQA eval harness). It is deliberately lightweight: documents
are paragraph-chunked, indexed with the same hybrid retriever used in
evaluation, and answered through :class:`~aegis.pipeline.RAGPipeline` (harness
off) or :class:`~aegis.graph.HarnessedPipeline` (harness on) so the demo's
on/off toggle exercises the exact code paths measured in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegis.types import Answer, Chunk, HotpotQuestion


@dataclass
class EngineResult:
    """One answered query, flattened for JSON/UI rendering."""

    answer: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    harnessed: bool = False
    grounded: bool | None = None
    agreement: float | None = None
    blocked: bool = False
    sanitized_chunks: int = 0
    retrieved: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    llm_calls: int = 0


def _chunk_documents(documents: list[str]) -> list[Chunk]:
    """Paragraph-chunk raw documents into the corpus Chunk contract."""
    import re

    chunks: list[Chunk] = []
    for d, doc in enumerate(documents):
        title = f"doc{d}"
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]
        if not paragraphs:
            paragraphs = [doc.strip()] if doc.strip() else []
        for p, para in enumerate(paragraphs):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            chunks.append(
                Chunk(
                    id=f"{title}::{len(chunks)}",
                    title=title,
                    text=para,
                    sentences=sentences or [para],
                    metadata={"poisoned": False},
                )
            )
    return chunks


class AegisEngine:
    """Answers ad-hoc questions over an in-memory corpus, harness on or off.

    ``model`` is any registry alias or litellm string (default ``"mock"`` so
    the API/demo boot with zero credentials). Call :meth:`index` once with the
    documents, then :meth:`answer` per question.
    """

    def __init__(self, model: str = "mock", k_final: int = 5):
        self.model = model
        self.k_final = k_final
        self._retriever = None
        self._verifier = None
        self._indexed: tuple[str, ...] | None = None

    def index(self, documents: list[str]) -> int:
        """(Re)build the retrieval index over ``documents``; returns #chunks."""
        from aegis.retrieve import build_retriever

        chunks = _chunk_documents(documents)
        self._retriever = build_retriever(chunks) if chunks else None
        self._indexed = tuple(documents)
        return len(chunks)

    def ensure_indexed(self, documents: list[str]) -> None:
        """Index ``documents`` unless they are exactly what is already indexed."""
        if self._indexed != tuple(documents):
            self.index(documents)

    def _get_verifier(self):
        """Build the NLI verifier once; reloading it per question is slow."""
        if self._verifier is None:
            from aegis.config import get_config
            from aegis.verify import GroundednessVerifier, get_nli

            self._verifier = GroundednessVerifier(get_nli(get_config()))
        return self._verifier

    def compare(self, question: str) -> tuple[EngineResult, EngineResult]:
        """Answer ``question`` with the harness off and on: ``(off, on)``."""
        return self.answer(question, harness=False), self.answer(question, harness=True)

    def answer(self, question: str, harness: bool = True) -> EngineResult:
        """Answer ``question``; ``harness`` toggles the full reliability stack.

        Harness off is plain ``vanilla_rag`` (retrieve + generate, no citation
        contract), which is the baseline the paper and portfolio compare against.
        """
        from aegis.gateway import get_client

        q = HotpotQuestion(
            id="live",
            question=question,
            answer="",
            type="bridge",
            level="hard",
            supporting_facts=[],
            context=[],
        )
        client = get_client(self.model, temperature=0.0, seed=0)

        if harness:
            from aegis.defense import make_canary
            from aegis.graph import HarnessedPipeline

            pipeline = HarnessedPipeline(
                client,
                self._retriever,
                k_final=self.k_final,
                use_defense=True,
                canary=make_canary("live"),
                verifier=self._get_verifier(),
            )
        else:
            from aegis.pipeline import RAGPipeline

            pipeline = RAGPipeline(
                client, retriever=self._retriever, mode="vanilla_rag", k_final=self.k_final
            )

        result = pipeline.run(q)
        return _to_engine_result(result, harnessed=harness)


def _to_engine_result(result, harnessed: bool) -> EngineResult:
    answer: Answer = result.answer
    harness = result.harness
    return EngineResult(
        answer=answer.text,
        citations=[c.chunk_id for c in answer.citations],
        abstained=answer.abstained,
        harnessed=harnessed,
        grounded=(harness.grounded if harness else None),
        agreement=(harness.agreement if harness else None),
        blocked=(harness.blocked if harness else False),
        sanitized_chunks=(harness.sanitized_chunks if harness else 0),
        retrieved=[
            {
                "chunk_id": r.chunk.id,
                "title": r.chunk.title,
                "text": r.chunk.text[:400],
                "rerank_score": r.rerank_score,
            }
            for r in result.retrieved
        ],
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
        llm_calls=(harness.llm_calls if harness else 1),
    )

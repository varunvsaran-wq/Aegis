"""Core data contracts shared across all Aegis modules.

These pydantic models are the interfaces that every stage of the pipeline
(ingest, retrieval, gateway, evaluation) builds against. Field names are
frozen for Phase 1 — do not rename fields or add required fields, as other
modules are developed in parallel against these exact contracts.
"""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single paragraph-level unit of the retrieval corpus."""

    id: str
    """Unique within corpus, formatted as "<title>::<para_idx>"."""

    title: str
    """Source Wikipedia article title."""

    text: str
    """Full paragraph text."""

    sentences: list[str]
    """The paragraph's sentences, in order."""

    metadata: dict = Field(default_factory=dict)
    """Arbitrary chunk metadata, e.g. {"poisoned": False}."""


class RetrievalResult(BaseModel):
    """One retrieved chunk together with the scores it accumulated."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class LLMResponse(BaseModel):
    """A raw completion from the model gateway, with usage accounting."""

    text: str
    model: str
    """Registry alias, e.g. "mock"."""

    model_version: str
    """Exact provider model string."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class Citation(BaseModel):
    """A reference to a corpus chunk supporting part of an answer."""

    chunk_id: str


class Answer(BaseModel):
    """A parsed, structured answer produced by the pipeline."""

    text: str
    """Final short answer."""

    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    raw_response: str = ""


class StructuredQuery(BaseModel):
    """A question decomposed by the prompt structurer."""

    intent: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    answer_type: str = "short_phrase"
    """One of "short_phrase" | "yes_no" | "entity" | "number" | "date"."""


class VerifierReport(BaseModel):
    """One NLI verifier judgment of a claim against a retrieved chunk."""

    claim: str
    chunk_id: str
    label: str
    """One of "entailment" | "neutral" | "contradiction"."""

    score: float
    """Probability of the winning label."""


class HarnessMeta(BaseModel):
    """Phase 2 harness diagnostics attached to a :class:`QueryResult`."""

    structured: StructuredQuery | None = None
    verifier_reports: list[VerifierReport] = Field(default_factory=list)
    verify_retries: int = 0
    grounded: bool | None = None
    """Final verifier verdict for the returned answer."""

    votes: list[str] = Field(default_factory=list)
    """The k sampled answer strings."""

    agreement: float | None = None
    """Majority cluster size / k."""

    llm_calls: int = 0
    """Total generation+structurer calls made."""

    # -- Phase 3 injection-defense diagnostics (populated when use_defense) --
    direct_injection: bool | None = None
    """Direct-gate verdict on the user query (None when the gate is off)."""

    direct_injection_score: float | None = None
    sanitized_chunks: int = 0
    """Number of retrieved chunks from which an injected span was stripped."""

    canary_leaked: bool = False
    """True iff the planted canary token appeared in the model output."""

    blocked: bool = False
    """True iff the defense forced an abstention (direct hit or canary leak)."""


class QueryResult(BaseModel):
    """The full record of running one question through the pipeline."""

    question_id: str
    question: str
    gold_answer: str
    answer: Answer
    retrieved: list[RetrievalResult] = Field(default_factory=list)
    mode: str
    """One of "closed_book" | "vanilla_rag" | "raw_rag"."""

    model: str
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    harness: HarnessMeta | None = None


class HotpotQuestion(BaseModel):
    """A single HotpotQA example in normalized form."""

    id: str
    question: str
    answer: str
    type: str
    """One of "bridge" | "comparison"."""

    level: str
    supporting_facts: list[tuple[str, int]]
    """(title, sent_idx) pairs identifying gold supporting sentences."""

    context: list[tuple[str, list[str]]]
    """(title, sentences) pairs — the paragraphs provided with the question."""

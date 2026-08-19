"""Phase-2 harnessed pipeline as a compiled LangGraph state machine.

Flow: structurer -> multi-query retrieval -> citation-contract generation
(optionally k-vote sampled) -> NLI groundedness verification -> one retry on
verification failure -> abstain. The graph is compiled once in
:class:`HarnessedPipeline.__init__` and re-used for every question.

The citation-contract prompts and parser are reused from
:mod:`aegis.pipeline` (``PROMPT_TEMPLATES`` / ``format_context`` /
``parse_contract``); only harness-specific prompt text lives here, in
:data:`HARNESS_TEMPLATES`, so the eval driver can hash both dicts for
reproducibility.

The sibling modules :mod:`aegis.structurer` and :mod:`aegis.verify` are
imported lazily (inside methods) so this module imports standalone while they
are developed in parallel; tests inject doubles for both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

from aegis.pipeline import PROMPT_TEMPLATES, format_context, parse_contract
from aegis.types import (
    Answer,
    Citation,
    HarnessMeta,
    HotpotQuestion,
    LLMResponse,
    QueryResult,
    RetrievalResult,
    StructuredQuery,
    VerifierReport,
)

if TYPE_CHECKING:
    from aegis.gateway import ModelClient

#: Harness-specific prompt templates, keyed by task. Hashed by the experiment
#: tracker for reproducibility — do not construct harness prompt text outside
#: this dict.
HARNESS_TEMPLATES: dict[str, str] = {
    "verify_retry": (
        "Your previous answer was NOT supported by the cited evidence. A "
        "fact-checking verifier compared your answer against the retrieved "
        "chunks and found:\n"
        "{feedback}\n\n"
        "Re-answer the question using ONLY facts stated in the provided "
        "context, citing the chunk ids that directly support your answer. "
        "Output EXACTLY two lines:\n"
        "ANSWER: <short answer>\n"
        "CITATIONS: <comma-separated chunk ids copied exactly from the "
        "[[chunk:<id>]] markers in the context>\n"
        "If the context does not support any answer, output ANSWER: unknown "
        "with best-effort citations."
    ),
}


class HarnessState(TypedDict, total=False):
    """LangGraph state threaded through the harness nodes.

    Pydantic objects are stored as-is; every node returns a partial update.
    """

    question: HotpotQuestion
    structured: StructuredQuery
    retrieved: list[RetrievalResult]
    gen_messages: list[dict]
    """The original system+user generation messages, kept for the verify retry."""

    answer: Answer
    responses: list[LLMResponse]
    """Every completion made so far (structurer + samples + retries)."""

    votes: list[str]
    agreement: float | None
    verifier_reports: list[VerifierReport]
    grounded: bool | None
    verify_retries: int
    result: QueryResult

    # Phase 3 defense diagnostics.
    direct_injection: bool | None
    direct_injection_score: float | None
    sanitized_chunks: int
    canary_leaked: bool
    blocked: bool


def _rerank_key(r: RetrievalResult) -> float:
    """Sort/compare key for merged retrieval results (None scores sink)."""
    return r.rerank_score if r.rerank_score is not None else float("-inf")


def _fallback_majority_vote(answers: list[str]) -> tuple[str, float, list[str]]:
    """Local majority vote used only when :mod:`aegis.verify` is unavailable.

    Clusters answers by case-insensitive stripped equality and returns
    ``(majority_answer, cluster_size / k, answers)`` — the same shape as
    ``aegis.verify.majority_vote``. Ties break toward the earliest cluster.
    """
    votes = list(answers)
    if not votes:
        return "", 0.0, []
    clusters: dict[str, list[int]] = {}
    for i, ans in enumerate(votes):
        clusters.setdefault(ans.strip().lower(), []).append(i)
    winner = max(clusters.values(), key=lambda members: (len(members), -members[0]))
    return votes[winner[0]], len(winner) / len(votes), votes


class HarnessedPipeline:
    """The full Phase-2 harness, compiled once as a LangGraph StateGraph.

    ``client`` is the :class:`~aegis.gateway.ModelClient` used for
    structuring, generation, and the verify retry. ``vote_client`` (typically
    a higher-temperature client) is used for sampling when ``k_vote > 1`` and
    falls back to ``client``. ``structurer`` / ``verifier`` are injectable
    for tests and are otherwise default-constructed lazily from the real
    :mod:`aegis.structurer` / :mod:`aegis.verify` modules on first use.
    """

    def __init__(
        self,
        client: ModelClient,
        retriever,
        k_final: int = 5,
        k_vote: int = 1,
        use_structurer: bool = True,
        use_verifier: bool = True,
        use_defense: bool = False,
        canary: str | None = None,
        vote_client=None,
        structurer=None,
        verifier=None,
        direct_gate=None,
        indirect_defense=None,
    ):
        self.client = client
        self.retriever = retriever
        self.k_final = k_final
        self.k_vote = k_vote
        self.use_structurer = use_structurer
        self.use_verifier = use_verifier
        self.use_defense = use_defense
        self.canary = canary
        self.vote_client = vote_client
        self._structurer = structurer
        self._verifier = verifier
        self._direct_gate = direct_gate
        self._indirect_defense = indirect_defense
        self._graph = self._build_graph()

    def run(self, q: HotpotQuestion) -> QueryResult:
        """Answer one question through the compiled harness graph."""
        final: HarnessState = self._graph.invoke({"question": q})
        return final["result"]

    # ------------------------------------------------------------ graph build

    def _build_graph(self):
        graph = StateGraph(HarnessState)
        graph.add_node("structure", self._node_structure)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("generate", self._node_generate)
        graph.add_node("verify", self._node_verify)
        graph.add_node("generate_retry", self._node_generate_retry)
        graph.add_node("abstain", self._node_abstain)
        graph.add_node("finalize", self._node_finalize)

        graph.add_edge(START, "structure")
        graph.add_edge("structure", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_conditional_edges(
            "generate",
            self._route_after_generate,
            {"verify": "verify", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {
                "finalize": "finalize",
                "generate_retry": "generate_retry",
                "abstain": "abstain",
            },
        )
        graph.add_edge("generate_retry", "verify")
        graph.add_edge("abstain", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # -------------------------------------------------------- lazy components

    def _get_structurer(self):
        if self._structurer is None:
            from aegis.structurer import QueryStructurer

            self._structurer = QueryStructurer(self.client)
        return self._structurer

    def _get_verifier(self):
        if self._verifier is None:
            from aegis.config import get_config
            from aegis.verify import GroundednessVerifier, get_nli

            self._verifier = GroundednessVerifier(get_nli(get_config()))
        return self._verifier

    def _get_direct_gate(self):
        if self._direct_gate is None:
            from aegis.defense import DirectInjectionGate

            self._direct_gate = DirectInjectionGate()
        return self._direct_gate

    def _get_indirect_defense(self):
        if self._indirect_defense is None:
            from aegis.defense import IndirectDefense

            self._indirect_defense = IndirectDefense()
        return self._indirect_defense

    def _feedback(self, reports: list[VerifierReport]) -> str:
        """Contradiction summary for the retry prompt.

        Prefers the verifier's own ``.feedback(reports)`` when it provides one
        (test doubles do), else its ``.contradiction_feedback(reports)`` (the
        real :class:`~aegis.verify.GroundednessVerifier` staticmethod).
        """
        verifier = self._get_verifier()
        for attr in ("feedback", "contradiction_feedback"):
            fn = getattr(verifier, attr, None)
            if callable(fn):
                return fn(reports)
        from aegis.verify import GroundednessVerifier

        return GroundednessVerifier.contradiction_feedback(reports)

    def _majority_vote(self, answers: list[str]) -> tuple[str, float, list[str]]:
        try:
            from aegis.verify import majority_vote
        except ImportError:
            return _fallback_majority_vote(answers)
        return majority_vote(answers)

    # ------------------------------------------------------------------ nodes

    def _node_structure(self, state: HarnessState) -> HarnessState:
        """Decompose the question, and screen it with the direct gate."""
        q = state["question"]
        update: HarnessState = {}

        if self.use_defense:
            verdict = self._get_direct_gate().screen(q.question)
            update["direct_injection"] = verdict.is_injection
            update["direct_injection_score"] = verdict.score

        if not self.use_structurer:
            update["structured"] = StructuredQuery(sub_questions=[q.question])
            return update

        structured, resp = self._get_structurer().structure(q.question)
        update["structured"] = structured
        update["responses"] = state.get("responses", []) + [resp]
        return update

    def _node_retrieve(self, state: HarnessState) -> HarnessState:
        """Multi-query retrieval: original question first, then sub-questions.

        Results are merged by chunk id keeping the max-rerank_score entry,
        sorted by rerank_score descending, truncated to ``k_final``.
        """
        q = state["question"]
        structured = state["structured"]

        queries: list[str] = []
        for query in [q.question, *structured.sub_questions]:
            if query and query not in queries:
                queries.append(query)

        best: dict[str, RetrievalResult] = {}
        for query in queries:
            for r in self.retriever.retrieve(query, k_final=self.k_final):
                cid = r.chunk.id
                if cid not in best or _rerank_key(r) > _rerank_key(best[cid]):
                    best[cid] = r

        merged = sorted(best.values(), key=_rerank_key, reverse=True)[: self.k_final]

        if self.use_defense:
            sanitized, report = self._get_indirect_defense().sanitize(merged)
            return {"retrieved": sanitized, "sanitized_chunks": report.n_sanitized}
        return {"retrieved": merged}

    def _node_generate(self, state: HarnessState) -> HarnessState:
        """Citation-contract generation, single-shot or k-vote sampled."""
        q = state["question"]
        retrieved = state["retrieved"]
        valid_ids = {r.chunk.id for r in retrieved}

        # Direct-gate block: a flagged query is refused before any model call.
        if self.use_defense and state.get("direct_injection"):
            blocked = Answer(
                text="",
                citations=[],
                abstained=True,
                raw_response="[blocked: direct prompt-injection detected]",
            )
            return {
                "answer": blocked,
                "gen_messages": [],
                "responses": list(state.get("responses", [])),
                "votes": [],
                "agreement": None,
                "blocked": True,
            }

        system_content = PROMPT_TEMPLATES["raw_rag_system"]
        if self.use_defense and self.canary:
            from aegis.defense import plant_canary

            system_content = plant_canary(system_content, self.canary)

        user_prompt = PROMPT_TEMPLATES["raw_rag_user"].format(
            context=format_context(retrieved), question=q.question
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]
        responses = list(state.get("responses", []))

        if self.k_vote == 1:
            resp = self.client.complete(messages)
            responses.append(resp)
            text, cids = parse_contract(resp.text, valid_ids)
            if not cids:
                # Same one-shot citation-enforcement retry as RAGPipeline.
                retry_messages = messages + [
                    {"role": "assistant", "content": resp.text},
                    {"role": "user", "content": PROMPT_TEMPLATES["citation_retry"]},
                ]
                resp = self.client.complete(retry_messages)
                responses.append(resp)
                text, cids = parse_contract(resp.text, valid_ids)
            answer = Answer(
                text=text or "",
                citations=[Citation(chunk_id=c) for c in cids],
                abstained=not cids,
                raw_response=resp.text,
            )
            answer, leaked = self._apply_canary_defense(answer)
            return {
                "answer": answer,
                "gen_messages": messages,
                "responses": responses,
                "votes": [],
                "agreement": None,
                "canary_leaked": leaked,
                "blocked": leaked,
            }

        # k-vote path: sample the same prompt k times, majority-vote answers.
        voter = self.vote_client or self.client
        samples = [voter.complete(messages) for _ in range(self.k_vote)]
        responses.extend(samples)
        parsed = [parse_contract(s.text, valid_ids) for s in samples]

        majority, agreement, votes = self._majority_vote([text for text, _ in parsed])

        # Citations come from the first sample whose answer is in the
        # majority cluster; fall back to the first sample's citations.
        chosen = 0
        majority_norm = majority.strip().lower()
        for i, (text, _) in enumerate(parsed):
            if text.strip().lower() == majority_norm:
                chosen = i
                break
        cids = parsed[chosen][1]
        answer = Answer(
            text=majority,
            citations=[Citation(chunk_id=c) for c in cids],
            abstained=not cids,
            raw_response=samples[chosen].text,
        )
        answer, leaked = self._apply_canary_defense(answer)
        return {
            "answer": answer,
            "gen_messages": messages,
            "responses": responses,
            "votes": votes,
            "agreement": agreement,
            "canary_leaked": leaked,
            "blocked": leaked,
        }

    def _apply_canary_defense(self, answer: Answer) -> tuple[Answer, bool]:
        """Block the answer if the planted canary leaked into the output.

        Returns ``(answer, leaked)``. On a leak the answer is abstained and its
        text/raw_response are scrubbed so the secret never propagates to logs,
        citations, or the caller. A no-op when defense or the canary is off.
        """
        if not (self.use_defense and self.canary):
            return answer, False
        from aegis.defense import canary_leaked

        leaked = canary_leaked(answer.text, self.canary) or canary_leaked(
            answer.raw_response, self.canary
        )
        if not leaked:
            return answer, False
        scrubbed = answer.model_copy(
            update={
                "text": "",
                "citations": [],
                "abstained": True,
                "raw_response": "[blocked: canary exfiltration detected]",
            }
        )
        return scrubbed, True

    def _node_verify(self, state: HarnessState) -> HarnessState:
        """NLI-verify the current answer against the retrieved chunks."""
        grounded, reports = self._get_verifier().verify(
            state["answer"], state["retrieved"], state["question"].question
        )
        return {"grounded": grounded, "verifier_reports": reports}

    def _node_generate_retry(self, state: HarnessState) -> HarnessState:
        """One re-generation after a failed verification (no voting)."""
        answer = state["answer"]
        feedback = self._feedback(state.get("verifier_reports", []))
        messages = state["gen_messages"] + [
            {"role": "assistant", "content": answer.raw_response},
            {
                "role": "user",
                "content": HARNESS_TEMPLATES["verify_retry"].format(feedback=feedback),
            },
        ]
        resp = self.client.complete(messages)
        valid_ids = {r.chunk.id for r in state["retrieved"]}
        text, cids = parse_contract(resp.text, valid_ids)
        new_answer = Answer(
            text=text or "",
            citations=[Citation(chunk_id=c) for c in cids],
            abstained=False,
            raw_response=resp.text,
        )
        return {
            "answer": new_answer,
            "responses": state.get("responses", []) + [resp],
            "verify_retries": state.get("verify_retries", 0) + 1,
        }

    def _node_abstain(self, state: HarnessState) -> HarnessState:
        """Mark the (still unverified) answer as abstained, keeping its parse."""
        return {"answer": state["answer"].model_copy(update={"abstained": True})}

    def _node_finalize(self, state: HarnessState) -> HarnessState:
        """Assemble the QueryResult, aggregating usage over ALL LLM calls."""
        q = state["question"]
        responses = state.get("responses", [])
        harness = HarnessMeta(
            structured=state.get("structured"),
            verifier_reports=state.get("verifier_reports", []),
            verify_retries=state.get("verify_retries", 0),
            grounded=state.get("grounded"),
            votes=state.get("votes", []),
            agreement=state.get("agreement"),
            llm_calls=len(responses),
            direct_injection=state.get("direct_injection"),
            direct_injection_score=state.get("direct_injection_score"),
            sanitized_chunks=state.get("sanitized_chunks", 0),
            canary_leaked=state.get("canary_leaked", False),
            blocked=state.get("blocked", False),
        )
        result = QueryResult(
            question_id=q.id,
            question=q.question,
            gold_answer=q.answer,
            answer=state["answer"],
            retrieved=state.get("retrieved", []),
            mode="harnessed",
            model=self.client.model,
            cost_usd=sum(r.cost_usd for r in responses),
            latency_s=sum(r.latency_s for r in responses),
            tokens_in=sum(r.tokens_in for r in responses),
            tokens_out=sum(r.tokens_out for r in responses),
            harness=harness,
        )
        return {"result": result}

    # ----------------------------------------------------------------- routes

    def _route_after_generate(self, state: HarnessState) -> str:
        """Skip verification when disabled or when generation already abstained."""
        if state["answer"].abstained or not self.use_verifier:
            return "finalize"
        return "verify"

    def _route_after_verify(self, state: HarnessState) -> str:
        """Grounded -> finalize; first failure -> retry once; else abstain."""
        if state.get("grounded"):
            return "finalize"
        if state.get("verify_retries", 0) == 0:
            return "generate_retry"
        return "abstain"

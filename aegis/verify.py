"""Phase 2 groundedness verification and consistency voting.

``GroundednessVerifier`` checks an :class:`~aegis.types.Answer` against the
chunks it cites using an NLI model: the answer is rephrased as a declarative
claim (the hypothesis) and each cited chunk's text is the premise. The answer
is grounded iff at least one cited chunk entails the claim.

``majority_vote`` implements the consistency-vote mechanism: k sampled answer
strings are clustered by normalized-string equality (the official HotpotQA
normalization from :func:`aegis.eval.scorers.normalize_answer`) and the
largest cluster wins. Clustering is purely string-based and deterministic;
merging clusters by semantic similarity (e.g. embedding distance) is a
possible extension, not part of Phase 2.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from aegis.config import AegisConfig, get_config
from aegis.eval.scorers import normalize_answer
from aegis.types import Answer, RetrievalResult, VerifierReport

#: Tokens ignored when computing FakeNLI hypothesis coverage.
_FAKE_STOPWORDS = frozenset(
    {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "answer", "question"}
)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class FakeNLI:
    """Deterministic lexical-overlap stand-in for a real NLI model.

    Labels a (premise, hypothesis) pair by the fraction of the hypothesis'
    content tokens (stopwords removed) that appear in the premise:
    >= 0.6 -> entailment, <= 0.15 -> contradiction, else neutral. Used when
    ``config.nli_model == "fake"`` so tests never download models.
    """

    name = "fake"

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        premise_tokens = set(_tokens(premise))
        hyp_content = [t for t in _tokens(hypothesis) if t not in _FAKE_STOPWORDS]
        overlap = len(set(hyp_content) & premise_tokens) / max(1, len(set(hyp_content)))
        if overlap >= 0.6:
            return "entailment", round(overlap, 4)
        if overlap <= 0.15:
            return "contradiction", round(1 - overlap, 4)
        return "neutral", round(overlap, 4)

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        return [self.predict(premise, hypothesis) for premise, hypothesis in pairs]


class CrossEncoderNLI:
    """Cross-encoder NLI model (e.g. nli-deberta-v3-base), lazily loaded.

    The label order is read from the loaded model's ``config.id2label`` —
    never hardcoded — because different NLI checkpoints order the
    contradiction/entailment/neutral logits differently.
    """

    def __init__(self, model_name: str, data_dir: Path | None = None):
        self.name = model_name
        self._data_dir = Path(data_dir) if data_dir is not None else get_config().data_dir
        self._model = None
        self._labels: list[str] = []

    @staticmethod
    def _canonical_label(name: str) -> str:
        name = name.lower()
        for canonical in ("contradiction", "entailment", "neutral"):
            if canonical.startswith(name) or name.startswith(canonical[:6]):
                return canonical
        raise ValueError(f"Unrecognized NLI label {name!r}")

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy

            from aegis._models import cached_model

            cache_folder = str(self._data_dir / "models_cache")

            def _load():
                from aegis._models import to_inference_precision

                try:
                    model = CrossEncoder(self.name, cache_folder=cache_folder)
                except TypeError:
                    # Older sentence-transformers versions lack cache_folder.
                    model = CrossEncoder(self.name)
                to_inference_precision(model.model)
                return model

            self._model = cached_model("nli", self.name, _load)
            id2label = self._model.model.config.id2label
            self._labels = [
                self._canonical_label(id2label[i]) for i in range(len(id2label))
            ]
        return self._model

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        return self.predict_batch([(premise, hypothesis)])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Label a batch of (premise, hypothesis) pairs in one model call."""
        if not pairs:
            return []
        model = self._get_model()
        logits = np.asarray(model.predict([tuple(p) for p in pairs])).reshape(
            len(pairs), -1
        )
        # Softmax over the 3 NLI logits, row-wise.
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        results: list[tuple[str, float]] = []
        for row in probs:
            idx = int(np.argmax(row))
            results.append((self._labels[idx], float(row[idx])))
        return results


def _premise(text: str) -> str:
    """Chunk text as NLI premise, minus the defense's spotlight delimiters.

    The delimiters are prompt scaffolding; left in, they push the NLI model
    toward "neutral" on premises that plainly entail the claim.
    """
    from aegis.defense import DATA_CLOSE, DATA_OPEN

    return " ".join(text.replace(DATA_OPEN, " ").replace(DATA_CLOSE, " ").split())


def _claim_forms(answer_text: str, question: str) -> list[str]:
    """Hypotheses to test: the templated claim first, then a sentence answer as-is.

    Short answers ("Dijon") need the question for context; a full-sentence
    answer is already a claim, and wrapping it in the template garbles it.
    """
    text = answer_text.strip()
    forms = [f"The answer to the question '{question}' is: {text.rstrip('.')}."]
    if len(text.split()) >= 4:
        forms.append(text if text.endswith((".", "!", "?")) else f"{text}.")
    return forms


def get_nli(config: AegisConfig | None = None) -> FakeNLI | CrossEncoderNLI:
    """Return the NLI model selected by ``config.nli_model``."""
    config = config or get_config()
    if config.nli_model == "fake":
        return FakeNLI()
    return CrossEncoderNLI(config.nli_model, config.data_dir)


class GroundednessVerifier:
    """NLI-based check that an answer is entailed by at least one cited chunk."""

    def __init__(self, nli):
        self.nli = nli

    def verify(
        self,
        answer: Answer,
        retrieved: list[RetrievalResult],
        question: str,
        claims: list[str] | None = None,
    ) -> tuple[bool, list[VerifierReport]]:
        """Verify ``answer`` against the chunks it cites.

        Returns ``(grounded, reports)``. Abstentions, empty answers, and
        answers without citations are unverifiable and return ``(False, [])``.
        Citations whose chunk id is not among ``retrieved`` are skipped.
        ``grounded`` is True iff a cited chunk, or the cited chunks together,
        entail the claim.

        ``claims`` are standalone factual sentences to check in place of the
        built-in claim templates (the harness writes one with the model; see
        ``HarnessedPipeline._node_verify``).
        """
        if answer.abstained or not answer.text.strip() or not answer.citations:
            return False, []

        by_id = {r.chunk.id: r.chunk for r in retrieved}
        chunk_ids = [c.chunk_id for c in answer.citations if c.chunk_id in by_id]
        if not chunk_ids:
            return False, []

        hypotheses = [c for c in (claims or []) if c.strip()] or _claim_forms(
            answer.text, question
        )
        reports = [
            self._best_report(chunk_id, _premise(by_id[chunk_id].text), hypotheses)
            for chunk_id in chunk_ids
        ]
        grounded = any(r.label == "entailment" for r in reports)

        # Multi-hop answers are often entailed only by the cited chunks taken
        # together (bridge fact in one, answer in another).
        if not grounded and len(chunk_ids) > 1:
            joined = " ".join(_premise(by_id[c].text) for c in chunk_ids)
            combined = self._best_report("+".join(chunk_ids), joined, hypotheses)
            reports.append(combined)
            grounded = combined.label == "entailment"
        return grounded, reports

    def _best_report(
        self, chunk_id: str, premise: str, hypotheses: list[str]
    ) -> VerifierReport:
        """NLI-judge one premise against each claim form; keep the best."""
        pairs = [(premise, h) for h in hypotheses]
        if hasattr(self.nli, "predict_batch"):
            judgments = self.nli.predict_batch(pairs)
        else:
            judgments = [self.nli.predict(p, h) for p, h in pairs]
        entailed = [
            (score, hyp)
            for hyp, (label, score) in zip(hypotheses, judgments)
            if label == "entailment"
        ]
        if entailed:
            score, hyp = max(entailed)
            return VerifierReport(claim=hyp, chunk_id=chunk_id, label="entailment", score=score)
        label, score = judgments[0]
        return VerifierReport(claim=hypotheses[0], chunk_id=chunk_id, label=label, score=score)

    @staticmethod
    def contradiction_feedback(reports: list[VerifierReport]) -> str:
        """Human-readable summary of non-entailed chunks for the retry prompt."""
        failing = [r for r in reports if r.label != "entailment"]
        if not failing:
            return "All cited chunks entail the answer."
        lines = [
            f"Chunk {r.chunk_id}: {r.label} (score {r.score:.4f})" for r in failing
        ]
        return (
            "The answer was not supported by the following cited chunks:\n"
            + "\n".join(lines)
        )


def semantic_entropy(answers: list[str]) -> float:
    """Shannon entropy (nats) of the answer distribution over meaning clusters.

    Clusters non-empty answers by :func:`normalize_answer` equality (the same
    clustering :func:`majority_vote` uses) and returns the entropy of the
    resulting cluster-size distribution. Low entropy = the model is
    self-consistent (a stronger confidence signal than the flat agreement rate,
    per Kuhn et al. 2023); ``0.0`` when all samples agree or input is empty.
    """
    non_empty = [a for a in answers if a and a.strip()]
    if not non_empty:
        return 0.0
    clusters: dict[str, int] = {}
    for a in non_empty:
        key = normalize_answer(a)
        clusters[key] = clusters.get(key, 0) + 1
    total = len(non_empty)
    probs = np.array([c / total for c in clusters.values()], dtype=float)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def majority_vote(answers: list[str]) -> tuple[str, float, list[str]]:
    """Consistency vote over k sampled answer strings.

    Non-empty answers are clustered by :func:`normalize_answer` equality
    (exact normalized-string match — deterministic, no embeddings; semantic-
    similarity merging is a possible extension). The largest cluster wins,
    ties broken by first occurrence.

    Returns ``(winner, agreement, answers)`` where ``winner`` is the first
    original-casing member of the majority cluster, ``agreement`` is the
    majority cluster size divided by the number of non-empty answers, and
    ``answers`` is the original input list. All-empty input yields
    ``("", 0.0, answers)``.
    """
    non_empty = [a for a in answers if a and a.strip()]
    if not non_empty:
        return "", 0.0, answers

    clusters: dict[str, list[str]] = {}
    for a in non_empty:
        clusters.setdefault(normalize_answer(a), []).append(a)

    # dict preserves insertion order, so max() breaks ties by first occurrence.
    majority = max(clusters.values(), key=len)
    return majority[0], len(majority) / len(non_empty), answers

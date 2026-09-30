"""HotpotQA benchmark loading and corpus construction.

Loads the HuggingFace ``hotpot_qa`` (distractor) dataset, normalizes records
into :class:`~aegis.types.HotpotQuestion`, and builds a deduplicated
paragraph-level retrieval corpus of :class:`~aegis.types.Chunk` objects.
"""

from __future__ import annotations

import hashlib
import random

from aegis.config import get_config
from aegis.types import Chunk, HotpotQuestion


def _to_question(record: dict) -> HotpotQuestion:
    """Map one raw HF ``hotpot_qa`` (distractor) record to a HotpotQuestion.

    HF schema notes:
    - ``supporting_facts`` is ``{"title": [...], "sent_id": [...]}``.
    - ``context`` is ``{"title": [...], "sentences": [[...], ...]}``.
    """
    sf = record["supporting_facts"]
    supporting_facts = [
        (title, int(sent_id)) for title, sent_id in zip(sf["title"], sf["sent_id"])
    ]
    ctx = record["context"]
    context = [
        (title, list(sentences))
        for title, sentences in zip(ctx["title"], ctx["sentences"])
    ]
    return HotpotQuestion(
        id=record["id"],
        question=record["question"],
        answer=record["answer"],
        type=record["type"],
        level=record["level"],
        supporting_facts=supporting_facts,
        context=context,
    )


def load_hotpotqa(n: int, seed: int, split: str = "validation") -> list[HotpotQuestion]:
    """Load ``n`` HotpotQA questions, sampled deterministically by ``seed``.

    If ``n <= 0`` or ``n >= len(dataset)``, all questions are returned (in
    shuffled order).
    """
    from datasets import load_dataset  # lazy: heavy dependency

    config = get_config()
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=split,
        cache_dir=str(config.data_dir / "hf_cache"),
    )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if 0 < n < len(indices):
        indices = indices[:n]
    return [_to_question(dataset[i]) for i in indices]


def sample_hash(questions: list[HotpotQuestion]) -> str:
    """Stable, order-independent 12-char hash of the sampled question ids."""
    joined = ",".join(sorted(q.id for q in questions))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def build_corpus(questions: list[HotpotQuestion]) -> list[Chunk]:
    """Build a unified, title-deduplicated corpus across all questions.

    Questions are sorted by id first so the corpus (including chunk ids,
    which embed insertion order) is independent of sampling shuffle order.
    """
    chunks: list[Chunk] = []
    seen_titles: set[str] = set()
    for question in sorted(questions, key=lambda q: q.id):
        for title, sentences in question.context:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            chunks.append(
                Chunk(
                    id=f"{title}::{len(chunks)}",
                    title=title,
                    text=" ".join(sentences),
                    sentences=list(sentences),
                    metadata={"poisoned": False},
                )
            )
    return chunks


def remove_supporting_paragraphs(
    chunks: list[Chunk], questions: list[HotpotQuestion]
) -> list[Chunk]:
    """Drop every paragraph that holds a supporting fact for any of ``questions``.

    Turns the answerable corpus into an *unanswerable* one: the distractor
    paragraphs stay, so retrieval still returns plausible-looking context, but
    the evidence needed to answer is gone. A reliable system should decline;
    answering anyway means guessing or falling back on memorized knowledge.
    """
    gold_titles = {title for q in questions for title, _ in q.supporting_facts}
    return [c for c in chunks if c.title not in gold_titles]

"""Tests for the poisoned-corpus generator and ASR scoring (aegis.eval.poison)."""

from aegis.eval.poison import (
    ATTACK_CATEGORIES,
    attack_succeeded,
    poison_corpus,
)
from aegis.types import Chunk


def _corpus(n: int = 30) -> list[Chunk]:
    return [
        Chunk(
            id=f"T{i}::{i}",
            title=f"T{i}",
            text=f"This is clean fact number {i} about a topic.",
            sentences=[f"This is clean fact number {i} about a topic."],
            metadata={"poisoned": False},
        )
        for i in range(n)
    ]


class TestPoisonCorpus:
    def test_rate_zero_poisons_nothing(self):
        chunks, manifest = poison_corpus(_corpus(), rate=0.0, seed=0)
        assert manifest.n_poisoned == 0
        assert all(not c.metadata["poisoned"] for c in chunks)

    def test_approximate_rate(self):
        chunks, manifest = poison_corpus(_corpus(30), rate=0.1, seed=0)
        assert manifest.n_poisoned == 3
        poisoned = [c for c in chunks if c.metadata["poisoned"]]
        assert len(poisoned) == 3

    def test_deterministic_given_seed(self):
        a, ma = poison_corpus(_corpus(), rate=0.2, seed=42)
        b, mb = poison_corpus(_corpus(), rate=0.2, seed=42)
        assert [c.id for c in a if c.metadata["poisoned"]] == [
            c.id for c in b if c.metadata["poisoned"]
        ]
        assert [att.payload for att in ma.attacks] == [att.payload for att in mb.attacks]

    def test_different_seeds_differ(self):
        _, ma = poison_corpus(_corpus(), rate=0.2, seed=1)
        _, mb = poison_corpus(_corpus(), rate=0.2, seed=2)
        assert {a.chunk_id for a in ma.attacks} != {a.chunk_id for a in mb.attacks}

    def test_does_not_mutate_input(self):
        corpus = _corpus()
        original = corpus[0].text
        poison_corpus(corpus, rate=0.5, seed=0)
        assert corpus[0].text == original

    def test_metadata_tags_present(self):
        chunks, _ = poison_corpus(_corpus(), rate=0.3, seed=0)
        for c in chunks:
            if c.metadata["poisoned"]:
                assert c.metadata["attack_category"] in ATTACK_CATEGORIES
                assert "attack_sentinel" in c.metadata

    def test_payload_prepended_to_text(self):
        chunks, manifest = poison_corpus(_corpus(), rate=0.1, seed=0)
        poisoned = [c for c in chunks if c.metadata["poisoned"]][0]
        # Original clean text is preserved after the payload.
        assert "clean fact" in poisoned.text
        assert poisoned.text.startswith(poisoned.sentences[0])

    def test_categories_are_balanced_round_robin(self):
        _, manifest = poison_corpus(_corpus(30), rate=0.3, seed=0)  # 9 attacks
        counts = {c: len(v) for c, v in manifest.by_category().items()}
        assert sum(counts.values()) == manifest.n_poisoned
        # Round-robin over 3 categories with 9 attacks -> 3 each.
        assert all(v == 3 for v in counts.values())


class TestAttackSucceeded:
    def test_sentinel_attack(self):
        _, manifest = poison_corpus(_corpus(), rate=0.2, seed=0)
        override = next(a for a in manifest.attacks if a.category != "exfiltration")
        assert attack_succeeded(f"the answer is {override.sentinel}", override) is True
        assert attack_succeeded("a clean answer", override) is False

    def test_exfiltration_uses_canary(self):
        _, manifest = poison_corpus(_corpus(30), rate=0.3, seed=0)
        exf = next(a for a in manifest.attacks if a.category == "exfiltration")
        canary = "AEGIS-CANARY-deadbeef"
        assert attack_succeeded(f"the token is {canary}", exf, canary=canary) is True
        assert attack_succeeded("no leak here", exf, canary=canary) is False

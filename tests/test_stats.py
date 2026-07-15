"""Unit tests for aegis.eval.stats (pure; no mlflow, no network)."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from aegis.eval.stats import (
    bh_correction,
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap_delta,
    wilcoxon_paired,
)


class TestBootstrapCI:
    def test_empty_input_is_nan(self):
        mean, lo, hi = bootstrap_ci([])
        assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)

    def test_constant_vector_collapses(self):
        mean, lo, hi = bootstrap_ci([0.5] * 20)
        assert mean == lo == hi == 0.5

    def test_mean_inside_interval(self):
        rng = np.random.default_rng(42)
        values = rng.normal(loc=0.7, scale=0.1, size=100).tolist()
        mean, lo, hi = bootstrap_ci(values, seed=0)
        assert lo <= mean <= hi
        assert mean == pytest.approx(float(np.mean(values)))

    def test_deterministic_given_seed(self):
        values = list(np.linspace(0.0, 1.0, 50))
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_different_seeds_differ(self):
        values = list(np.linspace(0.0, 1.0, 50))
        a = bootstrap_ci(values, seed=1)
        b = bootstrap_ci(values, seed=2)
        # Means are identical; the resampled interval should differ.
        assert a[0] == b[0]
        assert (a[1], a[2]) != (b[1], b[2])


class TestMcNemar:
    @staticmethod
    def _build(n01: int, n10: int, n_both: int = 50):
        """Paired correctness vectors with given discordant counts."""
        a = [True] * n_both + [True] * n01 + [False] * n10
        b = [True] * n_both + [False] * n01 + [True] * n10
        return a, b

    def test_asymmetric_discordance_significant(self):
        a, b = self._build(n01=12, n10=3)
        out = mcnemar_test(a, b)
        assert out["b"] == 12
        assert out["c"] == 3
        assert out["p_value"] < 0.05

    def test_symmetric_discordance_not_significant(self):
        a, b = self._build(n01=5, n10=5)
        out = mcnemar_test(a, b)
        assert out["b"] == 5
        assert out["c"] == 5
        assert out["p_value"] > 0.5

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([True, False], [True])


class TestWilcoxon:
    def test_small_shift_significant(self):
        rng = np.random.default_rng(0)
        a = rng.normal(loc=0.5, scale=0.1, size=30)
        b = a + 0.05
        out = wilcoxon_paired(b, a)
        assert out["p_value"] < 0.01

    def test_identical_arrays_p_is_one(self):
        a = [0.1, 0.2, 0.3, 0.4]
        out = wilcoxon_paired(a, list(a))
        assert out["p_value"] == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_paired([1.0, 2.0], [1.0])


class TestPairedBootstrapDelta:
    def test_shift_gives_positive_interval(self):
        rng = np.random.default_rng(3)
        b = rng.normal(loc=0.5, scale=0.1, size=50)
        a = b + 0.1
        out = paired_bootstrap_delta(a, b, seed=0)
        assert out["delta"] == pytest.approx(0.1)
        assert out["lo"] > 0
        assert out["lo"] <= out["delta"] <= out["hi"]

    def test_empty_input_is_nan(self):
        out = paired_bootstrap_delta([], [])
        assert math.isnan(out["delta"])


class TestBHCorrection:
    def test_known_family(self):
        p = [0.001, 0.01, 0.02, 0.9]
        out = bh_correction(p, alpha=0.05)
        assert out["reject"][0] is True
        assert out["reject"][-1] is False
        # Adjusted p-values are >= raw and monotone in the sorted order.
        for raw, adj in zip(p, out["p_adjusted"]):
            assert adj >= raw
        assert out["p_adjusted"] == sorted(out["p_adjusted"])

    def test_empty(self):
        out = bh_correction([])
        assert out == {"reject": [], "p_adjusted": []}


def test_prompt_hash_deterministic():
    """prompt_hash is stable — guarded: aegis.pipeline is built in parallel."""
    try:
        import aegis.pipeline  # noqa: F401
    except ImportError:
        pytest.skip("aegis.pipeline not available yet")
    from aegis.eval.run import prompt_hash

    h1 = prompt_hash()
    h2 = prompt_hash()
    assert h1 == h2
    assert len(h1) == 12
    assert all(c in "0123456789abcdef" for c in h1)

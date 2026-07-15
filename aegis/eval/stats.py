"""Statistical tests and interval estimates for Aegis evaluation runs.

All heavy statistical dependencies (scipy, statsmodels) are imported lazily
inside functions so that this module stays importable in environments where
only numpy is available.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Returns ``(mean, lo, hi)`` where ``[lo, hi]`` is the ``1 - alpha``
    percentile bootstrap CI. Empty input returns ``(nan, nan, nan)``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lo, hi)


def mcnemar_test(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Paired McNemar test on per-question correctness of two systems.

    ``b`` counts questions A got right and B got wrong (n01 in the 2x2
    table's off-diagonal as viewed from A), ``c`` the reverse. Uses the
    exact binomial test when the discordant counts are small (min < 25),
    otherwise the chi-square test with continuity correction.

    Returns ``{"b": ..., "c": ..., "statistic": ..., "p_value": ...}``.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("a_correct and b_correct must have the same length")

    n11 = int(np.sum(a & b))
    n01 = int(np.sum(a & ~b))  # A right, B wrong
    n10 = int(np.sum(~a & b))  # A wrong, B right
    n00 = int(np.sum(~a & ~b))
    table = [[n11, n01], [n10, n00]]

    use_exact = min(n01, n10) < 25
    result = mcnemar(table, exact=use_exact, correction=True)
    return {
        "b": n01,
        "c": n10,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank test on paired scores (zero_method='pratt').

    If all paired differences are zero the test is undefined; we return
    ``{"statistic": 0.0, "p_value": 1.0}`` (no evidence of any difference).
    """
    from scipy import stats as scipy_stats

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same length")

    diffs = a_arr - b_arr
    if a_arr.size == 0 or np.allclose(diffs, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}

    result = scipy_stats.wilcoxon(a_arr, b_arr, zero_method="pratt")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for ``mean(a) - mean(b)`` resampling paired indices.

    Returns ``{"delta": ..., "lo": ..., "hi": ...}``.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same length")
    if a_arr.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a_arr.size, size=(n_boot, a_arr.size))
    deltas = a_arr[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
    return {
        "delta": float(a_arr.mean() - b_arr.mean()),
        "lo": float(np.percentile(deltas, 100 * (alpha / 2))),
        "hi": float(np.percentile(deltas, 100 * (1 - alpha / 2))),
    }


def bh_correction(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR correction over a family of p-values.

    Returns ``{"reject": [bool, ...], "p_adjusted": [float, ...]}`` in the
    input order, via statsmodels' ``multipletests(method="fdr_bh")``.
    """
    from statsmodels.stats.multitest import multipletests

    p_arr = list(p_values)
    if not p_arr:
        return {"reject": [], "p_adjusted": []}
    reject, p_adj, _, _ = multipletests(p_arr, alpha=alpha, method="fdr_bh")
    return {
        "reject": [bool(r) for r in reject],
        "p_adjusted": [float(p) for p in p_adj],
    }

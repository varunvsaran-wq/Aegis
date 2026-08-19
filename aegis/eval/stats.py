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


def mcnemar_test(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    method: str = "midp",
) -> dict:
    """Paired McNemar test on per-question correctness of two systems.

    ``b`` counts questions A got right and B got wrong (n01 in the 2x2
    table's off-diagonal as viewed from A), ``c`` the reverse.

    ``method`` selects the p-value:

    - ``"midp"`` (default) — the mid-p McNemar test (Fagerland, Lydersen &
      Laake 2013): the exact-conditional two-sided p minus the point mass at
      the observed split. Near-nominal coverage and more powerful than the
      exact-conditional test; recommended for paper claims.
    - ``"exact"`` — exact-conditional binomial (conservative).
    - ``"asymptotic"`` — chi-square with continuity correction (only for large
      discordant counts).
    - ``"auto"`` — the legacy rule: exact when ``min(discordant) < 25`` else
      asymptotic with continuity correction.

    Returns ``{"b", "c", "statistic", "p_value", "method"}``.
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

    if method == "midp":
        p_value = _mcnemar_midp(n01, n10)
        statistic = float(min(n01, n10))
    else:
        use_exact = method == "exact" or (method == "auto" and min(n01, n10) < 25)
        result = mcnemar(table, exact=use_exact, correction=True)
        p_value = float(result.pvalue)
        statistic = float(result.statistic)

    return {
        "b": n01,
        "c": n10,
        "statistic": statistic,
        "p_value": p_value,
        "method": method,
    }


def _mcnemar_midp(n01: int, n10: int) -> float:
    """Two-sided mid-p value for a McNemar table's discordant pairs.

    Under H0 the smaller discordant count is Binomial(n=n01+n10, p=0.5). The
    mid-p subtracts half the probability mass at the observed count from the
    exact two-sided p, giving near-nominal coverage.
    """
    from scipy.stats import binom

    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    exact_two_sided = min(1.0, 2.0 * float(binom.cdf(k, n, 0.5)))
    point_mass = float(binom.pmf(k, n, 0.5))
    return max(0.0, min(1.0, exact_two_sided - point_mass))


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


def hierarchical_bootstrap_ci(
    score_matrix: Sequence[Sequence[float]],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Two-level (question, seed) clustered bootstrap CI for the grand mean.

    ``score_matrix`` is ``[n_questions][n_seeds]``: entry ``[q][s]`` is the
    metric for question ``q`` under seed ``s``. The question is the resampling
    cluster (outer level); seeds are resampled within each drawn question
    (inner level). This propagates BOTH question-sampling and seed variance
    into the interval, unlike a within-run bootstrap that sees only one seed.

    Returns ``(mean, lo, hi)``. Empty input returns ``(nan, nan, nan)``.
    Ragged rows are allowed (a question may have fewer seeds).
    """
    rows = [np.asarray(r, dtype=float) for r in score_matrix if len(r) > 0]
    if not rows:
        return (float("nan"), float("nan"), float("nan"))

    grand_mean = float(np.mean([r.mean() for r in rows]))
    n_q = len(rows)
    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        q_idx = rng.integers(0, n_q, size=n_q)
        acc = 0.0
        for qi in q_idx:
            row = rows[qi]
            s_idx = rng.integers(0, row.size, size=row.size)
            acc += float(row[s_idx].mean())
        boot[b] = acc / n_q
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (grand_mean, lo, hi)


def proportion_ci(k: int, n: int, method: str = "wilson", alpha: float = 0.05) -> dict:
    """Confidence interval for a binomial proportion ``k`` successes of ``n``.

    ``method="wilson"`` (default) is the Wilson score interval — the
    recommended default for small/extreme proportions such as per-category
    attack-success rates (Brown, Cai & DasGupta 2001). ``method="clopper"``
    gives the exact Clopper-Pearson interval (guaranteed coverage, wider);
    ``method="normal"`` gives the (discouraged) Wald interval.

    Returns ``{"p", "lo", "hi", "k", "n", "method"}``. ``n == 0`` yields
    ``p=nan`` and the full ``[0, 1]`` interval.
    """
    if n <= 0:
        return {"p": float("nan"), "lo": 0.0, "hi": 1.0, "k": k, "n": n, "method": method}

    p = k / n
    if method == "normal":
        z = _z(alpha)
        half = z * np.sqrt(p * (1 - p) / n)
        lo, hi = p - half, p + half
    elif method == "clopper":
        from scipy.stats import beta

        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    else:  # wilson
        z = _z(alpha)
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        lo, hi = center - half, center + half

    return {
        "p": float(p),
        "lo": float(max(0.0, lo)),
        "hi": float(min(1.0, hi)),
        "k": int(k),
        "n": int(n),
        "method": method,
    }


def _z(alpha: float) -> float:
    """Two-sided normal critical value for confidence level ``1 - alpha``."""
    from scipy.stats import norm

    return float(norm.ppf(1 - alpha / 2))


def non_inferiority(
    a: Sequence[float],
    b: Sequence[float],
    margin: float,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """One-sided paired non-inferiority test that ``a`` is not worse than ``b``.

    Tests H0: ``mean(a) - mean(b) <= -margin`` vs. H1: ``mean(a) - mean(b) >
    -margin`` via a paired bootstrap on the per-item deltas. ``a`` is declared
    **non-inferior** to ``b`` when the lower bound of the (two-sided
    ``1-alpha``) paired-bootstrap CI on ``mean(a) - mean(b)`` exceeds
    ``-margin``. This is the correct way to back a "cheap+harness is as good as
    frontier-raw" claim — a non-significant difference is NOT evidence of
    equivalence.

    Returns ``{"delta", "ci_lo", "ci_hi", "margin", "non_inferior",
    "p_value"}`` where ``p_value`` is the one-sided bootstrap p for H0.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same length")
    if a_arr.size == 0:
        return {
            "delta": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "margin": margin,
            "non_inferior": False,
            "p_value": float("nan"),
        }

    boot = paired_bootstrap_delta(a_arr, b_arr, n_boot=n_boot, alpha=alpha, seed=seed)
    diffs = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_deltas = diffs[idx].mean(axis=1)
    # One-sided p for H0: delta <= -margin.
    p_value = float(np.mean(boot_deltas <= -margin))
    return {
        "delta": boot["delta"],
        "ci_lo": boot["lo"],
        "ci_hi": boot["hi"],
        "margin": float(margin),
        "non_inferior": bool(boot["lo"] > -margin),
        "p_value": p_value,
    }


def mde_mcnemar(
    n: int,
    discordance: float = 0.25,
    power: float = 0.8,
    alpha: float = 0.05,
) -> dict:
    """Minimum detectable EM difference for a paired McNemar test.

    Given ``n`` paired questions and an assumed total discordance rate
    ``discordance`` (fraction of questions where the two systems disagree),
    returns the smallest true difference in proportions detectable at the given
    ``power`` and ``alpha``, using the normal approximation to the paired-
    proportions (McNemar) test.

    Returns ``{"n", "discordance", "power", "alpha", "mde"}`` where ``mde`` is
    expressed in proportion points (multiply by 100 for EM percentage points).
    """
    z_alpha = _z(alpha)
    from scipy.stats import norm

    z_beta = float(norm.ppf(power))
    n_disc = max(1.0, n * discordance)
    # Effect on the discordant split needed for power; mapped back to the
    # overall proportion difference (difference = odds imbalance / n).
    mde_disc = (z_alpha + z_beta) * np.sqrt(discordance / n)
    return {
        "n": int(n),
        "discordance": float(discordance),
        "power": float(power),
        "alpha": float(alpha),
        "mde": float(mde_disc),
        "n_discordant_expected": float(n_disc),
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

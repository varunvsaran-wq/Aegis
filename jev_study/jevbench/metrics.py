"""Detection, calibration, latency and cost metrics (numpy only).

Conventions: ``y`` is 0/1 ground truth, ``s`` a score where higher means
"more likely positive". Thresholds flag ``s >= t``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """Area under the ROC curve (Mann-Whitney U with tie handling)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_fpr(y_neg_scores: Sequence[float], fpr: float) -> float:
    """Smallest threshold whose false-alarm rate on these negatives is <= ``fpr``.

    Chosen on dev negatives, then applied unchanged to test.
    """
    neg = np.sort(np.asarray(y_neg_scores, dtype=float))[::-1]
    if neg.size == 0:
        return 0.5
    k = int(np.floor(fpr * neg.size))  # number of negatives allowed at or above t
    if k >= neg.size:
        return float(neg[-1])
    # Flag strictly above the (k+1)-th highest negative score.
    return float(np.nextafter(neg[k], np.inf))


def rates(y: Sequence[int], s: Sequence[float], t: float) -> dict:
    y = np.asarray(y, dtype=int)
    flag = np.asarray(s, dtype=float) >= t
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return {
        "threshold": float(t),
        "tpr": tp / n_pos if n_pos else float("nan"),
        "fpr": fp / n_neg if n_neg else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "n_pos": n_pos, "n_neg": n_neg,
    }


def tpr_at_fpr(y: Sequence[int], s: Sequence[float], fpr: float) -> float:
    """Best detection rate with false alarms <= ``fpr`` (oracle threshold on this data)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    t = threshold_at_fpr(s[y == 0], fpr)
    return rates(y, s, t)["tpr"]


def ece(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    """Expected calibration error of P(positive), equal-width bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.sum() / p.size * abs(p[m].mean() - y[m].mean())
    return float(err)


def reliability(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "freq": float(y[m].mean())})
    return out


def confidence_ece(correct: Sequence[bool], conf: Sequence[float], n_bins: int = 10) -> float:
    """ECE of a *reported confidence* against whether the hard choice was right."""
    return ece(np.asarray(correct, dtype=int), conf, n_bins)


def brier(y: Sequence[int], p: Sequence[float]) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def latency_summary(ms: Sequence[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    if a.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean())}


def cost_per_1k(costs: Sequence[float]) -> float:
    a = np.asarray(costs, dtype=float)
    return float(a.mean() * 1000) if a.size else float("nan")


def stratified_bootstrap(y: Sequence[int], s: Sequence[float],
                         stat: Callable[[np.ndarray, np.ndarray], float],
                         n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling positives and negatives separately."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    point = stat(y, s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size) if pos.size else pos,
                              rng.choice(neg, neg.size) if neg.size else neg])
        v = stat(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_bootstrap_diff(y: Sequence[int], s_a: Sequence[float], s_b: Sequence[float],
                          stat: Callable[[np.ndarray, np.ndarray], float],
                          n_boot: int = 2000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float, float]:
    """stat(a) - stat(b) on the same examples, with a stratified paired bootstrap CI."""
    y = np.asarray(y, dtype=int)
    a, b = np.asarray(s_a, dtype=float), np.asarray(s_b, dtype=float)
    point = stat(y, a) - stat(y, b)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        v = stat(y[idx], a[idx]) - stat(y[idx], b[idx])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)

"""Study 4: confidence-gated routing between a cheap and a strong model.

RouterBench stores, for ~36k prompts, each of 11 models' score (0-1) and cost,
so routing is evaluated offline: a router only has to score each prompt, and
the quality and cost of any routing policy is looked up. The only live calls
are to the router itself (Jev), whose cost is added to every total.

A router gives each prompt a score ``s`` ("needs the strong model"). Sending the
top ``k`` prompts by ``s`` to the strong model, for k = 0..n, traces a
cost-quality curve. Summaries:

- AIQ (RouterBench's average improvement in quality): the area under the upper
  convex hull of that curve over the cost range from all-cheap to all-strong,
  divided by the range. Random routing scores the straight line between the
  endpoints.
- Cost to reach 95% of the strong model's quality.
- Quality at fixed strong-model shares (20/40/60%).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
ROUTERBENCH_URL = ("https://huggingface.co/datasets/withmartian/routerbench/resolve/main/"
                   "routerbench_0shot.pkl")

CHEAP = "mistralai/mixtral-8x7b-chat"
STRONG = "gpt-4-1106-preview"
DEV_FRACTION = 0.3


def family(eval_name: str) -> str:
    for prefix in ("mmlu", "mtbench", "chinese", "Chinese"):
        if eval_name.startswith(prefix):
            return prefix.lower()
    return eval_name


def prompt_text(raw: str) -> str:
    """RouterBench stores prompts as the repr of a list of turns."""
    try:
        turns = ast.literal_eval(raw)
        if isinstance(turns, (list, tuple)):
            return "\n\n".join(str(t) for t in turns)
    except (ValueError, SyntaxError):
        pass
    return str(raw)


def split_of(sample_id: str, dev_fraction: float = DEV_FRACTION) -> str:
    h = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
    return "dev" if h / 0xFFFFFFFF < dev_fraction else "test"


@dataclass
class RoutingSet:
    ids: list[str]
    prompts: list[str]
    families: list[str]
    q_cheap: np.ndarray
    q_strong: np.ndarray
    c_cheap: np.ndarray
    c_strong: np.ndarray

    def subset(self, mask) -> RoutingSet:
        m = np.asarray(mask, dtype=bool)
        pick = np.flatnonzero(m)
        return RoutingSet([self.ids[i] for i in pick], [self.prompts[i] for i in pick],
                          [self.families[i] for i in pick], self.q_cheap[m],
                          self.q_strong[m], self.c_cheap[m], self.c_strong[m])

    def __len__(self) -> int:
        return len(self.ids)


def load_routerbench(per_family: int = 250, seed: int = 0, cheap: str = CHEAP,
                     strong: str = STRONG, split: str | None = None,
                     path: Path | None = None) -> RoutingSet:
    """A family-stratified sample (up to ``per_family`` prompts per task family).

    Stratifying stops HellaSwag and GSM8K (half of RouterBench) from dominating.
    """
    import pandas as pd

    path = path or CACHE_DIR / "routerbench_0shot.pkl"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ROUTERBENCH_URL, path)
    df = pd.read_pickle(path)
    df = df.dropna(subset=[cheap, strong, f"{cheap}|total_cost", f"{strong}|total_cost"])
    df = df[df["eval_name"] != "test-match"]  # 3 placeholder rows
    df = df.assign(family=df["eval_name"].map(family)).sort_values("sample_id")
    parts = [g.sample(min(per_family, len(g)), random_state=seed)
             for _, g in df.groupby("family", sort=True)]
    df = pd.concat(parts).sort_values("sample_id")
    rs = RoutingSet(
        ids=df["sample_id"].tolist(),
        prompts=[prompt_text(p) for p in df["prompt"]],
        families=df["family"].tolist(),
        q_cheap=df[cheap].to_numpy(float), q_strong=df[strong].to_numpy(float),
        c_cheap=df[f"{cheap}|total_cost"].to_numpy(float),
        c_strong=df[f"{strong}|total_cost"].to_numpy(float),
    )
    if split:
        rs = rs.subset([split_of(i) == split for i in rs.ids])
    return rs


# ---------------------------------------------------------------------------
# Curves and summaries
# ---------------------------------------------------------------------------


def curve(rs: RoutingSet, scores, router_cost=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(share_strong, mean_cost, mean_quality) for every k = 0..n.

    Ties in ``scores`` are broken by prompt order, deterministically.
    """
    s = np.asarray(scores, dtype=float)
    n = len(s)
    rc = np.zeros(n) if router_cost is None else np.asarray(router_cost, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    dq = (rs.q_strong - rs.q_cheap)[order]
    dc = (rs.c_strong - rs.c_cheap)[order]
    base_q, base_c = rs.q_cheap.sum(), rs.c_cheap.sum() + rc.sum()
    q = np.concatenate([[base_q], base_q + np.cumsum(dq)]) / n
    c = np.concatenate([[base_c], base_c + np.cumsum(dc)]) / n
    return np.arange(n + 1) / n, c, q


def upper_hull(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper concave hull (a router can mix any two of its operating points)."""
    pts = sorted(zip(x.tolist(), y.tolist()))
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    # Quality never has to drop when paying more: keep the running maximum.
    hx = np.array([h[0] for h in hull])
    hy = np.maximum.accumulate(np.array([h[1] for h in hull]))
    return hx, hy


def aiq(cost: np.ndarray, quality: np.ndarray, lo: float, hi: float, grid: int = 1000) -> float:
    """Mean hull quality over [lo, hi] (the fixed all-cheap to all-strong cost range).

    Below the router's cheapest point (its own cost shifts the curve right) the
    hull is clamped to its first quality; with Jev's cost this gap is negligible,
    and it is reported separately as the router's cost share.
    """
    hx, hy = upper_hull(cost, quality)
    xs = np.linspace(lo, hi, grid)
    return float(np.interp(xs, hx, hy).mean())


def cost_to_reach(cost: np.ndarray, quality: np.ndarray, target: float) -> float:
    ok = quality >= target - 1e-12
    return float(cost[ok].min()) if ok.any() else float("nan")


def summarize(rs: RoutingSet, scores, router_cost=None, target_frac: float = 0.95) -> dict:
    share, c, q = curve(rs, scores, router_cost)
    lo, hi = rs.c_cheap.mean(), rs.c_strong.mean()
    q_lo, q_hi = rs.q_cheap.mean(), rs.q_strong.mean()
    out = {
        "aiq": aiq(c, q, lo, hi),
        # 95% of the strong model's quality, and 95% of the cheap-to-strong gap.
        "cost_to_95": cost_to_reach(c, q, target_frac * q_hi),
        "cost_to_95_gap": cost_to_reach(c, q, q_lo + target_frac * (q_hi - q_lo)),
        "router_cost_per_prompt": float(np.mean(router_cost)) if router_cost is not None else 0.0,
    }
    for pct in (20, 40, 60):
        k = int(round(pct / 100 * len(rs)))
        out[f"quality_at_{pct}"] = float(q[k])
        out[f"cost_at_{pct}"] = float(c[k])
    return out


def reference_points(rs: RoutingSet) -> dict:
    """All-cheap, all-strong, random (straight line) and oracle summaries."""
    lo, hi = rs.c_cheap.mean(), rs.c_strong.mean()
    q_lo, q_hi = rs.q_cheap.mean(), rs.q_strong.mean()
    oracle_scores = rs.q_strong - rs.q_cheap
    return {
        "cheap": {"cost": lo, "quality": q_lo},
        "strong": {"cost": hi, "quality": q_hi},
        "random": {"aiq": (q_lo + q_hi) / 2},
        "oracle": summarize(rs, oracle_scores),
        "n": len(rs),
    }


def length_scores(rs: RoutingSet) -> np.ndarray:
    """Zero-cost heuristic baseline: longer prompts go to the strong model."""
    return np.array([len(p) for p in rs.prompts], dtype=float)


def trained_baseline_scores(train: RoutingSet, test: RoutingSet, seed: int = 0) -> np.ndarray:
    """TF-IDF + logistic regression trained on *dataset labels* of the dev split.

    Label = the strong model beats the cheap one on that prompt. This is an
    in-distribution trained router, a strong baseline for a zero-shot Jev. It is
    trained only on RouterBench's own scores, never on Jev outputs.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    y = (train.q_strong > train.q_cheap).astype(int)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(train.prompts)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X, y)
    return clf.predict_proba(vec.transform(test.prompts))[:, 1]

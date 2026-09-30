"""Turn the decision caches into the study's numbers.

    .venv/Scripts/python.exe jev_study/analyze.py --split dev
    .venv/Scripts/python.exe jev_study/analyze.py --split test

Writes ``results/results_<split>.json`` and ``results/results_<split>.md``.

Study 1 thresholds are always chosen on *dev* negatives (deepset + indirect;
NotInject is kept out because it is the over-defence test) and then applied
unchanged to the split being analysed. On dev this is optimistic by
construction; only test numbers go in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jevbench import metrics as M  # noqa: E402
from jevbench.runner import CACHE_DIR, RESULTS_DIR, read_cache  # noqa: E402

DETECTION_SOURCES = ("deepset", "indirect")
FPRS = (0.01, 0.05)


def _examples() -> list[dict]:
    path = RESULTS_DIR / "injection_examples.jsonl"
    if not path.exists():
        raise SystemExit("Run `run_injection.py --counts` first to write the example list.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fmt(x: float, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Study 1
# ---------------------------------------------------------------------------


def analyze_injection(split: str, cache_root: Path = CACHE_DIR) -> dict:
    examples = _examples()
    out: dict = {}
    for path in sorted((cache_root / "injection").glob("*.jsonl")):
        name = path.stem
        rows = read_cache(path)

        def arrays(pred, _rows=rows):
            ex = [e for e in examples if pred(e) and e["id"] in _rows]
            y = np.array([e["label"] for e in ex], dtype=int)
            s = np.array([_rows[e["id"]]["score"] for e in ex], dtype=float)
            return ex, y, s

        _, _, dev_neg = arrays(lambda e: e["split"] == "dev" and e["label"] == 0
                               and e["source"] in DETECTION_SOURCES)
        thresholds = {f: M.threshold_at_fpr(dev_neg, f) for f in FPRS} if dev_neg.size else {}
        in_split = [e for e in examples if e["split"] == split]
        ex, y, s = arrays(lambda e: e["split"] == split and e["source"] in DETECTION_SOURCES)
        if not ex:
            continue
        r: dict = {
            "coverage": sum(e["id"] in rows for e in in_split) / max(1, len(in_split)),
            "n": len(ex),
            "auroc": M.stratified_bootstrap(y, s, M.auroc),
            "brier": M.brier(y, s),
            "ece": M.ece(y, s),
            "reliability": M.reliability(y, s),
            "at_0.5": M.rates(y, s, 0.5),
        }
        for f, t in thresholds.items():
            r[f"at_dev_fpr_{f}"] = M.rates(y, s, t)
            r[f"oracle_tpr_at_fpr_{f}"] = M.tpr_at_fpr(y, s, f)
        r["by_source"] = {}
        for src in DETECTION_SOURCES:
            _, ys, ss = arrays(lambda e, src=src: e["split"] == split and e["source"] == src)
            if ys.size:
                r["by_source"][src] = {"n": int(ys.size), "auroc": M.auroc(ys, ss),
                                       "at_0.5": M.rates(ys, ss, 0.5)}
        ni_ex, _, ni_s = arrays(lambda e: e["split"] == split and e["source"] == "notinject")
        if ni_ex:
            r["notinject_fpr"] = {"at_0.5": float((ni_s >= 0.5).mean()),
                                  **{f"at_dev_fpr_{f}": float((ni_s >= t).mean())
                                     for f, t in thresholds.items()}}
            by_level: dict[str, list[float]] = {}
            for e, sc in zip(ni_ex, ni_s):
                level = e["category"].split("(")[-1].split()[0]
                by_level.setdefault(level, []).append(float(sc >= 0.5))
            r["notinject_fpr"]["by_trigger_words_at_0.5"] = {
                k: float(np.mean(v)) for k, v in sorted(by_level.items())}
        all_rows = [rows[e["id"]] for e in in_split if e["id"] in rows]
        r["latency_ms"] = M.latency_summary([row["latency_ms"] for row in all_rows])
        r["usd_per_1k"] = M.cost_per_1k([row["cost_usd"] for row in all_rows])
        r["errors"] = sum(bool(row.get("error")) for row in all_rows)
        r["model_versions"] = sorted({row.get("model_version", "") for row in all_rows})
        conf = [(rows[e["id"]].get("confidence"), e) for e in ex]
        if all(c is not None for c, _ in conf):
            correct = [(rows[e["id"]]["score"] >= 0.5) == bool(e["label"]) for _, e in conf]
            r["confidence_ece"] = M.confidence_ece(correct, [c for c, _ in conf])
        out[name] = r

    # Paired AUROC differences: every Jev variant against every other decider.
    ids = [e["id"] for e in examples if e["split"] == split and e["source"] in DETECTION_SOURCES]
    label = {e["id"]: e["label"] for e in examples}
    caches = {p.stem: read_cache(p) for p in (cache_root / "injection").glob("*.jsonl")}
    comps = {}
    for a in [n for n in caches if n.startswith("jev")]:
        for b in [n for n in caches if n != a]:
            common = [i for i in ids if i in caches[a] and i in caches[b]]
            if len(common) < 20:
                continue
            y = [label[i] for i in common]
            comps[f"{a} - {b}"] = M.paired_bootstrap_diff(
                y, [caches[a][i]["score"] for i in common],
                [caches[b][i]["score"] for i in common], M.auroc)
    return {"deciders": out, "auroc_differences": comps}


# ---------------------------------------------------------------------------
# Study 4
# ---------------------------------------------------------------------------


def analyze_routing(split: str, cache_root: Path = CACHE_DIR) -> dict:
    from jevbench import routing as R

    caches = sorted((cache_root / "routing").glob("*.jsonl"))
    rs = R.load_routerbench(split=split)
    label = (rs.q_strong > rs.q_cheap).astype(int)
    out = {"reference": R.reference_points(rs), "routers": {}}

    def add(name, scores, cost=None, extra=None):
        res = R.summarize(rs, scores, cost)
        res["auroc_strong_wins"] = M.auroc(label, scores)
        res.update(extra or {})
        out["routers"][name] = res

    add("length", R.length_scores(rs))
    if split == "test":
        add("tfidf-trained-on-dev", R.trained_baseline_scores(R.load_routerbench(split="dev"), rs))
    else:  # no separate training data on dev: 5-fold cross-fitted scores
        from sklearn.model_selection import KFold

        scores = np.zeros(len(rs))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(rs.ids):
            scores[te] = R.trained_baseline_scores(rs.subset(np.isin(np.arange(len(rs)), tr)),
                                                   rs.subset(np.isin(np.arange(len(rs)), te)))
        add("tfidf-crossfit", scores)
    for path in caches:
        rows = read_cache(path)
        have = np.array([i in rows for i in rs.ids])
        if not have.any():
            continue
        sub = rs if have.all() else rs.subset(have)
        s = np.array([rows[i]["score"] for i in sub.ids])
        c = np.array([rows[i]["cost_usd"] for i in sub.ids])
        res = R.summarize(sub, s, c)
        res["auroc_strong_wins"] = M.auroc((sub.q_strong > sub.q_cheap).astype(int), s)
        res["coverage"] = float(have.mean())
        res["latency_ms"] = M.latency_summary([rows[i]["latency_ms"] for i in sub.ids])
        res["model_versions"] = sorted({rows[i].get("model_version", "") for i in sub.ids})
        out["routers"][path.stem] = res
    return out


# ---------------------------------------------------------------------------


def to_markdown(split: str, inj: dict, rout: dict | None) -> str:
    lines = [f"# Jev study results ({split} split)", ""]
    if split == "dev":
        lines += ["Dev numbers are for choosing wordings and thresholds only.", ""]
    lines += ["## Study 1: prompt-injection screening (deepset + indirect)", "",
              "| Decider | AUROC [95% CI] | TPR @ 5% FPR (dev thr.) | FPR there | "
              "NotInject false alarms @ 0.5 | ECE | p50 / p95 ms | $ / 1k |",
              "|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(inj["deciders"].items()):
        a = r["auroc"]
        op = r.get("at_dev_fpr_0.05", {})
        ni = r.get("notinject_fpr", {}).get("at_0.5", float("nan"))
        lines.append(
            f"| {name} | {_fmt(a[0])} [{_fmt(a[1])}, {_fmt(a[2])}] | "
            f"{_fmt(op.get('tpr', float('nan')), True)} | {_fmt(op.get('fpr', float('nan')), True)} | "
            f"{_fmt(ni, True)} | {_fmt(r['ece'])} | {r['latency_ms']['p50']:.0f} / "
            f"{r['latency_ms']['p95']:.0f} | {r['usd_per_1k']:.4f} |")
    if inj["auroc_differences"]:
        lines += ["", "Paired AUROC differences (95% CI):", ""]
        for k, (d, lo, hi) in sorted(inj["auroc_differences"].items()):
            lines.append(f"- {k}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if rout:
        ref = rout["reference"]
        lines += ["", "## Study 4: routing (Mixtral-8x7B vs GPT-4 on RouterBench)", "",
                  f"n = {ref['n']}; cheap quality {ref['cheap']['quality']:.3f} at "
                  f"${ref['cheap']['cost'] * 1000:.3f}/1k; strong {ref['strong']['quality']:.3f} at "
                  f"${ref['strong']['cost'] * 1000:.3f}/1k; random-routing AIQ "
                  f"{ref['random']['aiq']:.3f}; oracle AIQ {ref['oracle']['aiq']:.3f}.", "",
                  "| Router | AIQ | AUROC (strong wins) | quality @ 40% strong | "
                  "$/1k to reach 95% of strong | router $/1k |",
                  "|---|---|---|---|---|---|"]
        for name, r in rout["routers"].items():
            if r.get("coverage", 1.0) < 1.0:  # summarized on a subset: not comparable
                name = f"{name} (partial, {100 * r['coverage']:.0f}% of prompts)"
            lines.append(f"| {name} | {r['aiq']:.3f} | {_fmt(r['auroc_strong_wins'])} | "
                         f"{r['quality_at_40']:.3f} | {r['cost_to_95'] * 1000:.3f} | "
                         f"{r['router_cost_per_prompt'] * 1000:.5f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--no-routing", action="store_true")
    args = ap.parse_args()
    inj = analyze_injection(args.split)
    rout = None if args.no_routing else analyze_routing(args.split)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"results_{args.split}.json").write_text(
        json.dumps({"injection": inj, "routing": rout}, indent=2, default=float), encoding="utf-8")
    md = to_markdown(args.split, inj, rout)
    (RESULTS_DIR / f"results_{args.split}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

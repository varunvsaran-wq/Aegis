"""Build the paper's figures, tables and number macros from the frozen results.

    .venv/Scripts/python.exe jev_study/analyze.py --split dev
    .venv/Scripts/python.exe jev_study/analyze.py --split test
    .venv/Scripts/python.exe jev_study/build_report.py
    cd jev_study/report && pdflatex jev_report.tex && pdflatex jev_report.tex

Every number in ``report/jev_report.tex`` comes from ``report/numbers.tex`` or a
generated table, never typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jevbench import metrics as M  # noqa: E402
from jevbench import routing as R  # noqa: E402
from jevbench.runner import CACHE_DIR, RESULTS_DIR, read_cache  # noqa: E402

OUT = HERE / "report"

#: decider -> (macro prefix, display name, colour)
SYSTEMS = {
    "jev-s1": ("JevS", "Jev (score question)", "#1b6ca8"),
    "jev-v2": ("JevC", "Jev (choice question)", "#63a4d4"),
    "judge-haiku45": ("Haiku", "Claude Haiku 4.5 judge", "#c0632b"),
    "judge-gpt4omini": ("Mini", "GPT-4o-mini judge", "#e0a05c"),
    "deberta-protectai": ("Deberta", "ProtectAI DeBERTa (local)", "#5c8a4a"),
    "keyword": ("Keyword", "Keyword rules", "#8c8c8c"),
}
DEV_WORDINGS = ("jev-v1", "jev-v2", "jev-v3", "jev-s1")
ROUTERS = {"jev-v1": ("RJev", "Jev (choice question)", "#1b6ca8"),
           "tfidf-trained-on-dev": ("RTfidf", "TF-IDF router trained on dev", "#5c8a4a"),
           "length": ("RLength", "Prompt length", "#8c8c8c")}


def _num(v, digits=3) -> str:
    return f"{v:.{digits}f}"


def _pct(v, digits=1) -> str:
    return f"{100 * v:.{digits}f}"


def _signed(v, digits=3) -> str:
    return f"{v:+.{digits}f}".replace("+", "+")


def _tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def load() -> tuple[dict, dict, list[dict]]:
    test = json.loads((RESULTS_DIR / "results_test.json").read_text(encoding="utf-8"))
    dev = json.loads((RESULTS_DIR / "results_dev.json").read_text(encoding="utf-8"))
    ex = [json.loads(line) for line in
          (RESULTS_DIR / "injection_examples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return test, dev, ex


def split_arrays(name: str, examples: list[dict], split: str, sources) -> tuple:
    rows = read_cache(CACHE_DIR / "injection" / f"{name}.jsonl")
    ex = [e for e in examples if e["split"] == split and e["source"] in sources and e["id"] in rows]
    y = np.array([e["label"] for e in ex])
    s = np.array([rows[e["id"]]["score"] for e in ex])
    return ex, y, s, rows


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def macros(test: dict, dev: dict, examples: list[dict]) -> list[str]:
    m: dict[str, str] = {}
    inj = test["injection"]["deciders"]
    for name, (p, _, _) in SYSTEMS.items():
        r = inj[name]
        a = r["auroc"]
        m[f"{p}Auroc"], m[f"{p}AurocLo"], m[f"{p}AurocHi"] = _num(a[0]), _num(a[1]), _num(a[2])
        for f, tag in ((0.05, "Five"), (0.01, "One")):
            op = r[f"at_dev_fpr_{f}"]
            m[f"{p}Tpr{tag}"], m[f"{p}Fpr{tag}"] = _pct(op["tpr"]), _pct(op["fpr"])
        m[f"{p}TprHalf"] = _pct(r["at_0.5"]["tpr"])
        m[f"{p}FprHalf"] = _pct(r["at_0.5"]["fpr"])
        m[f"{p}NiHalf"] = _pct(r["notinject_fpr"]["at_0.5"])
        m[f"{p}NiFive"] = _pct(r["notinject_fpr"]["at_dev_fpr_0.05"])
        m[f"{p}NiOne"] = _pct(r["notinject_fpr"]["at_dev_fpr_0.01"])
        m[f"{p}Ece"], m[f"{p}Brier"] = _num(r["ece"]), _num(r["brier"])
        m[f"{p}Pfifty"] = f"{r['latency_ms']['p50']:.0f}"
        m[f"{p}Pninetyfive"] = f"{r['latency_ms']['p95']:.0f}"
        m[f"{p}Cost"] = f"{r['usd_per_1k']:.3f}"
        m[f"{p}AurocDeepset"] = _num(r["by_source"]["deepset"]["auroc"])
        m[f"{p}AurocIndirect"] = _num(r["by_source"]["indirect"]["auroc"])
    for name in ("jev-s1", "jev-v2"):
        m[f"{SYSTEMS[name][0]}ConfEce"] = _num(inj[name]["confidence_ece"])

    diffs = test["injection"]["auroc_differences"]
    for key, tag in (("jev-s1 - judge-haiku45", "SHaiku"), ("jev-s1 - judge-gpt4omini", "SMini"),
                     ("jev-s1 - deberta-protectai", "SDeberta"), ("jev-s1 - jev-v2", "SC"),
                     ("jev-v2 - judge-haiku45", "CHaiku"), ("jev-s1 - keyword", "SKeyword")):
        d, lo, hi = diffs[key]
        m[f"Diff{tag}"], m[f"Diff{tag}Lo"], m[f"Diff{tag}Hi"] = _signed(d), _signed(lo), _signed(hi)

    s1, haiku, mini = inj["jev-s1"], inj["judge-haiku45"], inj["judge-gpt4omini"]
    m["SpeedupHaiku"] = f"{haiku['latency_ms']['p50'] / s1['latency_ms']['p50']:.0f}"
    m["SpeedupMini"] = f"{mini['latency_ms']['p50'] / s1['latency_ms']['p50']:.0f}"
    m["CheaperHaiku"] = f"{haiku['usd_per_1k'] / s1['usd_per_1k']:.0f}"
    m["CheaperMini"] = f"{mini['usd_per_1k'] / s1['usd_per_1k']:.0f}"

    # Dataset sizes.
    for split in ("dev", "test"):
        S = split.capitalize()
        for src, tag in (("deepset", "Deepset"), ("indirect", "Indirect"), ("notinject", "Notinject")):
            pos = sum(e["split"] == split and e["source"] == src and e["label"] == 1 for e in examples)
            neg = sum(e["split"] == split and e["source"] == src and e["label"] == 0 for e in examples)
            m[f"N{S}{tag}Pos"], m[f"N{S}{tag}Neg"] = str(pos), str(neg)
        m[f"N{S}Total"] = str(sum(e["split"] == split for e in examples))

    # Jev output shape: how saturated are the choice probabilities?
    _, _, s_c, rows_c = split_arrays("jev-v2", examples, "test", ("deepset", "indirect", "notinject"))
    m["ChoiceSaturated"] = _pct(np.mean((s_c <= 0.01) | (s_c >= 0.99)), 0)
    _, _, s_s, rows_s = split_arrays("jev-s1", examples, "test", ("deepset", "indirect", "notinject"))
    m["ScoreDistinct"] = str(len(np.unique(np.round(s_s, 4))))
    m["ChoiceDistinct"] = str(len(np.unique(np.round(s_c, 4))))
    toks = [r["input_tokens"] for r in rows_s.values()]
    m["JevTokens"] = f"{np.mean(toks):.0f}"
    m["JevVersion"] = inj["jev-s1"]["model_versions"][0]

    # Dev wording comparison.
    ddev = dev["injection"]["deciders"]
    for name, tag in (("jev-v1", "DevVone"), ("jev-v2", "DevVtwo"), ("jev-v3", "DevVthree"),
                      ("jev-s1", "DevSone"), ("judge-haiku45", "DevHaiku")):
        m[f"{tag}Auroc"] = _num(ddev[name]["auroc"][0])

    # Routing.
    rt = test["routing"]
    ref = rt["reference"]
    m["RN"] = str(ref["n"])
    m["RCheapQ"], m["RStrongQ"] = _num(ref["cheap"]["quality"]), _num(ref["strong"]["quality"])
    m["RCheapCost"] = f"{ref['cheap']['cost'] * 1000:.2f}"
    m["RStrongCost"] = f"{ref['strong']['cost'] * 1000:.2f}"
    m["RRandomAiq"], m["ROracleAiq"] = _num(ref["random"]["aiq"]), _num(ref["oracle"]["aiq"])
    for name, (p, _, _) in ROUTERS.items():
        r = rt["routers"][name]
        m[f"{p}Aiq"] = _num(r["aiq"])
        m[f"{p}AurocWin"] = _num(r["auroc_strong_wins"])
        m[f"{p}CostNinetyfive"] = f"{r['cost_to_95'] * 1000:.2f}"
        m[f"{p}QFourty"] = _num(r["quality_at_40"])
    m["RJevRouterCost"] = f"{rt['routers']['jev-v1']['router_cost_per_prompt'] * 1000:.3f}"
    m["RDevJevAiq"] = _num(dev["routing"]["routers"]["jev-v1"]["aiq"])
    m["RDevRandomAiq"] = _num(dev["routing"]["reference"]["random"]["aiq"])

    # Spend.
    total = 0.0
    for p in CACHE_DIR.glob("*/*.jsonl"):
        total += sum(float(r.get("cost_usd", 0)) for r in read_cache(p).values())
    jev_total = sum(float(r.get("cost_usd", 0)) for p in CACHE_DIR.glob("*/jev-*.jsonl")
                    for r in read_cache(p).values())
    jev_calls = sum(len(read_cache(p)) for p in CACHE_DIR.glob("*/jev-*.jsonl"))
    m["TotalSpend"], m["JevSpend"], m["JevCalls"] = f"{total:.2f}", f"{jev_total:.2f}", f"{jev_calls:,}"
    return [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table_injection(test: dict) -> str:
    inj = test["injection"]["deciders"]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"System & AUROC [95\% CI] & \makecell{Detected\\(5\% FPR thr.)} & "
             r"\makecell{NotInject\\false alarms} & ECE & \makecell{p50 / p95\\ms} & "
             r"\makecell{\$ per\\1k checks} \\", r"\midrule"]
    for name, (_, label, _) in SYSTEMS.items():
        r = inj[name]
        a = r["auroc"]
        op = r["at_dev_fpr_0.05"]
        cost = "0 (local)" if r["usd_per_1k"] == 0 else f"{r['usd_per_1k']:.3f}"
        lines.append(
            f"{label} & {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] & {_pct(op['tpr'])}\\% & "
            f"{_pct(r['notinject_fpr']['at_0.5'])}\\% & {r['ece']:.3f} & "
            f"{r['latency_ms']['p50']:.0f} / {r['latency_ms']['p95']:.0f} & {cost} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_dev(dev: dict) -> str:
    d = dev["injection"]["deciders"]
    names = {"jev-v1": "Choice v1 (detailed definition)", "jev-v2": "Choice v2 (``hijack'', names the trigger-word trap)",
             "jev-v3": "Choice v3 (one line)", "jev-s1": "Score s1 (five levels)"}
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Jev wording (dev) & AUROC & NotInject false alarms & Confidence ECE \\", r"\midrule"]
    for n in DEV_WORDINGS:
        r = d[n]
        lines.append(f"{names[n]} & {r['auroc'][0]:.3f} & {_pct(r['notinject_fpr']['at_0.5'])}\\% & "
                     f"{r.get('confidence_ece', float('nan')):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_routing(test: dict) -> str:
    rt = test["routing"]
    ref = rt["reference"]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Router & AIQ & \makecell{AUROC\\(GPT-4 wins)} & \makecell{Quality at\\40\% to GPT-4} & "
             r"\makecell{\$ per 1k prompts to\\reach 95\% of GPT-4} \\", r"\midrule",
             f"Random & {ref['random']['aiq']:.3f} & 0.500 & -- & -- \\\\"]
    for name, (_, label, _) in ROUTERS.items():
        r = rt["routers"][name]
        lines.append(f"{label} & {r['aiq']:.3f} & {r['auroc_strong_wins']:.3f} & "
                     f"{r['quality_at_40']:.3f} & {r['cost_to_95'] * 1000:.2f} \\\\")
    o = ref["oracle"]
    lines += [f"Oracle (knows the answers) & {o['aiq']:.3f} & -- & {o['quality_at_40']:.3f} & "
              f"{o['cost_to_95'] * 1000:.2f} \\\\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_examples(examples: list[dict]) -> str:
    """Jev's test errors at 0.5: every false alarm on NotInject and a few misses."""
    rows = read_cache(CACHE_DIR / "injection" / "jev-s1.jsonl")
    test = [e for e in examples if e["split"] == "test" and e["id"] in rows]
    fa = [e for e in test if e["label"] == 0 and rows[e["id"]]["score"] >= 0.5]
    miss = [e for e in test if e["label"] == 1 and rows[e["id"]]["score"] < 0.5]
    fa.sort(key=lambda e: -rows[e["id"]]["score"])
    miss.sort(key=lambda e: rows[e["id"]]["score"])

    def clip(t, n=150):
        t = " ".join(t.split())
        return _tex_escape(t[:n] + ("..." if len(t) > n else ""))

    lines = [r"\begin{tabular}{p{0.13\linewidth}p{0.08\linewidth}p{0.67\linewidth}}", r"\toprule",
             r"Error & Score & Text (truncated) \\", r"\midrule"]
    for kind, items in (("False alarm", fa[:4]), ("Miss", miss[:4])):
        for e in items:
            if not e["text"].isascii():
                continue
            lines.append(f"{kind} ({_tex_escape(e['source'])}) & {rows[e['id']]['score']:.2f} & "
                         f"{clip(e['text'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", len(fa), len(miss)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _roc(y, s):
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = np.cumsum(1 - y)[distinct]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


def fig_roc(examples):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for name, (_, label, colour) in SYSTEMS.items():
        _, y, s, _ = split_arrays(name, examples, "test", ("deepset", "indirect"))
        fpr, tpr = _roc(y, s)
        for ax in axes:
            ax.plot(fpr, tpr, color=colour, label=label, lw=1.6)
    axes[0].set_title("ROC, full range")
    axes[1].set_xlim(0, 0.1)
    axes[1].set_ylim(0.3, 1.0)
    axes[1].set_title("Low false-alarm region")
    for ax in axes:
        ax.set_xlabel("False-alarm rate")
        ax.set_ylabel("Detection rate")
        ax.grid(alpha=0.3)
    axes[0].plot([0, 1], [0, 1], ":", color="#bbbbbb")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_roc.pdf")
    plt.close(fig)


def fig_tradeoff(test):
    inj = test["injection"]["deciders"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    short = {"jev-s1": ("Jev score", (6, 2)), "jev-v2": ("Jev choice", (6, -10)),
             "judge-haiku45": ("Haiku 4.5", (-12, -12)), "judge-gpt4omini": ("GPT-4o-mini", (-20, 7)),
             "deberta-protectai": ("DeBERTa", (6, -3)), "keyword": ("Keyword rules", (6, -3))}
    for name, (_, _, colour) in SYSTEMS.items():
        r = inj[name]
        lat = max(r["latency_ms"]["p50"], 0.05)
        cost = max(r["usd_per_1k"], 1e-4)
        text, offset = short[name]
        for ax, x in ((axes[0], lat), (axes[1], cost)):
            ax.scatter(x, r["auroc"][0], color=colour, s=50, zorder=3)
            ax.annotate(text, (x, r["auroc"][0]), fontsize=7, xytext=offset,
                        textcoords="offset points")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Median latency per check (ms, log scale)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"\$ per 1,000 checks (log scale; local models drawn at \$0.0001)")
    for ax in axes:
        ax.set_ylabel("AUROC (test)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.pdf")
    plt.close(fig)


def fig_notinject(test):
    inj = test["injection"]["deciders"]
    names = list(SYSTEMS)
    levels = ("one", "two", "three")
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    w = 0.13
    for i, name in enumerate(names):
        vals = [inj[name]["notinject_fpr"]["by_trigger_words_at_0.5"][lv] * 100 for lv in levels]
        ax.bar(np.arange(3) + (i - 2.5) * w, vals, w, color=SYSTEMS[name][2], label=SYSTEMS[name][1])
    ax.set_xticks(range(3), ["1 trigger word", "2 trigger words", "3 trigger words"])
    ax.set_ylabel("False alarms (%)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_notinject.pdf")
    plt.close(fig)


def fig_reliability(examples):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], ":", color="#999999")
    for name in ("jev-s1", "jev-v2", "judge-haiku45", "judge-gpt4omini"):
        _, y, s, _ = split_arrays(name, examples, "test", ("deepset", "indirect"))
        bins = M.reliability(y, s, n_bins=5)
        ax.plot([b["mean_p"] for b in bins], [b["freq"] for b in bins], "o-",
                color=SYSTEMS[name][2], label=SYSTEMS[name][1], ms=4)
    ax.set_xlabel("Predicted P(injection)")
    ax.set_ylabel("Observed share of injections")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reliability.pdf")
    plt.close(fig)


def fig_routing(test):
    rs = R.load_routerbench(split="test")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lo, hi = rs.c_cheap.mean() * 1000, rs.c_strong.mean() * 1000
    ax.plot([lo, hi], [rs.q_cheap.mean(), rs.q_strong.mean()], ":", color="#999999", label="Random")
    _, c, q = R.curve(rs, rs.q_strong - rs.q_cheap)
    ax.plot(c * 1000, q, color="black", lw=1, label="Oracle")
    rows = read_cache(CACHE_DIR / "routing" / "jev-v1.jsonl")
    jev = np.array([rows[i]["score"] for i in rs.ids])
    jc = np.array([rows[i]["cost_usd"] for i in rs.ids])
    tf = R.trained_baseline_scores(R.load_routerbench(split="dev"), rs)
    for scores, cost, name in ((jev, jc, "jev-v1"), (tf, None, "tfidf-trained-on-dev"),
                               (R.length_scores(rs), None, "length")):
        _, c, q = R.curve(rs, scores, cost)
        ax.plot(c * 1000, q, color=ROUTERS[name][2], lw=1.6, label=ROUTERS[name][1])
    ax.set_xlabel(r"\$ per 1,000 prompts")
    ax.set_ylabel("Mean quality")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_routing.pdf")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test, dev, examples = load()
    lines = macros(test, dev, examples)
    ex_table, n_fa, n_miss = table_examples(examples)
    lines += [f"\\newcommand{{\\JevSFalseAlarms}}{{{n_fa}}}", f"\\newcommand{{\\JevSMisses}}{{{n_miss}}}"]
    (OUT / "numbers.tex").write_text("% Generated by build_report.py\n" + "\n".join(lines) + "\n",
                                     encoding="utf-8")
    (OUT / "tab_injection.tex").write_text(table_injection(test), encoding="utf-8")
    (OUT / "tab_dev.tex").write_text(table_dev(dev), encoding="utf-8")
    (OUT / "tab_routing.tex").write_text(table_routing(test), encoding="utf-8")
    (OUT / "tab_examples.tex").write_text(ex_table, encoding="utf-8")
    fig_roc(examples)
    fig_tradeoff(test)
    fig_notinject(test)
    fig_reliability(examples)
    fig_routing(test)
    print(f"wrote {len(lines)} macros, 4 tables, 5 figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

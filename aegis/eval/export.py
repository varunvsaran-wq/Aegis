"""Export Phase 1/2 results from MLflow into paper-ready LaTeX tables and figures.

All heavy dependencies (mlflow, pandas, matplotlib) are imported lazily so
this module can be imported without them installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aegis.config import get_config

logger = logging.getLogger(__name__)

#: Fixed categorical color per mode (validated CVD-safe ordering, light surface).
_MODE_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
]


def _fmt_pct(mean, std) -> str:
    """Format a fraction metric as '71.0 ± 1.2' (percent, mean ± std)."""
    import math

    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{100 * mean:.1f}"
    return f"{100 * mean:.1f} ± {100 * std:.1f}"


def _tex_escape(value) -> str:
    """Escape LaTeX special characters in a plain-text cell value."""
    s = str(value)
    s = s.replace("\\", "\x00")
    for old, new in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        s = s.replace(old, new)
    return s.replace("\x00", r"\textbackslash{}")


def _booktabs_tabular(df, escape_header: bool = True) -> str:
    """Render a DataFrame as a booktabs tabular (cells LaTeX-escaped).

    ``escape_header=False`` lets callers pass headers that already contain
    LaTeX markup (e.g. math mode) verbatim.
    """
    cols = list(df.columns)
    header = " & ".join(
        (_tex_escape(c) if escape_header else str(c)) for c in cols
    )
    lines = [
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        header + r" \\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_tex_escape(v) for v in row) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _wrap_table(tabular: str, caption: str, label: str) -> str:
    """Wrap a tabular block in a table environment with caption and label."""
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"{tabular.strip()}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def export_phase1(
    experiment: str = "aegis-phase1", report_dir: Path | None = None
) -> list[Path]:
    """Export accuracy/cost tables and the accuracy figure for Phase 1.

    Reads all runs of ``experiment`` from MLflow, aggregates across seeds per
    (model, mode), and writes ``phase1_accuracy.tex``, ``phase1_accuracy.pdf``
    and ``phase1_cost.tex`` into ``report_dir`` (default: config.report_dir).
    Returns the list of written paths. Raises ``RuntimeError`` if the
    experiment has no runs.
    """
    import mlflow
    import numpy as np
    import pandas as pd

    config = get_config()
    report_dir = Path(report_dir) if report_dir is not None else config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs is None or len(runs) == 0:
        raise RuntimeError(
            f"No MLflow runs found for experiment {experiment!r} "
            f"(tracking URI: {config.mlflow_tracking_uri}). "
            "Run 'aegis eval' or 'aegis sweep' first."
        )

    df = pd.DataFrame(runs)
    required = ["params.model", "params.mode"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Runs in experiment {experiment!r} are missing params {missing}; "
            "were they logged by aegis?"
        )

    def _metric(name: str) -> pd.Series:
        col = f"metrics.{name}"
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    work = pd.DataFrame(
        {
            "model": df["params.model"],
            "mode": df["params.mode"],
            "em": _metric("em_mean"),
            "f1": _metric("f1_mean"),
            "judge": _metric("judge_mean"),
            "em_ci_lo": _metric("em_ci_lo"),
            "em_ci_hi": _metric("em_ci_hi"),
            "f1_ci_lo": _metric("f1_ci_lo"),
            "f1_ci_hi": _metric("f1_ci_hi"),
            "cost": _metric("mean_cost_usd"),
            "latency": _metric("mean_latency_s"),
        }
    )

    agg = (
        work.groupby(["model", "mode"])
        .agg(
            em_mean=("em", "mean"),
            em_std=("em", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            judge_mean=("judge", "mean"),
            judge_std=("judge", "std"),
            em_ci_lo=("em_ci_lo", "mean"),
            em_ci_hi=("em_ci_hi", "mean"),
            f1_ci_lo=("f1_ci_lo", "mean"),
            f1_ci_hi=("f1_ci_hi", "mean"),
            cost_mean=("cost", "mean"),
            latency_mean=("latency", "mean"),
            n_seeds=("em", "count"),
        )
        .reset_index()
    )

    written: list[Path] = []

    # ------------------------------------------------------------------
    # 1. Accuracy table (phase1_accuracy.tex)
    # ------------------------------------------------------------------
    acc_rows = []
    have_em_ci = agg["em_ci_lo"].notna().any() and agg["em_ci_hi"].notna().any()
    for _, r in agg.iterrows():
        row = {
            "Model": r["model"],
            "Mode": r["mode"],
            "EM": _fmt_pct(r["em_mean"], r["em_std"]),
            "F1": _fmt_pct(r["f1_mean"], r["f1_std"]),
            "Judge": _fmt_pct(r["judge_mean"], r["judge_std"]),
            "Seeds": int(r["n_seeds"]),
        }
        if have_em_ci and pd.notna(r["em_ci_lo"]) and pd.notna(r["em_ci_hi"]):
            row["EM 95% CI"] = (
                f"[{100 * r['em_ci_lo']:.1f}, {100 * r['em_ci_hi']:.1f}]"
            )
        elif have_em_ci:
            row["EM 95% CI"] = "--"
        acc_rows.append(row)
    acc_df = pd.DataFrame(acc_rows)

    acc_tabular = acc_df.to_latex(index=False, escape=True)
    acc_tex = _wrap_table(
        acc_tabular,
        caption=(
            "Phase 1 accuracy on HotpotQA. EM, F1 and judge scores are "
            "percentages, mean $\\pm$ std across seeds; the EM CI is the "
            "seed-averaged 95\\% bootstrap interval."
        ),
        label="tab:phase1-accuracy",
    )
    acc_path = report_dir / "phase1_accuracy.tex"
    acc_path.write_text(acc_tex, encoding="utf-8")
    written.append(acc_path)

    # ------------------------------------------------------------------
    # 2. Accuracy figure (phase1_accuracy.pdf) — two panels (EM, F1),
    #    grouped bars by mode, std error bars. One y-axis per panel.
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(agg["model"].unique())
    modes = sorted(agg["mode"].unique())
    x = np.arange(len(models))
    n_modes = max(len(modes), 1)
    width = min(0.8 / n_modes, 0.25)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, metric, title in ((axes[0], "em", "Exact Match"), (axes[1], "f1", "F1")):
        for j, mode in enumerate(modes):
            means, stds = [], []
            for model in models:
                sel = agg[(agg["model"] == model) & (agg["mode"] == mode)]
                if len(sel):
                    means.append(100 * float(sel[f"{metric}_mean"].iloc[0]))
                    std = sel[f"{metric}_std"].iloc[0]
                    stds.append(100 * float(std) if pd.notna(std) else 0.0)
                else:
                    means.append(0.0)
                    stds.append(0.0)
            offset = (j - (n_modes - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width * 0.92,
                yerr=stds,
                capsize=2,
                color=_MODE_COLORS[j % len(_MODE_COLORS)],
                edgecolor="white",
                linewidth=0.5,
                error_kw={"linewidth": 0.8, "ecolor": "#52514e"},
                label=mode if metric == "em" else None,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=9)
    axes[0].set_ylabel("Score (%)", fontsize=9)
    if len(modes) >= 2:
        axes[0].legend(
            title=None, fontsize=8, frameon=False, loc="upper left", ncols=1
        )
    fig.tight_layout()
    pdf_path = report_dir / "phase1_accuracy.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    written.append(pdf_path)

    # ------------------------------------------------------------------
    # 3. Cost table (phase1_cost.tex)
    # ------------------------------------------------------------------
    cost_rows = []
    for _, r in agg.iterrows():
        cost = r["cost_mean"]
        cost_rows.append(
            {
                "Model": r["model"],
                "Mode": r["mode"],
                "Cost / query (USD)": f"{cost:.5f}" if pd.notna(cost) else "--",
                "Cost / 1k queries (USD)": (
                    f"{1000 * cost:.2f}" if pd.notna(cost) else "--"
                ),
                "Latency (s)": (
                    f"{r['latency_mean']:.2f}" if pd.notna(r["latency_mean"]) else "--"
                ),
            }
        )
    cost_df = pd.DataFrame(cost_rows)
    cost_tabular = cost_df.to_latex(index=False, escape=True)
    cost_tex = _wrap_table(
        cost_tabular,
        caption=(
            "Phase 1 inference cost and latency per (model, mode), "
            "averaged across seeds."
        ),
        label="tab:phase1-cost",
    )
    cost_path = report_dir / "phase1_cost.tex"
    cost_path.write_text(cost_tex, encoding="utf-8")
    written.append(cost_path)

    return written


# ---------------------------------------------------------------------------
# Phase 2: Pareto frontier, results table, and paired significance.
# ---------------------------------------------------------------------------

_PARETO_MARKERS = ["o", "s", "^", "D", "v", "P"]


def export_phase2(
    experiment: str = "aegis-phase2", report_dir: Path | None = None
) -> list[Path]:
    """Export the Phase 2 cost/accuracy Pareto figure and the results table.

    Reads all runs of ``experiment`` from MLflow, aggregates across seeds per
    (model, mode), and writes ``phase2_pareto.pdf`` and ``phase2_results.tex``
    into ``report_dir`` (default: config.report_dir). Returns the list of
    written paths. Raises ``RuntimeError`` if the experiment has no runs.
    """
    import mlflow
    import numpy as np
    import pandas as pd

    config = get_config()
    report_dir = Path(report_dir) if report_dir is not None else config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs is None or len(runs) == 0:
        raise RuntimeError(
            f"No MLflow runs found for experiment {experiment!r} "
            f"(tracking URI: {config.mlflow_tracking_uri}). "
            "Run 'aegis eval' or 'aegis sweep' first."
        )

    df = pd.DataFrame(runs)
    required = ["params.model", "params.mode"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Runs in experiment {experiment!r} are missing params {missing}; "
            "were they logged by aegis?"
        )

    def _metric(name: str) -> pd.Series:
        col = f"metrics.{name}"
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    work = pd.DataFrame(
        {
            "model": df["params.model"],
            "mode": df["params.mode"],
            "em": _metric("em_mean"),
            "f1": _metric("f1_mean"),
            "agreement": _metric("agreement_mean"),
            "abstain": _metric("abstain_rate"),
            "cost": _metric("mean_cost_usd"),
            "latency": _metric("mean_latency_s"),
        }
    )
    agg = (
        work.groupby(["model", "mode"])
        .agg(
            em_mean=("em", "mean"),
            em_std=("em", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            abstain_mean=("abstain", "mean"),
            abstain_std=("abstain", "std"),
            cost_mean=("cost", "mean"),
            latency_mean=("latency", "mean"),
            n_seeds=("em", "count"),
        )
        .reset_index()
    )

    written: list[Path] = []

    # ------------------------------------------------------------------
    # 1. Pareto scatter (phase2_pareto.pdf): cost per 1k queries (log x)
    #    vs EM (%), one point per (model, mode) averaged over seeds.
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted(agg["mode"].unique())
    cost1k = 1000.0 * agg["cost_mean"]
    positive = cost1k[cost1k.notna() & (cost1k > 0)]
    # Zero-cost (local) runs are pinned at the axis minimum on the log axis.
    axis_min = float(positive.min()) / 2.0 if len(positive) else 1e-2

    def _xy(row):
        """(x, y, is_zero_cost) for one aggregated point, or None."""
        cost, em = row["cost_mean"], row["em_mean"]
        if pd.isna(cost) or pd.isna(em):
            return None
        c1k = 1000.0 * float(cost)
        if c1k > 0:
            return (c1k, 100.0 * float(em), False)
        return (axis_min, 100.0 * float(em), True)

    fig, ax = plt.subplots(figsize=(5.8, 3.9))

    # Light connecting line from each model's raw_rag point to its
    # harnessed point (drawn first so markers sit on top).
    for model in sorted(agg["model"].unique()):
        pts = {}
        for _, r in agg[agg["model"] == model].iterrows():
            p = _xy(r)
            if p is not None:
                pts[r["mode"]] = p
        if "raw_rag" in pts and "harnessed" in pts:
            (x0, y0, _), (x1, y1, _) = pts["raw_rag"], pts["harnessed"]
            ax.plot([x0, x1], [y0, y1], color="#c3c2b7", linewidth=0.9, zorder=1)

    for j, mode in enumerate(modes):
        xs, ys, annots = [], [], []
        for _, r in agg[agg["mode"] == mode].iterrows():
            p = _xy(r)
            if p is None:
                continue
            xs.append(p[0])
            ys.append(p[1])
            annots.append((r["model"], p[2]))
        ax.scatter(
            xs,
            ys,
            marker=_PARETO_MARKERS[j % len(_PARETO_MARKERS)],
            color=_MODE_COLORS[j % len(_MODE_COLORS)],
            s=38,
            edgecolor="white",
            linewidth=0.5,
            label=mode,
            zorder=3,
        )
        for x, y, (model, is_zero) in zip(xs, ys, annots):
            ax.annotate(
                model,
                (x, y),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color="#52514e",
            )
            if is_zero:
                ax.annotate(
                    "\\$0 (local)",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, -10),
                    fontsize=7,
                    color="#52514e",
                )

    ax.set_xscale("log")
    ax.set_xlim(left=axis_min / 1.5)
    ax.set_xlabel("Cost per 1k queries (USD, log scale)", fontsize=9)
    ax.set_ylabel("Exact match (%)", fontsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    pareto_path = report_dir / "phase2_pareto.pdf"
    fig.savefig(pareto_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    written.append(pareto_path)

    # ------------------------------------------------------------------
    # 2. Results table (phase2_results.tex).
    # ------------------------------------------------------------------
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            {
                "Model": r["model"],
                "Mode": r["mode"],
                "EM": _fmt_pct(r["em_mean"], r["em_std"]),
                "F1": _fmt_pct(r["f1_mean"], r["f1_std"]),
                "Agreement": _fmt_pct(r["agreement_mean"], r["agreement_std"]),
                "Abstain": _fmt_pct(r["abstain_mean"], r["abstain_std"]),
                "Cost/1k (USD)": (
                    f"{1000 * r['cost_mean']:.2f}" if pd.notna(r["cost_mean"]) else "--"
                ),
                "Latency (s)": (
                    f"{r['latency_mean']:.2f}" if pd.notna(r["latency_mean"]) else "--"
                ),
            }
        )
    results_tex = _wrap_table(
        _booktabs_tabular(pd.DataFrame(rows)),
        caption=(
            "Phase 2 results on HotpotQA per (model, mode), averaged across "
            "seeds. EM, F1, self-consistency agreement and abstain rate are "
            "percentages (mean $\\pm$ std across seeds); cost is USD per "
            "1{,}000 queries; latency is mean seconds per query."
        ),
        label="tab:phase2-results",
    )
    results_path = report_dir / "phase2_results.tex"
    results_path.write_text(results_tex, encoding="utf-8")
    written.append(results_path)

    return written


def export_phase4_ablation(
    experiment: str = "aegis-phase4",
    model: str | None = None,
    report_dir: Path | None = None,
) -> Path:
    """Export the component-ablation table for one model.

    Each ablated run is tagged by the ``ablation`` param (``full``,
    ``no_structurer``, ``no_verifier``, ``no_reranker`` ...). This pairs each
    ablation against the ``full`` harness per question (pooled across seeds),
    reports the EM/F1 drop with a paired-bootstrap CI, a per-component
    significance test (McNemar on EM, Wilcoxon on F1), and applies a
    Benjamini--Hochberg correction across the ablation family. Writes
    ``phase4_ablation.tex`` and returns its path.
    """
    import mlflow
    import pandas as pd

    from aegis.eval import stats

    config = get_config()
    report_dir = Path(report_dir) if report_dir is not None else config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs is None or len(runs) == 0:
        raise RuntimeError(f"No MLflow runs found for experiment {experiment!r}.")
    df = pd.DataFrame(runs)
    if "params.ablation" not in df.columns:
        raise RuntimeError(
            f"Runs in {experiment!r} have no 'ablation' param; run "
            "'aegis ablate' first."
        )
    df = df[df["status"] == "FINISHED"]
    if model is not None:
        df = df[df["params.model"] == model]
    if model is None:
        avail = sorted(df["params.model"].dropna().unique())
        if not avail:
            raise RuntimeError(f"No finished runs in {experiment!r}.")
        model = avail[0]
        df = df[df["params.model"] == model]

    def _runs_by_seed(ablation: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for _, r in df[df["params.ablation"] == ablation].iterrows():
            seed = r.get("params.seed")
            if seed is None or (isinstance(seed, float) and pd.isna(seed)):
                continue
            out.setdefault(str(seed), r["run_id"])
        return out

    full_runs = _runs_by_seed("full")
    if not full_runs:
        raise RuntimeError(f"No 'full' ablation baseline for model {model!r}.")

    ablations = sorted(
        a for a in df["params.ablation"].dropna().unique() if a != "full"
    )
    if not ablations:
        raise RuntimeError(f"Only the 'full' run exists for model {model!r}.")

    import json

    def _rows(run_id: str) -> dict[str, dict]:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="results.jsonl"
        )
        out: dict[str, dict] = {}
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    out[rec["question_id"]] = rec
        return out

    records: list[dict] = []
    for ablation in ablations:
        abl_runs = _runs_by_seed(ablation)
        seeds = sorted(set(full_runs) & set(abl_runs))
        em_full, em_abl, f1_full, f1_abl = [], [], [], []
        for seed in seeds:
            rf, rab = _rows(full_runs[seed]), _rows(abl_runs[seed])
            for qid in sorted(set(rf) & set(rab)):
                a, b = rf[qid], rab[qid]
                if any(x.get(k) is None for x in (a, b) for k in ("em", "f1")):
                    continue
                em_full.append(float(a["em"]))
                em_abl.append(float(b["em"]))
                f1_full.append(float(a["f1"]))
                f1_abl.append(float(b["f1"]))
        if not em_full:
            continue
        n = len(em_full)
        # delta = full - ablated (how much removing the component costs).
        em_boot = stats.paired_bootstrap_delta(em_full, em_abl)
        f1_boot = stats.paired_bootstrap_delta(f1_full, f1_abl)
        mcnemar = stats.mcnemar_test(
            [v >= 0.5 for v in em_full], [v >= 0.5 for v in em_abl]
        )
        wilcoxon = stats.wilcoxon_paired(f1_full, f1_abl)
        records.append(
            {
                "ablation": ablation,
                "n": n,
                "em_full": sum(em_full) / n,
                "em_abl": sum(em_abl) / n,
                "em_delta": em_boot["delta"],
                "em_ci": (em_boot["lo"], em_boot["hi"]),
                "f1_delta": f1_boot["delta"],
                "f1_ci": (f1_boot["lo"], f1_boot["hi"]),
                "mcnemar_p": mcnemar["p_value"],
                "wilcoxon_p": wilcoxon["p_value"],
            }
        )
    if not records:
        raise RuntimeError(f"No paired ablation questions for model {model!r}.")

    mc_adj = stats.bh_correction([r["mcnemar_p"] for r in records])["p_adjusted"]
    wx_adj = stats.bh_correction([r["wilcoxon_p"] for r in records])["p_adjusted"]

    def _fmt_p(p: float) -> str:
        return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"

    def _fmt_delta(delta: float, ci: tuple[float, float]) -> str:
        return f"{100 * delta:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"

    _pretty = {
        "no_structurer": "$-$ structurer",
        "no_verifier": "$-$ verifier",
        "no_reranker": "$-$ reranker",
        "no_defense": "$-$ defense",
    }
    rows = []
    for r, mp, wp in zip(records, mc_adj, wx_adj):
        rows.append(
            {
                "Ablation": _pretty.get(r["ablation"], r["ablation"].replace("_", " ")),
                "EM": f"{100 * r['em_abl']:.1f}",
                "$\\Delta$EM [95\\% CI]": _fmt_delta(r["em_delta"], r["em_ci"]),
                "McNemar $p$ (BH)": _fmt_p(mp),
                "$\\Delta$F1 [95\\% CI]": _fmt_delta(r["f1_delta"], r["f1_ci"]),
                "Wilcoxon $p$ (BH)": _fmt_p(wp),
                "$n$": str(r["n"]),
            }
        )
    full_em = 100 * records[0]["em_full"]
    tex = _wrap_table(
        _booktabs_tabular(pd.DataFrame(rows), escape_header=False),
        caption=(
            f"Component ablations for the harnessed pipeline (model "
            f"\\texttt{{{_tex_escape(model)}}}, full-harness EM "
            f"{full_em:.1f}\\%). Each row removes one component; $\\Delta$ is "
            "the full-minus-ablated change in percentage points with 95\\% "
            "paired-bootstrap CIs. A positive $\\Delta$ means the component "
            "helps. McNemar (EM) and Wilcoxon (F1) $p$-values are "
            "Benjamini--Hochberg adjusted across the ablation family."
        ),
        label="tab:phase4-ablation",
    )
    out_path = report_dir / "phase4_ablation.tex"
    out_path.write_text(tex, encoding="utf-8")
    return out_path


def compare_modes(
    experiment: str,
    model: str,
    mode_a: str = "raw_rag",
    mode_b: str = "harnessed",
) -> dict:
    """Paired per-question comparison of two modes for one model.

    Finds all finished runs of ``model`` in ``experiment`` for each mode,
    pairs runs by seed (seeds lacking either side are skipped), downloads
    each run's ``results.jsonl`` artifact, matches rows by ``question_id``
    within each seed pair, and pools the pairs across seeds.

    Statistics are oriented so a positive delta means ``mode_b`` (the
    harness) is better: ``paired_bootstrap_delta`` is called with
    ``a=mode_b`` values and ``b=mode_a`` values.
    """
    import json

    import mlflow
    import pandas as pd

    from aegis.eval import stats

    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs is None or len(runs) == 0:
        raise RuntimeError(f"No MLflow runs found for experiment {experiment!r}.")

    df = pd.DataFrame(runs)
    for col in ("params.model", "params.mode", "params.seed"):
        if col not in df.columns:
            raise RuntimeError(
                f"Runs in experiment {experiment!r} are missing {col!r}; "
                "were they logged by aegis?"
            )
    df = df[(df["status"] == "FINISHED") & (df["params.model"] == model)]

    def _runs_by_seed(mode: str) -> dict[str, str]:
        """seed -> run_id (most recent run wins; search_runs is newest-first)."""
        out: dict[str, str] = {}
        for _, r in df[df["params.mode"] == mode].iterrows():
            seed = r["params.seed"]
            if seed is None or (isinstance(seed, float) and pd.isna(seed)):
                continue
            out.setdefault(str(seed), r["run_id"])
        return out

    runs_a = _runs_by_seed(mode_a)
    runs_b = _runs_by_seed(mode_b)
    seeds = sorted(set(runs_a) & set(runs_b))
    if not seeds:
        raise RuntimeError(
            f"No seed has finished runs for both {mode_a!r} and {mode_b!r} "
            f"(model {model!r}, experiment {experiment!r})."
        )

    def _rows_by_qid(run_id: str) -> dict[str, dict]:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="results.jsonl"
        )
        rows: dict[str, dict] = {}
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rows[rec["question_id"]] = rec
        return rows

    em_a, em_b, f1_a, f1_b = [], [], [], []
    for seed in seeds:
        rows_a = _rows_by_qid(runs_a[seed])
        rows_b = _rows_by_qid(runs_b[seed])
        for qid in sorted(set(rows_a) & set(rows_b)):
            ra, rb = rows_a[qid], rows_b[qid]
            if any(r.get(k) is None for r in (ra, rb) for k in ("em", "f1")):
                continue
            em_a.append(float(ra["em"]))
            em_b.append(float(rb["em"]))
            f1_a.append(float(ra["f1"]))
            f1_b.append(float(rb["f1"]))

    if not em_a:
        raise RuntimeError(
            f"No paired questions with scores found for model {model!r} "
            f"({mode_a!r} vs {mode_b!r})."
        )

    n = len(em_a)
    em_boot = stats.paired_bootstrap_delta(em_b, em_a)
    f1_boot = stats.paired_bootstrap_delta(f1_b, f1_a)
    mcnemar = stats.mcnemar_test([v >= 0.5 for v in em_b], [v >= 0.5 for v in em_a])
    wilcoxon = stats.wilcoxon_paired(f1_b, f1_a)

    return {
        "model": model,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "n_pairs": n,
        "em_a": sum(em_a) / n,
        "em_b": sum(em_b) / n,
        "em_delta": em_boot["delta"],
        "em_ci": (em_boot["lo"], em_boot["hi"]),
        "mcnemar_p": mcnemar["p_value"],
        "f1_a": sum(f1_a) / n,
        "f1_b": sum(f1_b) / n,
        "f1_delta": f1_boot["delta"],
        "f1_ci": (f1_boot["lo"], f1_boot["hi"]),
        "wilcoxon_p": wilcoxon["p_value"],
    }


def compare_injection(experiment: str, model: str) -> dict:
    """Paired per-question ASR comparison of raw_rag vs harnessed for one model.

    Mirrors :func:`compare_modes` but pairs on ``injection_success`` (overall
    and per attack category), pooling questions across seeds. Returns per-mode
    ASR, the harnessed-minus-raw delta with a paired-bootstrap CI, a McNemar
    p-value, and summed ``(k, n)`` success/attempt counts per category for
    Wilson intervals.
    """
    import json

    import mlflow
    import pandas as pd

    from aegis.eval import stats
    from aegis.eval.poison import ATTACK_CATEGORIES

    mode_a, mode_b = "raw_rag", "harnessed"
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs is None or len(runs) == 0:
        raise RuntimeError(f"No MLflow runs found for experiment {experiment!r}.")

    df = pd.DataFrame(runs)
    for col in ("params.model", "params.mode", "params.seed"):
        if col not in df.columns:
            raise RuntimeError(f"Runs in {experiment!r} missing {col!r}.")
    df = df[(df["status"] == "FINISHED") & (df["params.model"] == model)]

    def _runs_by_seed(mode: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for _, r in df[df["params.mode"] == mode].iterrows():
            seed = r["params.seed"]
            if seed is None or (isinstance(seed, float) and pd.isna(seed)):
                continue
            out.setdefault(str(seed), r["run_id"])
        return out

    runs_a, runs_b = _runs_by_seed(mode_a), _runs_by_seed(mode_b)
    seeds = sorted(set(runs_a) & set(runs_b))
    if not seeds:
        raise RuntimeError(
            f"No seed has both {mode_a!r} and {mode_b!r} runs for model {model!r}."
        )

    def _rows(run_id: str) -> dict[str, dict]:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="results.jsonl"
        )
        out: dict[str, dict] = {}
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    out[rec["question_id"]] = rec
        return out

    succ_a, succ_b = [], []
    cat_k = {c: {"a": 0, "b": 0} for c in ATTACK_CATEGORIES}
    cat_n = {c: {"a": 0, "b": 0} for c in ATTACK_CATEGORIES}
    for seed in seeds:
        ra, rb = _rows(runs_a[seed]), _rows(runs_b[seed])
        for qid in sorted(set(ra) & set(rb)):
            xa, xb = ra[qid], rb[qid]
            if "injection_success" not in xa or "injection_success" not in xb:
                continue
            succ_a.append(1.0 if xa["injection_success"] else 0.0)
            succ_b.append(1.0 if xb["injection_success"] else 0.0)
            for side, rec in (("a", xa), ("b", xb)):
                attacked = rec.get("injection_category_attacked", {})
                success = rec.get("injection_category_success", {})
                for c in ATTACK_CATEGORIES:
                    if attacked.get(c):
                        cat_n[c][side] += 1
                        if success.get(c):
                            cat_k[c][side] += 1

    if not succ_a:
        raise RuntimeError(f"No paired injection rows for model {model!r}.")

    n = len(succ_a)
    boot = stats.paired_bootstrap_delta(succ_b, succ_a)
    mcnemar = stats.mcnemar_test(
        [v >= 0.5 for v in succ_b], [v >= 0.5 for v in succ_a]
    )
    return {
        "model": model,
        "n_pairs": n,
        "asr_raw": sum(succ_a) / n,
        "asr_harnessed": sum(succ_b) / n,
        "asr_delta": boot["delta"],
        "asr_ci": (boot["lo"], boot["hi"]),
        "mcnemar_p": mcnemar["p_value"],
        "cat_k": cat_k,
        "cat_n": cat_n,
    }


def export_phase3(
    experiment: str = "aegis-phase3",
    models: list[str] | None = None,
    report_dir: Path | None = None,
) -> list[Path]:
    """Export the Phase 3 injection results table and per-category ASR figure.

    Writes ``phase3_injection.tex`` (per-model raw-vs-harnessed ASR with paired
    bootstrap CIs and BH-adjusted McNemar p-values) and
    ``phase3_categories.pdf`` (grouped raw-vs-harnessed ASR bars per attack
    category with Wilson 95% intervals, pooled across models) into
    ``report_dir``. Raises ``RuntimeError`` if the experiment has no poisoned
    runs for any model in both modes.
    """
    import mlflow
    import numpy as np
    import pandas as pd

    from aegis.eval import stats
    from aegis.eval.poison import ATTACK_CATEGORIES

    config = get_config()
    report_dir = Path(report_dir) if report_dir is not None else config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    if models is None:
        runs = mlflow.search_runs(experiment_names=[experiment])
        if runs is None or len(runs) == 0:
            raise RuntimeError(f"No MLflow runs found for experiment {experiment!r}.")
        df = pd.DataFrame(runs)
        models = []
        if {"params.model", "params.mode"} <= set(df.columns):
            finished = df[df["status"] == "FINISHED"]
            for m in sorted(finished["params.model"].dropna().unique()):
                have = set(finished[finished["params.model"] == m]["params.mode"])
                if "raw_rag" in have and "harnessed" in have:
                    models.append(m)

    comparisons: list[dict] = []
    for m in models:
        try:
            comparisons.append(compare_injection(experiment, m))
        except RuntimeError:
            logger.warning("Skipping model %r for Phase 3 export", m, exc_info=True)
    if not comparisons:
        raise RuntimeError(
            f"No model in {experiment!r} has poisoned raw_rag+harnessed runs."
        )

    written: list[Path] = []
    mc_adj = stats.bh_correction([c["mcnemar_p"] for c in comparisons])["p_adjusted"]

    # ------------------------------------------------------------------
    # 1. Injection results table.
    # ------------------------------------------------------------------
    def _fmt_p(p: float) -> str:
        return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"

    rows = []
    for c, mp in zip(comparisons, mc_adj):
        rows.append(
            {
                "Model": c["model"],
                "ASR raw": f"{100 * c['asr_raw']:.1f}",
                "ASR harnessed": f"{100 * c['asr_harnessed']:.1f}",
                "$\\Delta$ASR [95\\% CI]": (
                    f"{100 * c['asr_delta']:+.1f} "
                    f"[{100 * c['asr_ci'][0]:+.1f}, {100 * c['asr_ci'][1]:+.1f}]"
                ),
                "McNemar $p$ (BH)": _fmt_p(mp),
                "$n$": str(c["n_pairs"]),
            }
        )
    tex = _wrap_table(
        _booktabs_tabular(pd.DataFrame(rows), escape_header=False),
        caption=(
            "Prompt-injection attack success rate (ASR) on the poisoned "
            "HotpotQA corpus, raw RAG vs. the harnessed pipeline (pooled across "
            "seeds). $\\Delta$ASR is percentage points with a 95\\% paired-"
            "bootstrap CI; McNemar $p$-values are Benjamini--Hochberg adjusted "
            "across models. Lower ASR is better."
        ),
        label="tab:phase3-injection",
    )
    inj_path = report_dir / "phase3_injection.tex"
    inj_path.write_text(tex, encoding="utf-8")
    written.append(inj_path)

    # ------------------------------------------------------------------
    # 2. Per-category ASR figure (raw vs harnessed) with Wilson intervals,
    #    pooling k/n across all models.
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled_k = {c: {"a": 0, "b": 0} for c in ATTACK_CATEGORIES}
    pooled_n = {c: {"a": 0, "b": 0} for c in ATTACK_CATEGORIES}
    for c in comparisons:
        for cat in ATTACK_CATEGORIES:
            for side in ("a", "b"):
                pooled_k[cat][side] += c["cat_k"][cat][side]
                pooled_n[cat][side] += c["cat_n"][cat][side]

    cats = list(ATTACK_CATEGORIES)
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for j, (side, label, color) in enumerate(
        (("a", "raw_rag", _MODE_COLORS[2]), ("b", "harnessed", _MODE_COLORS[0]))
    ):
        rates, lo_err, hi_err = [], [], []
        for cat in cats:
            k, nn = pooled_k[cat][side], pooled_n[cat][side]
            ci = stats.proportion_ci(k, nn, method="wilson")
            rate = 100 * (ci["p"] if nn else 0.0)
            rates.append(rate)
            lo_err.append(max(0.0, rate - 100 * ci["lo"]) if nn else 0.0)
            hi_err.append(max(0.0, 100 * ci["hi"] - rate) if nn else 0.0)
        ax.bar(
            x + (j - 0.5) * width,
            rates,
            width * 0.92,
            yerr=[lo_err, hi_err],
            capsize=3,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "ecolor": "#52514e"},
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylabel("Attack success rate (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    cat_path = report_dir / "phase3_categories.pdf"
    fig.savefig(cat_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    written.append(cat_path)

    return written


def export_phase2_significance(
    experiment: str,
    models: list[str] | None = None,
    report_dir: Path | None = None,
) -> Path:
    """Write the paired raw_rag-vs-harnessed significance table.

    Runs :func:`compare_modes` for every model that has finished runs in
    both modes (auto-discovered when ``models`` is None), applies
    Benjamini-Hochberg correction across the McNemar p-values (and,
    separately, across the Wilcoxon p-values), and writes
    ``phase2_significance.tex`` into ``report_dir``.
    """
    import mlflow
    import pandas as pd

    from aegis.eval import stats

    mode_a, mode_b = "raw_rag", "harnessed"

    config = get_config()
    report_dir = Path(report_dir) if report_dir is not None else config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    if models is None:
        runs = mlflow.search_runs(experiment_names=[experiment])
        if runs is None or len(runs) == 0:
            raise RuntimeError(
                f"No MLflow runs found for experiment {experiment!r}."
            )
        df = pd.DataFrame(runs)
        models = []
        if "params.model" in df.columns and "params.mode" in df.columns:
            finished = df[df["status"] == "FINISHED"]
            for m in sorted(finished["params.model"].dropna().unique()):
                have = set(finished[finished["params.model"] == m]["params.mode"])
                if mode_a in have and mode_b in have:
                    models.append(m)

    comparisons: list[dict] = []
    for m in models:
        try:
            comparisons.append(
                compare_modes(experiment, m, mode_a=mode_a, mode_b=mode_b)
            )
        except RuntimeError:
            logger.warning(
                "Skipping model %r: could not build the %s/%s paired comparison",
                m,
                mode_a,
                mode_b,
                exc_info=True,
            )
    if not comparisons:
        raise RuntimeError(
            f"No model in experiment {experiment!r} has finished runs for "
            f"both {mode_a!r} and {mode_b!r}; nothing to compare."
        )

    mc_adj = stats.bh_correction([c["mcnemar_p"] for c in comparisons])["p_adjusted"]
    wx_adj = stats.bh_correction([c["wilcoxon_p"] for c in comparisons])["p_adjusted"]

    def _fmt_p(p: float) -> str:
        return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"

    def _fmt_delta(delta: float, ci: tuple[float, float]) -> str:
        return f"{100 * delta:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"

    rows = []
    for c, mp, wp in zip(comparisons, mc_adj, wx_adj):
        rows.append(
            {
                "Model": c["model"],
                "EM raw": f"{100 * c['em_a']:.1f}",
                "EM harnessed": f"{100 * c['em_b']:.1f}",
                "$\\Delta$EM [95\\% CI]": _fmt_delta(c["em_delta"], c["em_ci"]),
                "McNemar $p$ (BH)": _fmt_p(mp),
                "$\\Delta$F1 [95\\% CI]": _fmt_delta(c["f1_delta"], c["f1_ci"]),
                "Wilcoxon $p$ (BH)": _fmt_p(wp),
                "$n$": str(c["n_pairs"]),
            }
        )
    tex = _wrap_table(
        _booktabs_tabular(pd.DataFrame(rows), escape_header=False),
        caption=(
            "Paired per-question significance of the harnessed pipeline vs "
            "raw RAG on HotpotQA (pooled across seeds). $\\Delta$ values are "
            "percentage points with 95\\% paired-bootstrap CIs; McNemar (EM) "
            "and Wilcoxon (F1) $p$-values are Benjamini--Hochberg (BH) "
            "adjusted across models."
        ),
        label="tab:phase2-significance",
    )
    out_path = report_dir / "phase2_significance.tex"
    out_path.write_text(tex, encoding="utf-8")
    return out_path

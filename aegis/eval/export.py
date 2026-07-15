"""Export Phase 1 results from MLflow into paper-ready LaTeX tables and figures.

All heavy dependencies (mlflow, pandas, matplotlib) are imported lazily so
this module can be imported without them installed.
"""

from __future__ import annotations

from pathlib import Path

from aegis.config import get_config

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

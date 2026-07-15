"""Aegis command-line interface (Typer app; entry point ``aegis``)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="aegis",
    help="Aegis: a model-agnostic RAG reliability harness.",
    no_args_is_help=True,
)
console = Console()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str) -> list[int]:
    return [int(s) for s in _parse_csv(value)]


def _print_repro(command: str) -> None:
    console.print(f"\n[dim]Reproduce with:[/dim] [bold]{command}[/bold]")


def _metrics_table(title: str, rows: list[dict]) -> Table:
    """Build a rich table from run_eval outputs ({seed, run_id, metrics})."""
    table = Table(title=title)
    table.add_column("seed", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("EM 95% CI", justify="right")
    table.add_column("judge", justify="right")
    table.add_column("abstain", justify="right")
    table.add_column("cost/q ($)", justify="right")
    table.add_column("run_id")

    def _fmt(m: dict, key: str, spec: str = ".3f") -> str:
        return format(m[key], spec) if key in m else "-"

    for row in rows:
        m = row["metrics"]
        ci = (
            f"[{m['em_ci_lo']:.3f}, {m['em_ci_hi']:.3f}]"
            if "em_ci_lo" in m and "em_ci_hi" in m
            else "-"
        )
        table.add_row(
            str(row["seed"]),
            _fmt(m, "em_mean"),
            _fmt(m, "f1_mean"),
            ci,
            _fmt(m, "judge_mean"),
            _fmt(m, "abstain_rate"),
            _fmt(m, "mean_cost_usd", ".5f"),
            row["run_id"][:8],
        )
    return table


@app.command("eval")
def eval_cmd(
    model: str = typer.Option(..., "--model", help="Model alias or litellm string."),
    benchmark: str = typer.Option("hotpotqa", "--benchmark"),
    n: int = typer.Option(300, "--n", help="Number of questions."),
    seed: int = typer.Option(0, "--seed"),
    seeds: str = typer.Option(
        "", "--seeds", help="Comma-separated seeds; overrides --seed if set."
    ),
    mode: str = typer.Option("raw_rag", "--mode"),
    temperature: float = typer.Option(0.0, "--temperature"),
    k_final: int = typer.Option(5, "--k-final"),
    judge_model: str = typer.Option(None, "--judge-model"),
    experiment: str = typer.Option("aegis-phase1", "--experiment"),
    split: str = typer.Option("validation", "--split"),
) -> None:
    """Run one evaluation (per seed) and log it to MLflow."""
    from aegis.eval.run import run_eval

    seed_list = _parse_seeds(seeds) if seeds else [seed]
    rows = []
    for s in seed_list:
        out = run_eval(
            model=model,
            benchmark=benchmark,
            n=n,
            seed=s,
            mode=mode,
            temperature=temperature,
            k_final=k_final,
            judge_model=judge_model,
            experiment=experiment,
            split=split,
        )
        rows.append({"seed": s, "run_id": out["run_id"], "metrics": out["metrics"]})

    console.print(_metrics_table(f"aegis eval — {model} / {mode} (n={n})", rows))

    seeds_arg = (
        f"--seeds {','.join(str(s) for s in seed_list)}"
        if len(seed_list) > 1
        else f"--seed {seed_list[0]}"
    )
    judge_arg = f" --judge-model {judge_model}" if judge_model else ""
    _print_repro(
        f"aegis eval --model {model} --benchmark {benchmark} --n {n} {seeds_arg} "
        f"--mode {mode} --temperature {temperature} --k-final {k_final}"
        f"{judge_arg} --experiment {experiment} --split {split}"
    )


@app.command("sweep")
def sweep_cmd(
    models: str = typer.Option("mock", "--models", help="Comma-separated models."),
    modes: str = typer.Option(
        "closed_book,vanilla_rag,raw_rag", "--modes", help="Comma-separated modes."
    ),
    seeds: str = typer.Option("0,1,2", "--seeds", help="Comma-separated seeds."),
    n: int = typer.Option(300, "--n"),
    benchmark: str = typer.Option("hotpotqa", "--benchmark"),
    temperature: float = typer.Option(0.0, "--temperature"),
    k_final: int = typer.Option(5, "--k-final"),
    judge_model: str = typer.Option(None, "--judge-model"),
    experiment: str = typer.Option("aegis-phase1", "--experiment"),
    split: str = typer.Option("validation", "--split"),
) -> None:
    """Run the full models x modes x seeds grid."""
    from aegis.eval.run import run_sweep

    model_list = _parse_csv(models)
    mode_list = _parse_csv(modes)
    seed_list = _parse_seeds(seeds)

    run_sweep(
        models=model_list,
        modes=mode_list,
        seeds=seed_list,
        n=n,
        benchmark=benchmark,
        temperature=temperature,
        k_final=k_final,
        judge_model=judge_model,
        experiment=experiment,
        split=split,
    )

    judge_arg = f" --judge-model {judge_model}" if judge_model else ""
    _print_repro(
        f"aegis sweep --models {','.join(model_list)} --modes {','.join(mode_list)} "
        f"--seeds {','.join(str(s) for s in seed_list)} --n {n} "
        f"--benchmark {benchmark} --temperature {temperature} --k-final {k_final}"
        f"{judge_arg} --experiment {experiment} --split {split}"
    )


@app.command("ingest")
def ingest_cmd(
    n: int = typer.Option(300, "--n"),
    seed: int = typer.Option(0, "--seed"),
    split: str = typer.Option("validation", "--split"),
) -> None:
    """Load HotpotQA, build the corpus, and (pre)build the retrieval index."""
    from aegis.config import get_config
    from aegis.eval.benchmarks.hotpotqa import build_corpus, load_hotpotqa, sample_hash
    from aegis.retrieve import build_retriever

    config = get_config()
    questions = load_hotpotqa(n, seed, split)
    chunks = build_corpus(questions)
    build_retriever(chunks)

    console.print(f"Loaded [bold]{len(questions)}[/bold] questions "
                  f"(sample_hash={sample_hash(questions)})")
    console.print(f"Corpus size: [bold]{len(chunks)}[/bold] chunks")
    console.print(f"Index/data dir: [bold]{config.data_dir.resolve()}[/bold]")
    _print_repro(f"aegis ingest --n {n} --seed {seed} --split {split}")


@app.command("export")
def export_cmd(
    experiment: str = typer.Option("aegis-phase1", "--experiment"),
) -> None:
    """Export LaTeX tables and figures for the paper from MLflow runs."""
    from aegis.eval.export import export_phase1

    try:
        paths = export_phase1(experiment=experiment)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print("Wrote:")
    for p in paths:
        console.print(f"  [bold]{p.resolve()}[/bold]")
    _print_repro(f"aegis export --experiment {experiment}")


@app.command("models")
def models_cmd() -> None:
    """List the model registry (alias -> litellm model string)."""
    from aegis.config import MODEL_REGISTRY

    table = Table(title="Aegis model registry")
    table.add_column("alias")
    table.add_column("model string")
    for alias, model_string in MODEL_REGISTRY.items():
        table.add_row(alias, model_string)
    console.print(table)
    _print_repro("aegis models")


if __name__ == "__main__":
    app()

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


@app.callback()
def _main() -> None:
    """Load API keys from a local .env (if present) before any command runs."""
    from aegis.config import load_dotenv

    load_dotenv()


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
    k_vote: int = typer.Option(1, "--k-vote", help="Self-consistency votes (harnessed)."),
    vote_temperature: float = typer.Option(0.7, "--vote-temperature"),
    use_structurer: bool = typer.Option(
        True, "--use-structurer/--no-use-structurer"
    ),
    use_verifier: bool = typer.Option(True, "--use-verifier/--no-use-verifier"),
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
            k_vote=k_vote,
            vote_temperature=vote_temperature,
            use_structurer=use_structurer,
            use_verifier=use_verifier,
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
        f"{judge_arg} --experiment {experiment} --split {split} "
        f"--k-vote {k_vote} --vote-temperature {vote_temperature} "
        f"{'--use-structurer' if use_structurer else '--no-use-structurer'} "
        f"{'--use-verifier' if use_verifier else '--no-use-verifier'}"
    )


@app.command("sweep")
def sweep_cmd(
    models: str = typer.Option("mock", "--models", help="Comma-separated models."),
    modes: str = typer.Option(
        "closed_book,vanilla_rag,raw_rag,harnessed",
        "--modes",
        help="Comma-separated modes.",
    ),
    seeds: str = typer.Option("0,1,2", "--seeds", help="Comma-separated seeds."),
    n: int = typer.Option(300, "--n"),
    benchmark: str = typer.Option("hotpotqa", "--benchmark"),
    temperature: float = typer.Option(0.0, "--temperature"),
    k_final: int = typer.Option(5, "--k-final"),
    judge_model: str = typer.Option(None, "--judge-model"),
    experiment: str = typer.Option("aegis-phase1", "--experiment"),
    split: str = typer.Option("validation", "--split"),
    k_vote: int = typer.Option(1, "--k-vote", help="Self-consistency votes (harnessed)."),
    vote_temperature: float = typer.Option(0.7, "--vote-temperature"),
    use_structurer: bool = typer.Option(
        True, "--use-structurer/--no-use-structurer"
    ),
    use_verifier: bool = typer.Option(True, "--use-verifier/--no-use-verifier"),
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
        k_vote=k_vote,
        vote_temperature=vote_temperature,
        use_structurer=use_structurer,
        use_verifier=use_verifier,
    )

    judge_arg = f" --judge-model {judge_model}" if judge_model else ""
    _print_repro(
        f"aegis sweep --models {','.join(model_list)} --modes {','.join(mode_list)} "
        f"--seeds {','.join(str(s) for s in seed_list)} --n {n} "
        f"--benchmark {benchmark} --temperature {temperature} --k-final {k_final}"
        f"{judge_arg} --experiment {experiment} --split {split} "
        f"--k-vote {k_vote} --vote-temperature {vote_temperature} "
        f"{'--use-structurer' if use_structurer else '--no-use-structurer'} "
        f"{'--use-verifier' if use_verifier else '--no-use-verifier'}"
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


@app.command("compare")
def compare_cmd(
    experiment: str = typer.Option("aegis-phase2", "--experiment"),
    model: str = typer.Option(..., "--model", help="Model alias to compare."),
    mode_a: str = typer.Option("raw_rag", "--mode-a"),
    mode_b: str = typer.Option("harnessed", "--mode-b"),
) -> None:
    """Paired per-question significance of mode_b vs mode_a for one model."""
    from aegis.eval.export import compare_modes

    try:
        out = compare_modes(experiment, model, mode_a=mode_a, mode_b=mode_b)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    table = Table(
        title=(
            f"aegis compare — {out['model']}: {out['mode_a']} vs {out['mode_b']} "
            f"(n={out['n_pairs']} paired questions)"
        )
    )
    table.add_column("metric")
    table.add_column(out["mode_a"], justify="right")
    table.add_column(out["mode_b"], justify="right")
    table.add_column("delta", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p-value", justify="right")
    table.add_row(
        "EM",
        f"{out['em_a']:.3f}",
        f"{out['em_b']:.3f}",
        f"{out['em_delta']:+.3f}",
        f"[{out['em_ci'][0]:+.3f}, {out['em_ci'][1]:+.3f}]",
        f"{out['mcnemar_p']:.4g} (McNemar)",
    )
    table.add_row(
        "F1",
        f"{out['f1_a']:.3f}",
        f"{out['f1_b']:.3f}",
        f"{out['f1_delta']:+.3f}",
        f"[{out['f1_ci'][0]:+.3f}, {out['f1_ci'][1]:+.3f}]",
        f"{out['wilcoxon_p']:.4g} (Wilcoxon)",
    )
    console.print(table)
    _print_repro(
        f"aegis compare --experiment {experiment} --model {model} "
        f"--mode-a {mode_a} --mode-b {mode_b}"
    )


@app.command("export2")
def export2_cmd(
    experiment: str = typer.Option("aegis-phase2", "--experiment"),
    png: bool = typer.Option(False, "--png", help="Also write a web-ready PNG."),
) -> None:
    """Export the Phase 2 Pareto figure, results table, and significance table."""
    from aegis.eval.export import export_phase2, export_phase2_significance

    try:
        paths = export_phase2(experiment=experiment, png=png)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        paths.append(export_phase2_significance(experiment=experiment))
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] skipping significance table: {exc}"
        )

    console.print("Wrote:")
    for p in paths:
        console.print(f"  [bold]{p.resolve()}[/bold]")
    _print_repro(f"aegis export2 --experiment {experiment}")


@app.command("inject")
def inject_cmd(
    models: str = typer.Option("mock", "--models", help="Comma-separated models."),
    seeds: str = typer.Option("0,1,2", "--seeds"),
    n: int = typer.Option(300, "--n"),
    poison_rate: float = typer.Option(0.1, "--poison-rate"),
    k_final: int = typer.Option(5, "--k-final"),
    judge_model: str = typer.Option(None, "--judge-model"),
    experiment: str = typer.Option("aegis-phase3", "--experiment"),
    split: str = typer.Option("validation", "--split"),
) -> None:
    """Phase 3: run the injection experiment (raw vs harnessed on poisoned corpus).

    Runs ``raw_rag`` (undefended) and ``harnessed`` (defended) over the poisoned
    HotpotQA corpus for every model x seed, logging attack-success-rate metrics.
    """
    from aegis.eval.run import run_sweep

    run_sweep(
        models=_parse_csv(models),
        modes=["raw_rag", "harnessed"],
        seeds=_parse_seeds(seeds),
        n=n,
        benchmark="hotpotqa",
        k_final=k_final,
        judge_model=judge_model,
        experiment=experiment,
        split=split,
        poison=True,
        poison_rate=poison_rate,
        use_defense=True,
    )
    _print_repro(
        f"aegis inject --models {models} --seeds {seeds} --n {n} "
        f"--poison-rate {poison_rate} --experiment {experiment}"
    )


@app.command("ablate")
def ablate_cmd(
    model: str = typer.Option(..., "--model", help="Model alias or litellm string."),
    seeds: str = typer.Option("0,1,2", "--seeds"),
    n: int = typer.Option(300, "--n"),
    ablations: str = typer.Option(
        "full,no_structurer,no_verifier,no_reranker", "--ablations"
    ),
    k_final: int = typer.Option(5, "--k-final"),
    judge_model: str = typer.Option(None, "--judge-model"),
    experiment: str = typer.Option("aegis-phase4", "--experiment"),
    split: str = typer.Option("validation", "--split"),
) -> None:
    """Phase 4: run the component ablation study (full + one-component removals)."""
    from aegis.eval.run import run_ablation

    run_ablation(
        model=model,
        seeds=_parse_seeds(seeds),
        n=n,
        ablations=_parse_csv(ablations),
        experiment=experiment,
        benchmark="hotpotqa",
        k_final=k_final,
        judge_model=judge_model,
        split=split,
    )
    _print_repro(
        f"aegis ablate --model {model} --seeds {seeds} --n {n} "
        f"--ablations {ablations} --experiment {experiment}"
    )


@app.command("export3")
def export3_cmd(
    experiment: str = typer.Option("aegis-phase3", "--experiment"),
    png: bool = typer.Option(False, "--png", help="Also write a web-ready PNG."),
    csv: bool = typer.Option(False, "--csv", help="Also write injection.csv."),
) -> None:
    """Export the Phase 3 injection table and per-category ASR figure."""
    from aegis.eval.export import export_phase3

    try:
        paths = export_phase3(experiment=experiment, png=png, csv=csv)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print("Wrote:")
    for p in paths:
        console.print(f"  [bold]{p.resolve()}[/bold]")
    _print_repro(f"aegis export3 --experiment {experiment}")


@app.command("export4")
def export4_cmd(
    experiment: str = typer.Option("aegis-phase4", "--experiment"),
    model: str = typer.Option(None, "--model", help="Model to tabulate (default: first)."),
) -> None:
    """Export the Phase 4 component-ablation table."""
    from aegis.eval.export import export_phase4_ablation

    try:
        path = export_phase4_ablation(experiment=experiment, model=model)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"Wrote:\n  [bold]{path.resolve()}[/bold]")
    _print_repro(f"aegis export4 --experiment {experiment}")


@app.command("serve")
def serve_cmd(
    demo: bool = typer.Option(False, "--demo", help="Launch the Gradio demo."),
    api: bool = typer.Option(False, "--api", help="Launch the FastAPI server."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    model: str = typer.Option(
        None, "--model", help="Model alias for answers (default: $AEGIS_SERVE_MODEL or mock)."
    ),
) -> None:
    """Serve Aegis: the Gradio demo (--demo) or the FastAPI API (--api)."""
    import os

    if demo == api:
        console.print("[red]Error:[/red] pass exactly one of --demo or --api.")
        raise typer.Exit(code=1)
    if model:
        os.environ["AEGIS_SERVE_MODEL"] = model
    if demo:
        from aegis.serve.demo import build_demo

        build_demo().launch(server_name=host, server_port=port)
    else:
        import uvicorn

        uvicorn.run("aegis.serve.api:app", host=host, port=port)


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

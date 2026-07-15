"""Experiment driver: run one evaluation (or a sweep) and log it to MLflow.

Heavy dependencies (mlflow, the pipeline stack, retrieval) are imported
lazily inside functions so this module can be imported cheaply.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path

from aegis import __version__
from aegis.config import get_config, resolve_model
from aegis.eval import stats
from aegis.types import Answer, QueryResult

logger = logging.getLogger(__name__)


def prompt_hash() -> str:
    """Stable 12-hex-char hash of the pipeline's prompt templates.

    Logged as an MLflow param so any prompt change is visible across runs.
    """
    from aegis.pipeline import PROMPT_TEMPLATES

    payload = json.dumps(PROMPT_TEMPLATES, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _mean(values) -> float | None:
    """Mean over non-None values; None if nothing to average."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals)) / len(vals)


def _failed_result(q, mode: str, model: str) -> QueryResult:
    """Placeholder result for a question whose pipeline call raised."""
    return QueryResult(
        question_id=q.id,
        question=q.question,
        gold_answer=q.answer,
        answer=Answer(text="", abstained=True, raw_response=""),
        retrieved=[],
        mode=mode,
        model=model,
    )


def run_eval(
    model: str,
    benchmark: str = "hotpotqa",
    n: int = 300,
    seed: int = 0,
    mode: str = "raw_rag",
    temperature: float = 0.0,
    k_final: int = 5,
    judge_model: str | None = None,
    experiment: str = "aegis-phase1",
    split: str = "validation",
) -> dict:
    """Run one full evaluation and log params/metrics/artifacts to MLflow.

    Returns ``{"run_id": ..., "metrics": {...}}``.
    """
    import mlflow
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    from aegis.eval.benchmarks.hotpotqa import (
        build_corpus,
        load_hotpotqa,
        sample_hash,
    )
    from aegis.eval.scorers import score_results
    from aegis.gateway import get_client
    from aegis.pipeline import RAGPipeline

    if benchmark != "hotpotqa":
        raise ValueError(f"Unknown benchmark: {benchmark!r} (only 'hotpotqa' is supported)")

    config = get_config()

    # 1. Data and (unless closed-book) retrieval index.
    questions = load_hotpotqa(n, seed, split)
    retriever = None
    if mode != "closed_book":
        from aegis.retrieve import build_retriever

        chunks = build_corpus(questions)
        retriever = build_retriever(chunks)

    client = get_client(model, temperature=temperature, seed=seed)
    pipeline = RAGPipeline(client, retriever=retriever, mode=mode, k_final=k_final)

    # 2. MLflow run bookkeeping.
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"{model}-{mode}-s{seed}") as active_run:
        mlflow.log_params(
            {
                "model": model,
                "model_resolved": resolve_model(model),
                "mode": mode,
                "seed": seed,
                "temperature": temperature,
                "n": n,
                "split": split,
                "k_final": k_final,
                "sample_hash": sample_hash(questions),
                "prompt_hash": prompt_hash(),
                "embedder": config.embedder,
                "reranker": config.reranker,
                "retrieval_backend": config.retrieval_backend,
                "benchmark": benchmark,
                "aegis_version": __version__,
                "judge_model": judge_model or model,
            }
        )

        # 3. Run the pipeline on every question; never abort the run on one
        # bad question — record a failed/abstained result instead.
        results: list[QueryResult] = []
        n_failed = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"{model}/{mode} s{seed}", total=len(questions))
            for q in questions:
                try:
                    results.append(pipeline.run(q))
                except Exception:
                    n_failed += 1
                    logger.warning(
                        "Pipeline failed on question %s; recording abstained result",
                        q.id,
                        exc_info=True,
                    )
                    results.append(_failed_result(q, mode, model))
                progress.advance(task)

        # 4. Scoring (judge uses temperature 0 and the run seed).
        judge_client = get_client(judge_model or model, temperature=0.0, seed=seed)
        scored = score_results(results, questions, judge_client=judge_client)

        # 5. Metrics. Means are recomputed from the per-question lists so
        # None entries (e.g. judge on abstentions) are skipped consistently.
        total_cost = float(sum(r.cost_usd for r in results))
        n_q = max(len(results), 1)
        abstain_rate = scored.get("abstain_rate")
        if abstain_rate is None:
            abstain_rate = sum(1 for r in results if r.answer.abstained) / n_q

        em_mean, em_lo, em_hi = stats.bootstrap_ci(
            [v for v in scored.get("em", []) if v is not None], seed=seed
        )
        f1_mean, f1_lo, f1_hi = stats.bootstrap_ci(
            [v for v in scored.get("f1", []) if v is not None], seed=seed
        )

        metrics: dict[str, float] = {}

        def _put(key: str, value) -> None:
            if value is None:
                return
            value = float(value)
            if value != value:  # NaN
                return
            metrics[key] = value

        _put("em_mean", em_mean)
        _put("em_ci_lo", em_lo)
        _put("em_ci_hi", em_hi)
        _put("f1_mean", f1_mean)
        _put("f1_ci_lo", f1_lo)
        _put("f1_ci_hi", f1_hi)
        _put("judge_mean", _mean(scored.get("judge", [])))
        _put("retrieval_precision_mean", _mean(scored.get("retrieval_precision", [])))
        _put("retrieval_recall_mean", _mean(scored.get("retrieval_recall", [])))
        _put("citation_precision_mean", _mean(scored.get("citation_precision", [])))
        _put("abstain_rate", abstain_rate)
        _put("failed_questions", n_failed)
        _put("total_cost_usd", total_cost)
        _put("mean_cost_usd", total_cost / n_q)
        _put("mean_latency_s", _mean([r.latency_s for r in results]))
        _put("mean_tokens_in", _mean([r.tokens_in for r in results]))
        _put("mean_tokens_out", _mean([r.tokens_out for r in results]))

        mlflow.log_metrics(metrics)

        # 6. Artifacts: per-question JSONL + the full scored summary.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            def _pq(key: str, i: int):
                vals = scored.get(key)
                if isinstance(vals, list) and i < len(vals):
                    return vals[i]
                return None

            jsonl_path = tmp_dir / "results.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as fh:
                for i, r in enumerate(results):
                    fh.write(
                        json.dumps(
                            {
                                "question_id": r.question_id,
                                "question": r.question,
                                "gold": r.gold_answer,
                                "predicted": r.answer.text,
                                "abstained": r.answer.abstained,
                                "citations": [c.chunk_id for c in r.answer.citations],
                                "em": _pq("em", i),
                                "f1": _pq("f1", i),
                                "judge": _pq("judge", i),
                                "retrieval_precision": _pq("retrieval_precision", i),
                                "retrieval_recall": _pq("retrieval_recall", i),
                                "citation_precision": _pq("citation_precision", i),
                                "cost_usd": r.cost_usd,
                                "latency_s": r.latency_s,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            mlflow.log_artifact(str(jsonl_path))

            summary_path = tmp_dir / "metrics_summary.json"
            summary_path.write_text(
                json.dumps(scored, indent=2, default=str), encoding="utf-8"
            )
            mlflow.log_artifact(str(summary_path))

        return {"run_id": active_run.info.run_id, "metrics": metrics}


def run_sweep(
    models: list[str],
    modes: list[str],
    seeds: list[int],
    n: int,
    **kw,
) -> list[dict]:
    """Run the full ``models x modes x seeds`` grid via :func:`run_eval`.

    The retrieval index is disk-cached by the retriever layer, so repeated
    per-run corpus/retriever construction is cheap after the first build.
    Prints a rich summary table of all runs at the end.
    """
    from rich.console import Console
    from rich.table import Table

    rows: list[dict] = []
    for model in models:
        for mode in modes:
            for seed in seeds:
                out = run_eval(model=model, mode=mode, seed=seed, n=n, **kw)
                rows.append(
                    {
                        "model": model,
                        "mode": mode,
                        "seed": seed,
                        "run_id": out["run_id"],
                        "metrics": out["metrics"],
                    }
                )

    table = Table(title=f"Aegis sweep (n={n})")
    table.add_column("model")
    table.add_column("mode")
    table.add_column("seed", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("abstain", justify="right")
    table.add_column("cost/q ($)", justify="right")
    table.add_column("run_id")

    def _fmt(m: dict, key: str, spec: str = ".3f") -> str:
        return format(m[key], spec) if key in m else "-"

    for row in rows:
        m = row["metrics"]
        table.add_row(
            row["model"],
            row["mode"],
            str(row["seed"]),
            _fmt(m, "em_mean"),
            _fmt(m, "f1_mean"),
            _fmt(m, "abstain_rate"),
            _fmt(m, "mean_cost_usd", ".5f"),
            row["run_id"][:8],
        )
    Console().print(table)
    return rows

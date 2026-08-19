# Aegis

**Can a structured harness make a cheap model match an expensive model on grounded QA?**

Aegis is a model-agnostic RAG reliability harness evaluated on HotpotQA. It wraps any
litellm-compatible model in a fixed pipeline and measures how much of the reliability gap
between cheap and frontier models the harness itself can close.

## Architecture

The harness is a four-stage pipeline:

1. **Structurer** — decomposes the incoming question into a structured retrieval plan.
2. **Injection defense** — screens retrieved content for prompt-injection and poisoning.
3. **Hybrid RAG core** — dense + sparse retrieval with fusion and reranking, and a strict
   citation contract: every claim in the answer must cite a retrieved chunk.
4. **NLI verifier** — checks that cited chunks actually entail the answer; unsupported
   answers trigger abstention.

## Quickstart

```bash
uv sync                         # or: pip install -e .[dev]
aegis --help

# Phase 1/2 — accuracy + cost frontier (sweep models x modes x seeds)
aegis sweep --models mock --modes closed_book,vanilla_rag,raw_rag,harnessed --seeds 0,1,2 --n 300
aegis export --experiment aegis-phase1
aegis export2 --experiment aegis-phase2

# Phase 3 — prompt-injection robustness on a poisoned corpus
aegis inject --models mock --seeds 0,1,2 --n 300 --poison-rate 0.1
aegis export3 --experiment aegis-phase3

# Phase 4 — component ablations
aegis ablate --model mock --seeds 0,1,2 --n 300
aegis export4 --experiment aegis-phase4

# Live demo with a harness on/off toggle (needs the `serve` extra)
aegis serve --demo          # Gradio UI
aegis serve --api           # FastAPI (uvicorn) at /answer, /index, /health
```

Swap `--model mock` for a registry alias (`local`, `small`, `mid`, `frontier` — see
`aegis models`) to run real models via LiteLLM. The `mock` model is deterministic and
needs no credentials, so the whole pipeline (including the demo/API) runs offline.

## Status

All four phases are implemented:

- **Phase 1** — hybrid RAG core (dense bge + BM25 → RRF → cross-encoder rerank) under a
  citation contract, HotpotQA EM/F1 + LLM-as-judge, MLflow tracking, bootstrap CIs.
- **Phase 2** — query structurer, NLI groundedness verifier with retry-then-abstain,
  k-sample self-consistency, and the cost-vs-accuracy Pareto frontier (LangGraph).
- **Phase 3** — prompt-injection defense (direct heuristic gate, indirect
  spotlighting/imperative-stripping, canary-token exfiltration blocking) and a
  standalone poisoned-corpus generator; attack-success-rate reported per category.
- **Phase 4** — component ablations with per-component significance tests, the LaTeX
  paper (`paper/`), and the FastAPI + Gradio serving layer.

Statistical methods live in [`aegis/eval/stats.py`](aegis/eval/stats.py): bootstrap and
hierarchical (question-clustered) CIs, mid-p McNemar, Wilcoxon, paired-bootstrap deltas,
Wilson/Clopper-Pearson proportion intervals, TOST non-inferiority testing, power/MDE
analysis, and Benjamini-Hochberg correction.

## Reproducibility

All experiment runs are tracked with MLflow (local `mlruns/` by default; override with
`AEGIS_MLFLOW_URI`). Every number in the paper is generated from logged runs — no
hand-copied results.

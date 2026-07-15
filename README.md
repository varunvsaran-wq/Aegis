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
uv sync            # or: pip install -e .[dev]
aegis --help
```

## Status

Phase 1 (hybrid RAG core + HotpotQA evaluation) is implemented; the remaining harness
stages (structurer, injection defense, NLI verifier) are in progress.

## Reproducibility

All experiment runs are tracked with MLflow (local `mlruns/` by default; override with
`AEGIS_MLFLOW_URI`). Every number in the paper is generated from logged runs — no
hand-copied results.

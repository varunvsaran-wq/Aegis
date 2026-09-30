# Aegis

**Can a harness around a cheap model make it reliable enough to trust?**

Aegis is a set of reliability harnesses for LLM applications, each tested against
the same model without the harness on held-out data, with confidence intervals and
paired tests. It has two parts, and the second exists because of what the first showed.

## 1. Customer-service harness (current)

[`cs_harness/`](cs_harness/) · report: [`cs_harness/report/cs_report.pdf`](cs_harness/report/cs_report.pdf)

An airline support agent on [tau2-bench](https://github.com/sierra-research/tau2-bench),
where an AI plays the customer and a task only counts if the booking database ends up
exactly right. Before the agent can book, change, cancel or compensate, the harness checks
the action against the airline's written policy, **in code**, using the live booking
records. A blocked action never runs; the agent is told which rule it broke and can repair
the call or turn the customer down.

On 20 held-out tasks with GPT-4o-mini (4 tries each):

| | Solved | Solved on every try | $ per solved task |
|---|---|---|---|
| Plain agent | 26.2% | 0% | 0.057 |
| **Harness, code checker** | **43.8%** | **25%** | **0.038** |
| Harness, AI-model checker | 31.2% | 5% | 0.069 |
| Plain GPT-4.1-mini (stronger model) | 47.5% | 20% | 0.031 |

The code checker blocked 72 actions, 3 of them correct ones; the same loop with an AI
checker blocked 227, 17 of them correct. With 20 tasks the gain over the plain agent is
large but not yet statistically certain (95% CI -1.2 to +36.2 points, p = 0.088). Test plan,
frozen before the test run: [`cs_harness/PROTOCOL.md`](cs_harness/PROTOCOL.md).

## 2. Retrieval QA harness (first version)

[`aegis/`](aegis/) · report: [`report/demo/aegis_report.pdf`](report/demo/aegis_report.pdf)

A RAG pipeline on HotpotQA (hybrid BM25 + dense retrieval, cross-encoder reranking) with a
query structurer, prompt-injection defense, a citation contract, and an NLI verifier that
retries once and then abstains. On 150 held-out questions with Claude Haiku 4.5 it cut
answers given without evidence from 12.7% to 3.3%, but declined 39% of answerable
questions, and its accuracy on the questions it did answer was no better than plain RAG.
The verifier was an AI model that rejected correct answers about as often as wrong ones.
That result is why part 1 uses a checker written in code.

```bash
uv sync                         # or: pip install -e .[dev,serve]
aegis --help
aegis serve --demo --model small   # side-by-side demo: plain RAG vs the harness
python scripts/demo_benchmark.py --help   # the held-out benchmark behind the report
```

The HotpotQA stack also has the original research tooling: multi-seed MLflow sweeps,
a poisoned-corpus generator, component ablations, and a statistics module
([`aegis/eval/stats.py`](aegis/eval/stats.py)) with bootstrap and clustered CIs, mid-p
McNemar, Wilcoxon, Wilson intervals, non-inferiority tests and Benjamini-Hochberg
correction. See [`scripts/README.md`](scripts/README.md) for the benchmark and report
commands, and `aegis --help` for the rest.

## Setup notes

- API keys go in a local `.env` (gitignored): `ANTHROPIC_API_KEY` for the HotpotQA
  work, `OPENROUTER_API_KEY` for the customer-service harness.
- tau2-bench is not vendored. Clone it into `third_party/` as described in
  [`cs_harness/README.md`](cs_harness/README.md); it needs Python 3.12 or 3.13.
- The raw conversation records behind the customer-service report are committed,
  gzipped, in `cs_harness/results/runs/`, so every number in that report can be
  regenerated without re-running the models.

## Tests

```bash
pytest -q                                              # HotpotQA harness, offline
cd third_party/tau2-bench && .venv/Scripts/python.exe -m pytest ../../cs_harness/tests -q
```

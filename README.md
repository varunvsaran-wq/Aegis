# Aegis

**Can a harness around a cheap model make it reliable enough to trust?**

Aegis is a set of reliability harnesses for LLM applications and the components that
guard them, each tested on held-out data under a protocol frozen before the test run,
with confidence intervals and paired tests.

## Reports

| Report | Question | Headline |
|---|---|---|
| [**Fast decisions, cheap guards**](jev_study/report/jev_report.pdf) (Sep 2026, 8 pp.) | Can a fast "System One" decision model (Jev) replace an LLM judge for prompt-injection screening and model routing? | Injection AUROC 0.994 vs 0.980 for a Claude Haiku 4.5 judge, at 4× the speed and 1/18 the cost; routing no better than random |
| [**Checking a support agent's actions in code**](cs_harness/report/cs_report.pdf) (Sep 2026, 6 pp.) | Does a policy-as-code gate make a cheap model a reliable airline support agent? | GPT-4o-mini solves 43.8% of tasks with the gate vs 26.2% without, level with a stronger model |
| [**Aegis: does a reliability harness make a cheap model safer for question answering?**](report/demo/aegis_report.pdf) (first version, 7 pp.) | Does a verify-and-abstain harness make RAG answers trustworthy? | Fewer unsupported answers (12.7% → 3.3%), but 39% of answerable questions declined |

Each report's numbers, tables and figures are generated from committed results, so they
can be rebuilt without calling any model.

## 1. Customer-service harness

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

## 2. Jev study: fast models as guards

[`jev_study/`](jev_study/) · report: [`jev_study/report/jev_report.pdf`](jev_study/report/jev_report.pdf)

An independent test of Jev (TypeSafe AI's "System One" model, `jev-1.13.0`), which answers
multiple-choice and rating questions about a text in about 140 ms instead of generating
text. It is compared with LLM judges, a trained local classifier and keyword rules on
direct injections (deepset), injections planted in Wikipedia paragraphs, and harmless
prompts full of trigger words (NotInject); and, separately, as a router between
Mixtral-8x7B and GPT-4 on RouterBench.

On 991 held-out texts:

| | AUROC | Median latency | $ per 1k checks | False alarms on NotInject |
|---|---|---|---|---|
| **Jev (score question)** | **0.994** | 138 ms | 0.019 | 1.6% |
| Claude Haiku 4.5 judge | 0.980 | 556 ms | 0.344 | 2.5% |
| GPT-4o-mini judge | 0.842 | 1,188 ms | 0.038 | 2.9% |
| ProtectAI DeBERTa (local) | 0.809 | 22 ms | 0 | 44.0% |

The catch is the threshold: at its default cut-off Jev catches 69.9% of injections, and
the cut-off that catches 97.8% also flags about half of NotInject. For routing, Jev scored
the same as random routing (AIQ 0.641) and below a TF-IDF router trained on development
prompts (0.677). The whole study cost $0.69. Protocol: [`jev_study/PROTOCOL.md`](jev_study/PROTOCOL.md).

## 3. Retrieval QA harness (first version)

[`aegis/`](aegis/) · report: [`report/demo/aegis_report.pdf`](report/demo/aegis_report.pdf)

A RAG pipeline on HotpotQA (hybrid BM25 + dense retrieval, cross-encoder reranking) with a
query structurer, prompt-injection defense, a citation contract, and an NLI verifier that
retries once and then abstains. On 150 held-out questions with Claude Haiku 4.5 it cut
answers given without evidence from 12.7% to 3.3%, but declined 39% of answerable
questions, and its accuracy on the questions it did answer was no better than plain RAG.
The verifier was an AI model that rejected correct answers about as often as wrong ones.
That result is why the customer-service harness uses a checker written in code.

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
  work and the Haiku judge, `OPENROUTER_API_KEY` for the customer-service harness and
  the GPT-4o-mini judge, `TYPESAFE_API_KEY` for Jev.
- The Jev study's cached decisions are committed in `jev_study/results/cache/`;
  `jev_study/build_report.py` rebuilds its report from them.
- tau2-bench is not vendored. Clone it into `third_party/` as described in
  [`cs_harness/README.md`](cs_harness/README.md); it needs Python 3.12 or 3.13.
- The raw conversation records behind the customer-service report are committed,
  gzipped, in `cs_harness/results/runs/`, so every number in that report can be
  regenerated without re-running the models.

## Tests

```bash
pytest -q                                              # HotpotQA harness, offline
.venv/Scripts/python.exe -m pytest jev_study/tests -q  # Jev study, offline
cd third_party/tau2-bench && .venv/Scripts/python.exe -m pytest ../../cs_harness/tests -q
```

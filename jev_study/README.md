# Jev study

An independent evaluation of Jev (TypeSafe AI's "System One" decision model) on two
jobs inside LLM systems:

1. **Prompt-injection screening:** direct injections, instructions hidden in
   retrieved documents, and over-defence on benign text full of trigger words.
2. **Confidence-gated model routing:** decide per prompt whether Mixtral-8x7B is
   enough or GPT-4 is needed, scored offline on RouterBench.

Each system is measured on accuracy, calibration, latency and cost. The design is in
[PLAN.md](PLAN.md).

## Layout

| Path | What it is |
|---|---|
| `jevbench/deciders.py` | One `Decider` interface: Jev, keyword rules, local HF classifiers, LLM judges, and an offline `FakeJevClient` |
| `jevbench/injection_data.py` | deepset, NotInject, and HotpotQA paragraphs with planted notes; deterministic dev/test split |
| `jevbench/routing.py` | RouterBench loader, cost-quality curves, AIQ, and the length and TF-IDF baselines |
| `jevbench/runner.py` | Resumable JSONL cache and a hard budget guard shared by all runs |
| `jevbench/metrics.py` | AUROC, TPR at a fixed FPR, ECE, Brier, latency, $/1k, bootstrap CIs |
| `smoke_jev.py` | About 20 Jev calls to confirm the API behaves as the code expects |
| `run_injection.py`, `run_routing.py` | Run deciders on a split |
| `analyze.py` | Writes `results/results_<split>.{json,md}` |
| `spend.py` | Total spend across all caches |

## Setup

Uses the Aegis venv (`typesafe-sdk` 0.7.2 is already installed there). Add the key to
the repo's `.env`:

```
TYPESAFE_API_KEY=...
```

## Running it

```bash
# 0. offline tests (no keys, no network)
.venv/Scripts/python.exe -m pytest jev_study/tests -q

# 1. Jev smoke test (~20 calls, well under a cent)
.venv/Scripts/python.exe jev_study/smoke_jev.py

# 2. dev runs: free baselines, Jev wordings, LLM judges
.venv/Scripts/python.exe jev_study/run_injection.py --split dev --deciders keyword deberta-protectai
.venv/Scripts/python.exe jev_study/run_injection.py --split dev --deciders jev-v1 jev-v2 jev-v3
.venv/Scripts/python.exe jev_study/run_injection.py --split dev --deciders judge-gpt4omini judge-haiku45
.venv/Scripts/python.exe jev_study/run_routing.py   --split dev --deciders jev-v1 jev-v2
.venv/Scripts/python.exe jev_study/analyze.py --split dev

# 3. freeze PROTOCOL.md (chosen wordings, thresholds, hashes), then run test once
.venv/Scripts/python.exe jev_study/run_injection.py --split test --deciders ...
.venv/Scripts/python.exe jev_study/analyze.py --split test
```

Rebuild the report (figures, tables and every number are generated):

```bash
.venv/Scripts/python.exe jev_study/build_report.py
cd jev_study/report && pdflatex jev_report.tex && pdflatex jev_report.tex
```

The report is [report/jev_report.pdf](report/jev_report.pdf). The cached decisions
in `results/cache/` reproduce it without any API calls.

The test split refuses to run until `PROTOCOL.md` exists. Every run stops before the
total spend recorded in `results/cache/` would cross `--budget` (default $15). Check
spend with `.venv/Scripts/python.exe jev_study/spend.py`.

## Terms of service

Jev outputs are used for evaluation only. TypeSafe's Master Customer Agreement, section
2.3(b), forbids using them to train or distil an imitating model, so no classifier in
this study is trained on Jev labels. The TF-IDF routing baseline is trained on
RouterBench's own correctness scores.

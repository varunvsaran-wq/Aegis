# Portfolio export scripts

Produce the three artifacts the [varunvsaran.com](https://varunvsaran.com) Aegis
case page expects:

| Artifact | Produced by | Filename |
|---|---|---|
| Trace explorer data | `export_traces.py` | `traces.json` |
| Cost/accuracy Pareto | `aegis export2 --png` | `pareto.png` |
| Injection ASR by category | `aegis export3 --png --csv` | `injection.png` / `injection.csv` |
| (optional) demo video | you record `aegis serve --demo` | `demo.mp4` |

## Prerequisites

```bash
pip install -e .[dev,serve]
# Use a REAL model for traces (the answer text is the proof), not mock.
# Put the key in ./.env (gitignored); the scripts and CLI load it automatically:
#   ANTHROPIC_API_KEY=sk-ant-...          (alias `small` = claude-haiku)
# or run a local model instead:  ollama pull llama3.1:8b   (alias `local`)
```

## 1. Traces (`traces.json`)

Runs a pool of HotpotQA questions through the real pipeline and auto-selects
3 answered-with-citations, 1 abstain, and 1 injection-blocked trace (each with a
harness-off `vanilla_rag` baseline):

```bash
python scripts/export_traces.py --model small --pool 60 --out traces.json
```

- If it can't fill every slot, it says so — bump `--pool` (e.g. 100) or use a
  stronger `--model`. The page copes with 4 traces.
- Deterministic given `--seed`. Caps enforced: ≤6 chunks/trace, ≤400 chars/chunk.
- `--no-require-correct` is a dev/offline switch (accepts answered traces
  regardless of correctness) — do **not** use it for the real artifact.

## 2. Figures

These read your existing MLflow runs, so run the phase-2 sweep and phase-3
injection experiment first (see the top-level README), then:

```bash
aegis export2 --png            # -> report/phase2_pareto.png (~1600px, white bg)
aegis export3 --png --csv      # -> report/injection.png + report/injection.csv
```

## 3. Bundle it

Copy everything into your portfolio assets folder with the expected filenames:

```bash
python scripts/build_portfolio.py --out-dir ../Portfolio/assets/aegis --traces traces.json
```

It validates `traces.json`, reports the trace-kind counts, and tells you which
(if any) artifacts are still missing.

## Reliability report (PDF)

A controlled comparison of plain RAG, RAG + citations, and the full harness on
HotpotQA, written up as `report/demo/aegis_report.pdf`:

```bash
# dev set (tune here only), then the held-out test set
python scripts/demo_benchmark.py --model small --n 40  --offset 0  --out-dir report/demo_benchmark/dev_v2a
python scripts/demo_benchmark.py --model small --n 150 --offset 40 --out-dir report/demo_benchmark/test
# figures, tables, and number macros, then compile
python scripts/build_report.py --bench report/demo_benchmark/test --dev "Claim rewrite=report/demo_benchmark/dev_v2a"
cd report/demo && pdflatex aegis_report.tex && pdflatex aegis_report.tex
```

Each benchmark run caches every record in `records.jsonl`, so re-running
resumes without paying twice, and `--budget-usd` caps total spend (model +
judge). On this machine MiKTeX needs the `...\Python311\python.exe` entry
removed from PATH (it is a file, not a directory) to refresh its package list.

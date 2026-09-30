# Jev study: frozen test protocol

Frozen 2026-09-30, after the dev runs and before any test-split call. Nothing below
changes after the test split is run; any later analysis is labelled exploratory.

## Versions

- Jev: `jev-latest`, which resolved to **`jev-1.13.0`** in every dev call. Each test
  decision records `response.model`; a test result is valid only if it reads `jev-1.13.0`.
- `typesafe-sdk` 0.7.2. Price used for cost: $0.042 per million input tokens.
- Judges: `openrouter/openai/gpt-4o-mini`, `anthropic/claude-haiku-4-5-20251001`,
  temperature 0.
- DeBERTa: `protectai/deberta-v3-base-prompt-injection-v2`, fp32, RTX 4050, batch 1.

Code hashes (first 16 hex digits of SHA-256):

| File | Hash |
|---|---|
| `jevbench/deciders.py` | `0744862a0c8f93af` |
| `jevbench/injection_data.py` | `ac1ff7ac8b8c85ce` |
| `jevbench/metrics.py` | `b98ed6b2db78d6b1` |
| `jevbench/routing.py` | `1f7923c694b01765` |
| `jevbench/runner.py` | `4786547eb8adccc1` |
| `analyze.py` | `05a8b0b2a914a6af` |
| `results/injection_examples.jsonl` | `190ad19524b9b53c` |

## Study 1: prompt-injection screening

**Test data.** The hash-assigned test split: deepset 178 injections / 282 benign;
indirect 144 poisoned / 144 clean paragraphs (held-out note phrasings only); NotInject
243 benign.

**Systems on test.**
- Primary Jev wording: **`jev-s1`** (Score question, 5 levels). Chosen on dev because it
  had the highest AUROC (0.993, ahead of each Choice wording, CI excluding zero).
- Secondary Jev wording: **`jev-v2`** (two-option Choice). Chosen for the confidence
  analysis: lowest confidence ECE on dev (0.073) and no NotInject false alarms.
- Baselines: `keyword`, `deberta-protectai`, `judge-gpt4omini`, `judge-haiku45`.
  Wordings `jev-v1` and `jev-v3` are not run on test.

**Primary outcome.** AUROC on deepset + indirect test texts, with a stratified bootstrap
95% CI (2,000 resamples, seed 0).

**Primary comparisons.** Paired AUROC difference, `jev-s1` minus each baseline, with a
stratified paired bootstrap 95% CI. Headline claims about Jev against Haiku are
equivalence-style: a CI inside ±0.02 is reported as "no meaningful difference".

**Secondary outcomes.**
- Detection rate and false-alarm rate at thresholds fixed on dev negatives (1% and 5%
  FPR targets), applied unchanged to test.
- NotInject false-alarm rate at 0.5 and at the dev thresholds, and by number of trigger words.
- ECE and Brier score of the score as P(injection), and for `jev-v2`, the ECE of its
  reported confidence.
- Per-source AUROC (deepset and indirect separately).
- p50/p95 latency from this machine, and $ per 1,000 checks. Latency comes from runs
  with one call at a time.

## Study 4: routing

**Test data.** RouterBench 0-shot test split: 1,837 family-stratified prompts,
Mixtral-8x7B (cheap) and GPT-4-1106 (strong).

**Systems on test.** Primary Jev wording **`jev-v1`** (best dev AIQ among Jev wordings,
0.666). Baselines: all-cheap, all-strong, random, oracle, prompt length, TF-IDF router
trained on the whole dev split's labels.

**Primary outcome.** AIQ with Jev's own cost included. **Secondary:** AUROC for "GPT-4
beats Mixtral", cost to reach 95% of GPT-4's quality, and quality at 20/40/60% strong share.

**Expectation on record.** On dev, Jev routing was close to random (AIQ 0.666 vs 0.656)
and below the TF-IDF router (0.687). It will be reported as found.

## Budget

Dev spend was $0.23. Test is expected to cost under $1; the runner stops at the $15 cap.

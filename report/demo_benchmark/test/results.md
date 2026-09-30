# Aegis vs. plain RAG: reliability benchmark

Model under test: `small` · judge: `anthropic/claude-sonnet-4-6` · 150 HotpotQA questions (seed 0, offset 40) · generated 2026-09-23

Rates with bootstrap 95% CIs. p-values: paired mid-p McNemar, harness vs. each baseline, on the same questions (not corrected for multiple comparisons).

| Condition / metric | Plain RAG | RAG + citations | Aegis harness | p (vs plain) | p (vs +citations) |
|---|---|---|---|---|---|
| **answerable/accuracy**: Answered correctly (higher is better, n=150) | 81.3% [74.7, 87.3] | 77.3% [70.7, 84.0] | 56.7% [48.7, 64.7] | <0.001 | <0.001 |
| **answerable/wrong_answer_rate**: Answered, but wrong (lower is better, n=150) | 7.3% [3.3, 12.0] | 11.3% [6.7, 16.7] | 4.7% [1.3, 8.7] | 0.332 | 0.007 |
| **answerable/no_answer_rate**: Declined to answer (lower is better, n=150) | 11.3% [6.7, 16.7] | 11.3% [6.7, 16.7] | 38.7% [30.7, 46.0] | <0.001 | <0.001 |
| **unanswerable/answered_rate**: Answered without the evidence (lower is better, n=150) | 12.7% [7.3, 18.0] | 10.0% [5.3, 15.3] | 3.3% [0.7, 6.7] | <0.001 | 0.003 |
| **unanswerable/wrong_answer_rate**: Answered without the evidence, and wrong (lower is better, n=150) | 4.7% [1.3, 8.0] | 6.0% [2.7, 10.0] | 2.7% [0.7, 5.3] | 0.344 | 0.070 |
| **injection/attack_success_rate**: Repeated the planted fake answer (lower is better, n=142) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 1.000 | 1.000 |
| **injection/accuracy**: Still answered correctly under attack (higher is better, n=142) | 78.9% [71.8, 85.2] | 73.2% [65.5, 80.3] | 46.5% [38.0, 54.9] | <0.001 | <0.001 |

## Attack success by injection template

| Template | Plain RAG | Aegis harness |
|---|---|---|
| dev-0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| dev-1 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| dev-2 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| dev-3 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-1 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-2 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-3 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-4 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |

| Group | Plain RAG | Aegis harness |
|---|---|---|
| dev phrasings | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout phrasings | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |

## Cost and latency (answerable condition)

| System | USD / question | Mean latency (s) |
|---|---|---|
| Plain RAG | 0.00109 | 7.71 |
| RAG + citations | 0.00112 | 3.32 |
| Aegis harness | 0.00333 | 15.72 |

Errored records (excluded): 0

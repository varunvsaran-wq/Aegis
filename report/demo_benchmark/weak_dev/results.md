# Aegis vs. plain RAG: reliability benchmark

Model under test: `weak` · judge: `anthropic/claude-sonnet-4-6` · 40 HotpotQA questions (seed 0, offset 0) · generated 2026-09-23

Rates with bootstrap 95% CIs. p-values: paired mid-p McNemar, harness vs. each baseline, on the same questions (not corrected for multiple comparisons).

| Condition / metric | Plain RAG | RAG + citations | Aegis harness | p (vs plain) | p (vs +citations) |
|---|---|---|---|---|---|
| **answerable/accuracy**: Answered correctly (higher is better, n=40) | 70.0% [55.0, 85.0] | 77.5% [65.0, 90.0] | 55.0% [40.0, 70.0] | 0.092 | 0.006 |
| **answerable/wrong_answer_rate**: Answered, but wrong (lower is better, n=40) | 30.0% [15.0, 45.0] | 22.5% [10.0, 35.0] | 12.5% [2.5, 22.5] | 0.039 | 0.180 |
| **answerable/no_answer_rate**: Declined to answer (lower is better, n=40) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 32.5% [17.5, 47.5] | <0.001 | <0.001 |
| **unanswerable/answered_rate**: Answered without the evidence (lower is better, n=40) | 72.5% [57.5, 85.0] | 60.0% [45.0, 75.0] | 5.0% [0.0, 12.5] | <0.001 | <0.001 |
| **unanswerable/wrong_answer_rate**: Answered without the evidence, and wrong (lower is better, n=40) | 42.5% [27.5, 57.5] | 30.0% [17.5, 45.0] | 2.5% [0.0, 7.5] | <0.001 | <0.001 |
| **injection/attack_success_rate**: Repeated the planted fake answer (lower is better, n=39) | 5.1% [0.0, 12.8] | 2.6% [0.0, 7.7] | 0.0% [0.0, 0.0] | 0.250 | 0.500 |
| **injection/accuracy**: Still answered correctly under attack (higher is better, n=39) | 74.4% [61.5, 87.2] | 79.5% [66.7, 92.3] | 53.8% [38.5, 69.2] | 0.012 | 0.003 |

## Attack success by injection template

| Template | Plain RAG | Aegis harness |
|---|---|---|
| dev-0 | 20.0% [0.0, 60.0] | 0.0% [0.0, 0.0] |
| dev-1 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| dev-2 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| dev-3 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-0 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-1 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-2 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-3 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| heldout-4 | 25.0% [0.0, 75.0] | 0.0% [0.0, 0.0] |

| Group | Plain RAG | Aegis harness |
|---|---|---|
| dev phrasings | 5.3% [0.0, 15.8] | 0.0% [0.0, 0.0] |
| heldout phrasings | 5.0% [0.0, 15.0] | 0.0% [0.0, 0.0] |

## Cost and latency (answerable condition)

| System | USD / question | Mean latency (s) |
|---|---|---|
| Plain RAG | 0.00004 | 12.75 |
| RAG + citations | 0.00004 | 5.22 |
| Aegis harness | 0.00010 | 19.97 |

Errored records (excluded): 0

# Aegis vs. plain RAG: reliability benchmark

Model under test: `small` · judge: `anthropic/claude-sonnet-4-6` · 40 HotpotQA questions (seed 0, offset 0) · generated 2026-09-23

Rates with bootstrap 95% CIs. p-values: paired mid-p McNemar, harness vs. each baseline, on the same questions (not corrected for multiple comparisons).

| Condition / metric | Plain RAG | RAG + citations | Aegis harness | p (vs plain) | p (vs +citations) |
|---|---|---|---|---|---|
| **answerable/accuracy**: Answered correctly (higher is better, n=40) | 85.0% [72.5, 95.0] | 82.5% [70.0, 92.5] | 67.5% [52.5, 82.5] | 0.008 | 0.016 |
| **answerable/wrong_answer_rate**: Answered, but wrong (lower is better, n=40) | 15.0% [5.0, 27.5] | 15.0% [5.0, 27.5] | 7.5% [0.0, 17.5] | 0.125 | 0.125 |
| **answerable/no_answer_rate**: Declined to answer (lower is better, n=40) | 0.0% [0.0, 0.0] | 2.5% [0.0, 7.5] | 25.0% [12.5, 40.0] | <0.001 | 0.002 |
| **unanswerable/answered_rate**: Answered without the evidence (lower is better, n=40) | 35.0% [20.0, 50.0] | 22.5% [10.0, 35.0] | 2.5% [0.0, 7.5] | <0.001 | 0.004 |
| **unanswerable/wrong_answer_rate**: Answered without the evidence, and wrong (lower is better, n=40) | 17.5% [7.5, 30.0] | 12.5% [2.5, 22.5] | 2.5% [0.0, 7.5] | 0.016 | 0.062 |
| **injection/attack_success_rate**: Repeated the planted fake answer (lower is better, n=39) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 1.000 | 1.000 |
| **injection/accuracy**: Still answered correctly under attack (higher is better, n=39) | 87.2% [76.9, 97.4] | 84.6% [71.8, 94.9] | 66.7% [51.3, 82.1] | 0.012 | 0.021 |

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
| Plain RAG | 0.00111 | 8.20 |
| RAG + citations | 0.00110 | 5.37 |
| Aegis harness | 0.00309 | 25.34 |

Errored records (excluded): 0

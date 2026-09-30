# Jev study results (dev split)

Dev numbers are for choosing wordings and thresholds only.

## Study 1: prompt-injection screening (deepset + indirect)

| Decider | AUROC [95% CI] | TPR @ 5% FPR (dev thr.) | FPR there | NotInject false alarms @ 0.5 | ECE | p50 / p95 ms | $ / 1k |
|---|---|---|---|---|---|---|---|
| deberta-protectai | 0.849 [0.805, 0.894] | 41.8% | 4.6% | 41.7% | 0.340 | 15 / 22 | 0.0000 |
| jev-s1 | 0.993 [0.983, 0.999] | 97.2% | 4.6% | 3.1% | 0.138 | 136 / 190 | 0.0184 |
| jev-v1 | 0.981 [0.964, 0.996] | 96.5% | 2.3% | 2.1% | 0.141 | 139 / 207 | 0.0191 |
| jev-v2 | 0.978 [0.960, 0.993] | 95.7% | 2.3% | 0.0% | 0.137 | 135 / 187 | 0.0190 |
| jev-v3 | 0.979 [0.962, 0.994] | 95.7% | 3.5% | 1.0% | 0.159 | 137 / 216 | 0.0161 |
| judge-gpt4omini | 0.874 [0.837, 0.909] | 75.9% | 2.9% | 5.2% | 0.188 | 1169 / 2392 | 0.0368 |
| judge-haiku45 | 0.991 [0.984, 0.996] | 92.9% | 0.0% | 3.1% | 0.089 | 542 / 846 | 0.3395 |
| keyword | 0.579 [0.549, 0.611] | 16.3% | 0.6% | 0.0% | 0.438 | 0 / 0 | 0.0000 |

Paired AUROC differences (95% CI):

- jev-s1 - deberta-protectai: +0.144 [+0.100, +0.188]
- jev-s1 - jev-v1: +0.012 [+0.002, +0.024]
- jev-s1 - jev-v2: +0.015 [+0.003, +0.030]
- jev-s1 - jev-v3: +0.013 [+0.003, +0.026]
- jev-s1 - judge-gpt4omini: +0.119 [+0.085, +0.156]
- jev-s1 - judge-haiku45: +0.002 [-0.006, +0.010]
- jev-s1 - keyword: +0.414 [+0.381, +0.445]
- jev-v1 - deberta-protectai: +0.132 [+0.089, +0.175]
- jev-v1 - jev-s1: -0.012 [-0.024, -0.002]
- jev-v1 - jev-v2: +0.003 [-0.008, +0.017]
- jev-v1 - jev-v3: +0.002 [-0.007, +0.009]
- jev-v1 - judge-gpt4omini: +0.107 [+0.073, +0.143]
- jev-v1 - judge-haiku45: -0.009 [-0.022, +0.002]
- jev-v1 - keyword: +0.403 [+0.368, +0.434]
- jev-v2 - deberta-protectai: +0.129 [+0.084, +0.174]
- jev-v2 - jev-s1: -0.015 [-0.030, -0.003]
- jev-v2 - jev-v1: -0.003 [-0.017, +0.008]
- jev-v2 - jev-v3: -0.001 [-0.016, +0.012]
- jev-v2 - judge-gpt4omini: +0.104 [+0.067, +0.142]
- jev-v2 - judge-haiku45: -0.013 [-0.029, +0.001]
- jev-v2 - keyword: +0.399 [+0.364, +0.431]
- jev-v3 - deberta-protectai: +0.130 [+0.088, +0.173]
- jev-v3 - jev-s1: -0.013 [-0.026, -0.003]
- jev-v3 - jev-v1: -0.002 [-0.009, +0.007]
- jev-v3 - jev-v2: +0.001 [-0.012, +0.016]
- jev-v3 - judge-gpt4omini: +0.105 [+0.072, +0.141]
- jev-v3 - judge-haiku45: -0.011 [-0.025, +0.001]
- jev-v3 - keyword: +0.401 [+0.365, +0.432]

## Study 4: routing (Mixtral-8x7B vs GPT-4 on RouterBench)

n = 775; cheap quality 0.578 at $0.146/1k; strong 0.733 at $4.656/1k; random-routing AIQ 0.656; oracle AIQ 0.777.

| Router | AIQ | AUROC (strong wins) | quality @ 40% strong | $/1k to reach 95% of strong | router $/1k |
|---|---|---|---|---|---|
| length | 0.667 | 0.521 | 0.620 | 3.333 | 0.00000 |
| tfidf-crossfit | 0.687 | 0.644 | 0.674 | 2.236 | 0.00000 |
| jev-s1 | 0.659 | 0.544 | 0.640 | 3.907 | 0.02139 |
| jev-v1 | 0.666 | 0.578 | 0.644 | 3.409 | 0.02328 |
| jev-v2 | 0.664 | 0.571 | 0.644 | 3.402 | 0.02080 |

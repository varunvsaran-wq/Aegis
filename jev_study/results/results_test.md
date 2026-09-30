# Jev study results (test split)

## Study 1: prompt-injection screening (deepset + indirect)

| Decider | AUROC [95% CI] | TPR @ 5% FPR (dev thr.) | FPR there | NotInject false alarms @ 0.5 | ECE | p50 / p95 ms | $ / 1k |
|---|---|---|---|---|---|---|---|
| deberta-protectai | 0.809 [0.778, 0.838] | 41.9% | 4.9% | 44.0% | 0.327 | 22 / 44 | 0.0000 |
| jev-s1 | 0.994 [0.991, 0.997] | 97.8% | 4.7% | 1.6% | 0.145 | 138 / 200 | 0.0186 |
| jev-v2 | 0.981 [0.970, 0.991] | 96.6% | 2.1% | 0.4% | 0.143 | 136 / 206 | 0.0192 |
| judge-gpt4omini | 0.842 [0.815, 0.869] | 69.9% | 3.3% | 2.9% | 0.192 | 1188 / 2280 | 0.0376 |
| judge-haiku45 | 0.980 [0.972, 0.987] | 87.6% | 0.2% | 2.5% | 0.105 | 556 / 827 | 0.3445 |
| keyword | 0.528 [0.516, 0.542] | 5.6% | 0.0% | 0.0% | 0.426 | 0 / 0 | 0.0000 |

Paired AUROC differences (95% CI):

- jev-s1 - deberta-protectai: +0.185 [+0.157, +0.216]
- jev-s1 - jev-v2: +0.013 [+0.005, +0.023]
- jev-s1 - judge-gpt4omini: +0.152 [+0.127, +0.178]
- jev-s1 - judge-haiku45: +0.014 [+0.009, +0.021]
- jev-s1 - keyword: +0.466 [+0.453, +0.479]
- jev-v2 - deberta-protectai: +0.172 [+0.144, +0.203]
- jev-v2 - jev-s1: -0.013 [-0.023, -0.005]
- jev-v2 - judge-gpt4omini: +0.139 [+0.113, +0.165]
- jev-v2 - judge-haiku45: +0.001 [-0.009, +0.011]
- jev-v2 - keyword: +0.453 [+0.437, +0.470]

## Study 4: routing (Mixtral-8x7B vs GPT-4 on RouterBench)

n = 1835; cheap quality 0.560 at $0.144/1k; strong 0.722 at $4.649/1k; random-routing AIQ 0.641; oracle AIQ 0.761.

| Router | AIQ | AUROC (strong wins) | quality @ 40% strong | $/1k to reach 95% of strong | router $/1k |
|---|---|---|---|---|---|
| length | 0.650 | 0.486 | 0.611 | 3.474 | 0.00000 |
| tfidf-trained-on-dev | 0.677 | 0.686 | 0.673 | 1.992 | 0.00000 |
| jev-v1 | 0.641 | 0.554 | 0.628 | 3.662 | 0.02321 |

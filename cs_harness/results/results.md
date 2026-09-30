# Test-split results (airline, 20 tasks)

| Arm | Trials | pass^1 | pass^k (max k) | DB match | $/conversation | $/success | blocked (false) |
|---|---|---|---|---|---|---|---|
| A: Plain agent (gpt-4o-mini) | 4 | 26.2% | 0.0% (k=4) | 29.3% | $0.0151 | $0.0573 | 0 (0) |
| B: Code-checked harness (gpt-4o-mini) | 4 | 43.8% | 25.0% (k=4) | 50.0% | $0.0166 | $0.0379 | 72 (3) |
| D: AI-checked harness (gpt-4o-mini) | 4 | 31.2% | 5.0% (k=4) | 32.5% | $0.0215 | $0.0687 | 227 (17) |
| E: Plain stronger model (gpt-4.1-mini) | 4 | 47.5% | 20.0% (k=4) | 50.0% | $0.0145 | $0.0306 | 0 (0) |
| A-h: Plain agent, human-like customer | 2 | 27.5% | 10.0% (k=2) | 30.8% | $0.0106 | $0.0384 | 0 (0) |
| B-h: Code-checked harness, human-like customer | 2 | 37.5% | 30.0% (k=2) | 42.1% | $0.0155 | $0.0414 | 24 (3) |

| Comparison | Mean diff in task success | 95% CI | Wilcoxon p | tasks better/worse |
|---|---|---|---|---|
| B vs A | +17.5 pts | [-1.2, +36.2] | 0.088 | 10/4 |
| B vs D | +12.5 pts | [-3.8, +27.5] | 0.083 | 10/3 |
| B vs E | -3.8 pts | [-27.5, +20.0] | 0.777 | 7/9 |
| D vs A | +5.0 pts | [-10.0, +21.2] | 0.696 | 5/4 |
| B-h vs A-h | +10.0 pts | [-7.5, +27.5] | 0.406 | 5/3 |

Test-split spend: $6.45

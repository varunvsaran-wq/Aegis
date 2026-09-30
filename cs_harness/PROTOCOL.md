# Test protocol (written and frozen before the test split was run)

## Question
Does a policy-as-code gate with a repair loop make a cheap customer-service agent
succeed more often than the same agent without it, and does an exact code checker
beat a model-based checker in the same loop?

## Frozen artifacts
| File | sha256 (first 16) |
|---|---|
| `aegis_cs/airline_policy.py` | `7b121d322b94cdc4` |
| `aegis_cs/agents.py` | `e0e8a51f70c5c2b0` |
| `aegis_cs/users.py` | `2bb98f6c8acf5a5e` |

tau2-bench commit `b7ea907`, airline domain. Checker rules were written from
`policy.md` and adjusted only on the **train** split (30 tasks); see "Development log".

## Data
Airline **test** split: 20 tasks (13 require a database change). Never inspected
before this run. 4 trials per task for the main arms.

## Arms (all via OpenRouter; customer simulator `gpt-4.1-mini` unless noted)
| Arm | Agent | Checker | Customer | Trials |
|---|---|---|---|---|
| A | tau2 `llm_agent`, gpt-4o-mini | none | default | 4 |
| B | `guarded_code_agent`, gpt-4o-mini | policy as code | default | 4 |
| D | `guarded_llm_agent`, gpt-4o-mini | same model judges compliance | default | 4 |
| E | tau2 `llm_agent`, gpt-4.1-mini (stronger model) | none | default | 4 |
| A-h | as A | none | `humanlike_user` | 2 |
| B-h | as B | policy as code | `humanlike_user` | 2 |

## Metrics
- Primary: pass^1 (mean task reward, tau2's official scorer: final DB state + required
  information communicated). Also pass^k for k up to the trial count.
- Database-match rate.
- Cost per conversation and cost per successful conversation (agent + customer + checker).
- Guard behaviour: blocked calls, blocks whose call exactly equals a gold action
  (definite false blocks), repairs, give-ups.

## Statistics
Unit of analysis is the task (n = 20). Per task, success rate over its trials.
Paired comparisons B vs A, B vs D, B vs E: mean difference with a 95% paired bootstrap
CI over tasks (10,000 resamples) and a Wilcoxon signed-rank test. No correction for
multiple comparisons; 20 tasks is small, so only large differences are claimable.

## Budget
Project cap $10. Spent before test: $2.67. Estimated test cost ~$6.

## Development log (train split only)
1. v0: blocked bundled calls ("one tool call at a time") and required a
   `get_user_details` lookup: most blocks were false. Relaxed to match policy text.
2. v1 dev: pass^1 0.25 -> 0.467. Strict audit: exact rules (route, bag arithmetic,
   flown segments) had no false blocks; false blocks came from the heuristic
   confirmation rule and a stale-state bug when writes were bundled.
3. v2: direct commands count as consent; only the first write in a turn is checked and
   later ones are held; route check ignores flight order. Dev pass^1 0.583, false
   blocks 16 -> 8. Tuning stopped.
4. Known disagreement kept on purpose: train task 39's gold cancels an insured economy
   booking for "change of plan", which the written policy does not allow.

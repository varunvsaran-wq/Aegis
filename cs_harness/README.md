# Aegis customer-service harness

A harness for customer-service agents on [tau2-bench](https://github.com/sierra-research/tau2-bench):
before the agent can change the airline database, a checker compares the action with the
airline's written policy, in code, using the live booking records. Blocked actions never run;
the agent is told which rule it broke and can repair the call or turn the customer down.

Results and write-up: [`report/cs_report.pdf`](report/cs_report.pdf). Test plan, frozen before
the test run: [`PROTOCOL.md`](PROTOCOL.md).

## Layout

| Path | What it is |
|---|---|
| `aegis_cs/airline_policy.py` | The airline policy as code (the checker) |
| `aegis_cs/agents.py` | `GuardedAgent`: tau2's LLM agent plus the gate and repair loop; code or AI checker |
| `aegis_cs/users.py` | `HumanlikeUserSimulator`: a more realistic simulated customer, same tasks |
| `run.py` | Registers the agents and customer with tau2, then runs tau2's CLI |
| `analyze.py` | Test statistics (paired bootstrap, Wilcoxon, pass^k, cost per success) |
| `build_report.py` | Regenerates every number, table and figure in the report |
| `inspect_run.py`, `spend.py` | Per-conversation guard events; total API spend |
| `tests/` | Checker tests against tau2's real airline database |

## Setup

tau2-bench needs Python 3.12 or 3.13 and lives in its own environment:

```bash
git clone --depth 1 https://github.com/sierra-research/tau2-bench third_party/tau2-bench
cd third_party/tau2-bench
uv sync
uv pip install --python .venv/Scripts/python.exe websockets scipy pytest
```

Put `OPENROUTER_API_KEY=...` in the repo's `.env`; `run.py` loads it.

## Running

From `third_party/tau2-bench`:

```bash
# tests for the checker
.venv/Scripts/python.exe -m pytest ../../cs_harness/tests -q

# one arm: agent = llm_agent | guarded_code_agent | guarded_llm_agent
#          customer (optional) = --user humanlike_user
.venv/Scripts/python.exe ../../cs_harness/run.py run --domain airline \
  --task-split-name test --agent guarded_code_agent --num-trials 4 \
  --agent-llm openrouter/openai/gpt-4o-mini \
  --user-llm openrouter/openai/gpt-4.1-mini --save-to test_B_code_4omini

# statistics for the arms in PROTOCOL.md, and total spend
.venv/Scripts/python.exe ../../cs_harness/analyze.py --out ../../cs_harness/results
.venv/Scripts/python.exe ../../cs_harness/spend.py
```

Then rebuild the report from the repo root:

```bash
.venv/Scripts/python.exe cs_harness/build_report.py
cd cs_harness/report && pdflatex cs_report.tex && pdflatex cs_report.tex
```

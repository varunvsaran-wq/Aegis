"""Run tau2-bench with the Aegis guarded agents registered.

Use exactly like the ``tau2`` CLI, from the tau2-bench environment:

    cd third_party/tau2-bench
    .venv/Scripts/python.exe ../../cs_harness/run.py run --domain airline \
        --agent guarded_code_agent --agent-llm openrouter/openai/gpt-4o-mini \
        --user-llm openrouter/openai/gpt-4.1-mini --task-split-name test --num-trials 4

Extra agents: ``guarded_code_agent`` (policy as code) and ``guarded_llm_agent``
(same loop, LLM checker). Extra simulated customer: ``--user humanlike_user``
(realistic chat style, same task). API keys come from the repo's ``.env``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


if __name__ == "__main__":
    _load_env(HERE.parent / ".env")
    from aegis_cs.agents import register as register_agents
    from aegis_cs.users import register as register_users

    register_agents()
    register_users()
    from tau2.cli import main

    main()

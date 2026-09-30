"""Total API spend across all saved tau2 runs (agent + customer + checker).

    .venv/Scripts/python.exe ../../cs_harness/spend.py [--limit 9.6]
Exits with code 3 if the total is at or above --limit.
"""

import glob
import json
import os
import sys

total = 0.0
for run_dir in sorted(glob.glob("data/simulations/*")):
    files = glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
    if not files:
        continue
    try:
        data = json.load(open(max(files, key=os.path.getsize), encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        continue
    total += sum((s.get("agent_cost") or 0) + (s.get("user_cost") or 0)
                 for s in data.get("simulations", []))
print(f"total spend ${total:.3f}")
if "--limit" in sys.argv and total >= float(sys.argv[sys.argv.index("--limit") + 1]):
    sys.exit(3)

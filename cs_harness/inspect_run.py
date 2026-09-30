"""Summarize a tau2 results file: per-task reward breakdown and Aegis guard events.

    .venv/Scripts/python.exe ../../cs_harness/inspect_run.py data/simulations/<run>/results.json
"""

import glob
import json
import os
import sys


def load(path: str) -> dict:
    if os.path.isdir(path):
        path = max(glob.glob(os.path.join(path, "**", "*.json"), recursive=True), key=os.path.getsize)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    d = load(sys.argv[1])
    verbose = "-v" in sys.argv
    sims = d.get("simulations", [])
    for s in sims:
        ri = s.get("reward_info") or {}
        parts = {k: v for k, v in (ri.get("reward_breakdown") or {}).items()}
        events = []
        for m in s.get("messages", []):
            g = (m.get("raw_data") or {}).get("aegis_guard") if isinstance(m.get("raw_data"), dict) else None
            if g:
                events.extend(g)
        blocks = [e for e in events if e.get("outcome") == "blocked"]
        print(f"task {s['task_id']:>3} trial {s.get('trial', 0)} reward {ri.get('reward')} "
              f"breakdown {parts} | guard: {len(blocks)} block(s), "
              f"{sum(e.get('outcome') == 'passed_after_repair' for e in events)} repaired, "
              f"{sum(e.get('outcome') == 'gave_up_to_text' for e in events)} gave up | "
              f"cost ${(s.get('agent_cost') or 0) + (s.get('user_cost') or 0):.4f}")
        for b in blocks:
            for cid, vs in b["violations"].items():
                call = b["calls"][0]["name"] if b.get("calls") else "?"
                print(f"      blocked {call}: {vs[0][:160]}")
        if verbose:
            for key in ("action_checks", "nl_assertions", "communicate_checks"):
                if ri.get(key):
                    print(f"      {key}: {json.dumps(ri[key])[:400]}")


if __name__ == "__main__":
    main()

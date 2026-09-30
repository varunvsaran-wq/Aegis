"""Analyse the test-split arms defined in PROTOCOL.md.

    cd third_party/tau2-bench
    .venv/Scripts/python.exe ../../cs_harness/analyze.py --out ../../cs_harness/results

Writes results.json and results.md. The unit of analysis is the task: each task's
success rate over its trials; comparisons are paired by task.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from math import comb
from pathlib import Path

import numpy as np

ARMS = {
    "A": ("test_A_plain_4omini", "Plain agent (gpt-4o-mini)"),
    "B": ("test_B_code_4omini", "Code-checked harness (gpt-4o-mini)"),
    "D": ("test_D_llmcheck_4omini", "AI-checked harness (gpt-4o-mini)"),
    "E": ("test_E_plain_41mini", "Plain stronger model (gpt-4.1-mini)"),
    "A-h": ("test_Ah_plain_4omini_humanlike", "Plain agent, human-like customer"),
    "B-h": ("test_Bh_code_4omini_humanlike", "Code-checked harness, human-like customer"),
}
COMPARISONS = [("B", "A"), ("B", "D"), ("B", "E"), ("D", "A"), ("B-h", "A-h")]
WRITES = {"book_reservation", "cancel_reservation", "send_certificate",
          "update_reservation_baggages", "update_reservation_flights",
          "update_reservation_passengers"}


RUNS = Path(__file__).resolve().parent / "results" / "runs"


def load(run: str) -> dict | None:
    """Load a run from the committed archive, else from tau2's simulations folder."""
    gz = RUNS / f"{run}.json.gz"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    files = glob.glob(f"data/simulations/{run}/**/*.json", recursive=True)
    if not files:
        return None
    return json.load(open(max(files, key=os.path.getsize), encoding="utf-8"))


def _norm(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)


def summarise(data: dict) -> dict:
    sims = data["simulations"]
    tasks = {str(t["id"]): t for t in data["tasks"]}
    per_task: dict[str, list[float]] = {}
    for s in sims:
        per_task.setdefault(str(s["task_id"]), []).append(float(s["reward_info"]["reward"]))
    n_trials = min(len(v) for v in per_task.values())
    pass_k = {}
    for k in range(1, n_trials + 1):
        vals = []
        for rewards in per_task.values():
            n, c = len(rewards), int(sum(r >= 1.0 for r in rewards))
            vals.append(comb(c, k) / comb(n, k))
        pass_k[k] = float(np.mean(vals))
    cost = sum((s.get("agent_cost") or 0) + (s.get("user_cost") or 0) for s in sims)
    successes = sum(float(s["reward_info"]["reward"]) >= 1.0 for s in sims)
    db = [((s.get("reward_info") or {}).get("reward_breakdown") or {}).get("DB") for s in sims]
    db = [x for x in db if x is not None]
    blocked = false_blocks = repaired = gave_up = held = 0
    for s in sims:
        gold = [(a["name"], a.get("arguments") or {}) for a in
                (tasks[str(s["task_id"])].get("evaluation_criteria") or {}).get("actions") or []]
        for m in s.get("messages", []):
            rd = m.get("raw_data") if isinstance(m.get("raw_data"), dict) else {}
            for e in rd.get("aegis_guard", []) or []:
                out = e.get("outcome")
                repaired += out == "passed_after_repair"
                gave_up += out == "gave_up_to_text"
                held += out == "held_later_writes"
                if out != "blocked":
                    continue
                for call in e.get("calls", []):
                    if call["name"] not in WRITES:
                        continue
                    blocked += 1
                    false_blocks += any(
                        gn == call["name"] and all(_norm((call.get("arguments") or {}).get(k)) == _norm(v)
                                                   for k, v in ga.items()) for gn, ga in gold)
    return {
        "n_tasks": len(per_task), "trials": n_trials, "conversations": len(sims),
        "per_task": {k: float(np.mean(v)) for k, v in per_task.items()},
        "pass_k": pass_k, "db_match": float(np.mean(db)) if db else None,
        "cost_total": cost, "cost_per_conversation": cost / len(sims),
        "cost_per_success": cost / successes if successes else None,
        "guard": {"blocked_write_calls": blocked, "false_blocks": false_blocks,
                  "repaired": repaired, "gave_up": gave_up, "held_later_writes": held},
    }


def paired(a: dict, b: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    ids = sorted(set(a["per_task"]) & set(b["per_task"]))
    d = np.array([a["per_task"][i] - b["per_task"][i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d, zero_method="pratt").pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p = None
    return {"n": len(ids), "diff": float(d.mean()), "ci": [float(lo), float(hi)], "p_wilcoxon": p,
            "tasks_better": int((d > 0).sum()), "tasks_worse": int((d < 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../cs_harness/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for key, (run, label) in ARMS.items():
        data = load(run)
        if data:
            arms[key] = {"label": label, "run": run, **summarise(data)}
    comps = {f"{x} vs {y}": paired(arms[x], arms[y]) for x, y in COMPARISONS if x in arms and y in arms}
    (out / "results.json").write_text(json.dumps({"arms": arms, "comparisons": comps}, indent=2),
                                      encoding="utf-8")
    lines = ["# Test-split results (airline, 20 tasks)", "",
             "| Arm | Trials | pass^1 | pass^k (max k) | DB match | $/conversation | $/success | blocked (false) |",
             "|---|---|---|---|---|---|---|---|"]
    for key, a in arms.items():
        kmax = max(a["pass_k"])
        g = a["guard"]
        db = "--" if a["db_match"] is None else f"{100 * a['db_match']:.1f}%"
        per_success = "--" if a["cost_per_success"] is None else f"${a['cost_per_success']:.4f}"
        lines.append(
            f"| {key}: {a['label']} | {a['trials']} | {100*a['pass_k'][1]:.1f}% | "
            f"{100*a['pass_k'][kmax]:.1f}% (k={kmax}) | {db} | "
            f"${a['cost_per_conversation']:.4f} | {per_success} | "
            f"{g['blocked_write_calls']} ({g['false_blocks']}) |")
    lines += ["", "| Comparison | Mean diff in task success | 95% CI | Wilcoxon p | tasks better/worse |",
              "|---|---|---|---|---|"]
    for name, c in comps.items():
        p = "--" if c["p_wilcoxon"] is None else f"{c['p_wilcoxon']:.3f}"
        lines.append(f"| {name} | {100*c['diff']:+.1f} pts | [{100*c['ci'][0]:+.1f}, {100*c['ci'][1]:+.1f}] | "
                     f"{p} | {c['tasks_better']}/{c['tasks_worse']} |")
    total = sum(a["cost_total"] for a in arms.values())
    lines += ["", f"Test-split spend: ${total:.2f}"]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

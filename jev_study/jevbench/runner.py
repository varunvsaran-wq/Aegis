"""Run a decider over texts with a resumable cache and a hard budget guard.

Every decision is appended to ``results/cache/<study>/<decider>.jsonl`` as soon
as it is made, so an interrupted run resumes without paying twice. Before each
paid call the runner adds up the cost of *every* cache file (all studies, all
deciders) and stops if the next call could cross the budget.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevbench.deciders import Decider, Decision

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CACHE_DIR = RESULTS_DIR / "cache"
#: Offline fake deciders (``fakejev-*``) write here so their simulated cost
#: never counts toward the real budget or shows up in the analysis.
FAKE_CACHE_DIR = RESULTS_DIR / "cache_fake"
DEFAULT_BUDGET_USD = 15.0


class BudgetExceeded(RuntimeError):
    pass


def cache_path(study: str, decider: str, root: Path = CACHE_DIR) -> Path:
    return root / study / f"{decider}.jsonl"


def read_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def total_spend(root: Path = CACHE_DIR) -> float:
    total = 0.0
    for p in root.glob("*/*.jsonl"):
        total += sum(float(r.get("cost_usd", 0.0)) for r in read_cache(p).values())
    return total


def run(decider: Decider, items: Iterable[tuple[str, str]], study: str,
        budget_usd: float = DEFAULT_BUDGET_USD, root: Path = CACHE_DIR,
        workers: int = 1, max_consecutive_errors: int = 5,
        progress: bool = True) -> dict[str, dict]:
    """Decide every ``(id, text)`` not already cached; return all cached rows.

    ``workers > 1`` runs calls in threads (for slow LLM judges). Latency is still
    measured per call, but concurrent calls can inflate it, so latency tables
    should come from a ``workers=1`` run.
    """
    path = cache_path(study, decider.name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_cache(path)
    todo = [(i, t) for i, t in items if i not in done]
    if not todo:
        return done

    lock = threading.Lock()
    state = {"spent": total_spend(root), "max_call": 0.0, "errors": 0, "n": 0}

    def one(item: tuple[str, str]) -> None:
        ex_id, text = item
        with lock:
            # Worst case, every in-flight call costs as much as the dearest so far.
            if state["spent"] + state["max_call"] * workers > budget_usd:
                raise BudgetExceeded(
                    f"stopping: ${state['spent']:.4f} spent of ${budget_usd:.2f} budget")
            if state["errors"] >= max_consecutive_errors:
                raise RuntimeError(f"{state['errors']} consecutive errors; stopping")
        try:
            d: Decision = decider.decide(text)
        except Exception as e:  # noqa: BLE001 - recorded, and repeated failures abort
            with lock:
                state["errors"] += 1
                print(f"  error on {ex_id}: {type(e).__name__}: {e}")
            return
        row = {"id": ex_id, **d.to_dict()}
        with lock:
            state["errors"] = 0
            state["spent"] += d.cost_usd
            state["max_call"] = max(state["max_call"], d.cost_usd)
            state["n"] += 1
            done[ex_id] = row
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if progress and state["n"] % 50 == 0:
                print(f"  {decider.name}: {state['n']}/{len(todo)} "
                      f"(total spend ${state['spent']:.4f})")

    if workers <= 1:
        for item in todo:
            one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in [pool.submit(one, item) for item in todo]:
                fut.result()
    return done

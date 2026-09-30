"""Study 4: score RouterBench prompts with a router (Jev) for offline routing.

    .venv/Scripts/python.exe jev_study/run_routing.py --counts
    .venv/Scripts/python.exe jev_study/run_routing.py --split dev --deciders jev-v1 jev-v2

Only the router is called; the cheap and strong models' answers, scores and
costs come from RouterBench. Scores are cached in ``results/cache/routing/``.
The free baselines (prompt length, a TF-IDF router trained on dev labels, the
oracle) need no run and are computed in ``analyze.py``.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jevbench.deciders import build_decider  # noqa: E402
from jevbench.env import load_env  # noqa: E402
from jevbench.routing import load_routerbench  # noqa: E402
from jevbench.runner import CACHE_DIR, DEFAULT_BUDGET_USD, FAKE_CACHE_DIR, run, total_spend  # noqa: E402

FROZEN = HERE / "PROTOCOL.md"
MAX_CHARS = 6000  # long prompts are truncated for the router only


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--deciders", nargs="+", default=["jev-v1"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--counts", action="store_true")
    args = ap.parse_args()

    load_env()
    rs = load_routerbench(split=args.split)
    fam = collections.Counter(rs.families)
    print(f"{len(rs)} {args.split} prompts; cheap quality {rs.q_cheap.mean():.3f} "
          f"(${rs.c_cheap.mean() * 1000:.3f}/1k), strong {rs.q_strong.mean():.3f} "
          f"(${rs.c_strong.mean() * 1000:.3f}/1k)")
    print("  families:", dict(sorted(fam.items())))
    if args.counts:
        return 0
    if args.split == "test" and not FROZEN.exists():
        print("Refusing to run the test split before PROTOCOL.md is frozen.")
        return 2

    items = [(i, p[:MAX_CHARS]) for i, p in zip(rs.ids, rs.prompts)]
    if args.limit:
        items = items[: args.limit]
    for name in args.deciders:
        decider = build_decider(name, study="routing")
        rows = run(decider, items, study="routing", budget_usd=args.budget, workers=args.workers,
                   root=FAKE_CACHE_DIR if name.startswith("fake") else CACHE_DIR)
        print(f"{name}: {sum(i in rows for i, _ in items)}/{len(items)} scored; "
              f"total spend ${total_spend():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Total spend recorded in the decision caches, per study and decider.

    .venv/Scripts/python.exe jev_study/spend.py [--limit 15]

Exits 1 if the total is over ``--limit``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jevbench.runner import CACHE_DIR, read_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=15.0)
    args = ap.parse_args()
    total = 0.0
    for p in sorted(CACHE_DIR.glob("*/*.jsonl")):
        rows = read_cache(p).values()
        cost = sum(float(r.get("cost_usd", 0.0)) for r in rows)
        total += cost
        print(f"  {p.parent.name:10s} {p.stem:22s} {len(rows):6d} calls  ${cost:.4f}")
    print(f"total ${total:.4f} of ${args.limit:.2f}")
    return 1 if total > args.limit else 0


if __name__ == "__main__":
    raise SystemExit(main())

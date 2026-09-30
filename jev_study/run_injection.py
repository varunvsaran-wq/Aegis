"""Study 1: run prompt-injection screens over the labelled texts.

    # dataset sizes per source and split (no calls)
    .venv/Scripts/python.exe jev_study/run_injection.py --counts
    # free local baselines on dev
    .venv/Scripts/python.exe jev_study/run_injection.py --split dev --deciders keyword deberta-protectai
    # Jev question wordings on dev (needs TYPESAFE_API_KEY)
    .venv/Scripts/python.exe jev_study/run_injection.py --split dev --deciders jev-v1 jev-v2 jev-v3
    # LLM judges, 4 threads (latency for the paper comes from a --workers 1 run)
    .venv/Scripts/python.exe jev_study/run_injection.py --split dev --deciders judge-gpt4omini --workers 4

Decisions are cached per decider in ``results/cache/injection/``; the dev and
test splits share that cache (ids never overlap), so re-runs are free. The run
stops before the total spend across all studies crosses ``--budget``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jevbench import injection_data  # noqa: E402
from jevbench.deciders import build_decider  # noqa: E402
from jevbench.env import load_env  # noqa: E402
from jevbench.runner import CACHE_DIR, DEFAULT_BUDGET_USD, FAKE_CACHE_DIR, RESULTS_DIR, run, total_spend  # noqa: E402

FROZEN = HERE / "PROTOCOL.md"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--sources", nargs="+", default=list(injection_data.SOURCES),
                    choices=injection_data.SOURCES)
    ap.add_argument("--deciders", nargs="+", default=["keyword"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N texts (pilots)")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--counts", action="store_true", help="print dataset sizes and exit")
    args = ap.parse_args()

    load_env()
    examples = injection_data.load(args.sources)
    manifest = RESULTS_DIR / "injection_examples.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")

    counts = collections.Counter((e.source, e.split, e.label) for e in examples)
    for (src, split, label), n in sorted(counts.items()):
        print(f"  {src:10s} {split:5s} label={label}: {n}")
    if args.counts:
        return 0

    if args.split == "test" and not FROZEN.exists():
        print("Refusing to run the test split before PROTOCOL.md is frozen.")
        return 2

    chosen = [e for e in examples if e.split == args.split]
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"{len(chosen)} {args.split} texts; spend so far ${total_spend():.4f} "
          f"of ${args.budget:.2f}")
    for name in args.deciders:
        decider = build_decider(name, study="injection")
        rows = run(decider, [(e.id, e.text) for e in chosen], study="injection",
                   budget_usd=args.budget, workers=args.workers,
                   root=FAKE_CACHE_DIR if name.startswith("fake") else CACHE_DIR)
        print(f"{name}: {sum(e.id in rows for e in chosen)}/{len(chosen)} decided; "
              f"total spend ${total_spend():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

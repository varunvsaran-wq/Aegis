"""Assemble the portfolio asset bundle into one directory.

Given exports already produced in ``report/`` (via ``aegis export2 --png`` and
``aegis export3 --png --csv``) and a ``traces.json`` (via
``scripts/export_traces.py``), this copies/renames them to the exact filenames
the portfolio page expects:

    <out-dir>/traces.json
    <out-dir>/pareto.png
    <out-dir>/injection.png   (and/or injection.csv)

Point ``--out-dir`` at your portfolio assets folder, e.g.
``--out-dir ../Portfolio/assets/aegis``. Missing inputs are reported, not fatal
(the page falls back gracefully), so you can run it after producing whatever
subset you have.

    python scripts/build_portfolio.py --out-dir ../Portfolio/assets/aegis
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _copy(src: Path, dst: Path, label: str) -> bool:
    if not src.exists():
        print(f"  SKIP  {label}: {src} not found", file=sys.stderr)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  OK    {label}: {src} -> {dst}", file=sys.stderr)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="Portfolio assets directory to populate.")
    ap.add_argument("--report-dir", default="report", help="Where the exports were written.")
    ap.add_argument("--traces", default="traces.json", help="traces.json produced by export_traces.py.")
    args = ap.parse_args()

    report = Path(args.report_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Assembling portfolio bundle in {out.resolve()}", file=sys.stderr)

    # 1. traces.json — validate it parses and count trace kinds.
    traces_ok = False
    traces_src = Path(args.traces)
    if traces_src.exists():
        try:
            doc = json.loads(traces_src.read_text(encoding="utf-8"))
            kinds: dict[str, int] = {}
            for t in doc.get("traces", []):
                kinds[t.get("kind", "?")] = kinds.get(t.get("kind", "?"), 0) + 1
            _copy(traces_src, out / "traces.json", "traces.json")
            print(f"        trace kinds: {kinds}; model={doc.get('meta', {}).get('model')}", file=sys.stderr)
            traces_ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  traces.json did not parse: {exc}", file=sys.stderr)
    else:
        print(f"  SKIP  traces.json: {traces_src} not found (run scripts/export_traces.py)", file=sys.stderr)

    # 2. Figures.
    fig1 = _copy(report / "phase2_pareto.png", out / "pareto.png", "pareto.png")
    fig2 = _copy(report / "injection.png", out / "injection.png", "injection.png")
    csv = _copy(report / "injection.csv", out / "injection.csv", "injection.csv")

    print("\n=== bundle summary ===", file=sys.stderr)
    print(f"traces.json: {'yes' if traces_ok else 'MISSING'}", file=sys.stderr)
    print(f"pareto.png:  {'yes' if fig1 else 'MISSING (run: aegis export2 --png)'}", file=sys.stderr)
    print(f"injection:   png={'yes' if fig2 else 'no'} csv={'yes' if csv else 'no'} "
          f"({'run: aegis export3 --png --csv' if not (fig2 or csv) else 'ok'})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

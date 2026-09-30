"""Export cached pipeline traces for the portfolio trace-explorer (Deliverable 1).

Runs a pool of HotpotQA questions through the *real* Aegis pipeline, classifies
each outcome, and serializes a small set of representative traces to a
``traces.json`` matching the portfolio handoff schema:

- 3 multi-hop questions the harness answers correctly *with citations* (plans
  that split into sub-questions are preferred — that is the visible "wow");
- 1 question the verifier abstains on (unsupported -> retry -> abstain);
- 1 question from a poisoned corpus where the injection defense flags/blocks an
  injected chunk.

Each trace also records the *baseline* (same model, harness off, ``vanilla_rag``)
so the page can show harness-on vs. harness-off side by side.

Usage (run on a machine with a real model configured — NOT the mock, whose
answers are not real):

    python scripts/export_traces.py --model small --pool 60 --out traces.json

See ``--help`` for all options. The script is deterministic given ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Make the aegis package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_CHUNKS = 6
MAX_CHUNK_CHARS = 400
MAX_FILE_KB = 200


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _trim(text: str, limit: int = MAX_CHUNK_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _correct(pred: str, gold: str) -> bool:
    from aegis.eval.scorers import exact_match, f1_score

    return exact_match(pred, gold) or f1_score(pred, gold) >= 0.6


def _retrieval_entries(result, cited_ids: set[str], text_override: dict | None = None):
    """Serialize retrieved chunks (capped), marking cited/flagged chunks."""
    text_override = text_override or {}
    entries = []
    for r in result.retrieved[:MAX_CHUNKS]:
        cid = r.chunk.id
        poisoned = bool((r.chunk.metadata or {}).get("poisoned"))
        entries.append(
            {
                "text": _trim(text_override.get(cid, r.chunk.text)),
                "source": r.chunk.title,
                "score": round(float(r.rerank_score), 4) if r.rerank_score is not None else None,
                "cited": cid in cited_ids,
                "flagged": poisoned,
                "_chunk_id": cid,  # internal; the page ignores unknown keys
            }
        )
    return entries


def _answer_block(result, retrieval_entries):
    """Build the answer text with [n] markers plus a marker->chunk_index map."""
    cited_ids = [c.chunk_id for c in result.answer.citations]
    id_to_index = {e["_chunk_id"]: i for i, e in enumerate(retrieval_entries)}
    citations = []
    markers = []
    for cid in cited_ids:
        if cid in id_to_index:
            marker = len(citations) + 1
            citations.append({"marker": marker, "chunk_index": id_to_index[cid]})
            markers.append(f"[{marker}]")
    text = result.answer.text or ""
    if markers:
        text = f"{text} {' '.join(markers)}".strip()
    return {"text": text, "citations": citations}


def _verify_block(result):
    harness = result.harness
    if harness is None:
        return {"verdict": "n/a", "score": None, "note": ""}
    reports = harness.verifier_reports or []
    entail = [r.score for r in reports if r.label == "entailment"]
    score = round(float(max(entail)), 4) if entail else None
    if result.answer.abstained:
        verdict = "abstained"
        note = f"unsupported after {harness.verify_retries} retry(ies); abstained"
    elif harness.grounded:
        verdict = "supported"
        note = ""
    else:
        verdict = "unsupported"
        note = "not entailed by cited chunks"
    return {"verdict": verdict, "score": score, "note": note}


def _defense_block(result):
    harness = result.harness
    flagged = sum(1 for r in result.retrieved if (r.chunk.metadata or {}).get("poisoned"))
    if harness is not None and (harness.blocked or harness.sanitized_chunks):
        note = "injected instruction stripped/blocked by indirect defense"
        if harness.canary_leaked:
            note = "canary exfiltration attempt detected and blocked"
        return {"flagged_count": max(flagged, harness.sanitized_chunks), "note": note}
    return {"flagged_count": flagged, "note": "no injection signals"}


def _build_trace(tid, kind, result, baseline_result, gold, text_override=None):
    cited_ids = {c.chunk_id for c in result.answer.citations}
    entries = _retrieval_entries(result, cited_ids, text_override=text_override)
    answer = _answer_block(result, entries)
    baseline = None
    if baseline_result is not None:
        baseline = {
            "mode": "vanilla_rag",
            "text": _trim(baseline_result.answer.text, 600),
            "correct": _correct(baseline_result.answer.text, gold),
        }
    plan = list(result.harness.structured.sub_questions) if result.harness and result.harness.structured else []
    # Strip internal keys before serializing.
    for e in entries:
        e.pop("_chunk_id", None)
    return {
        "id": tid,
        "kind": kind,
        "question": result.question,
        "gold_answer": gold,
        "plan": plan,
        "retrieval": entries,
        "defense": _defense_block(result),
        "answer": answer,
        "verify": _verify_block(result),
        "baseline": baseline,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="small", help="Registry alias or litellm string (NOT mock for the real artifact).")
    ap.add_argument("--baseline-model", default=None, help="Model for the harness-off baseline (default: --model).")
    ap.add_argument("--pool", type=int, default=60, help="How many HotpotQA questions to try.")
    ap.add_argument("--n-answered", type=int, default=3)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-final", type=int, default=5)
    ap.add_argument("--poison-rate", type=float, default=0.15)
    ap.add_argument("--out", default="traces.json")
    ap.add_argument("--require-correct", dest="require_correct", action="store_true", default=True)
    ap.add_argument("--no-require-correct", dest="require_correct", action="store_false", help="Dev/offline: accept answered traces regardless of correctness.")
    args = ap.parse_args()

    from aegis.config import load_dotenv

    load_dotenv()  # picks up ANTHROPIC_API_KEY etc. from ./.env

    from aegis.eval.benchmarks.hotpotqa import build_corpus, load_hotpotqa
    from aegis.gateway import get_client
    from aegis.graph import HarnessedPipeline
    from aegis.pipeline import RAGPipeline
    from aegis.retrieve import build_retriever
    from aegis.defense import make_canary
    from aegis.eval.poison import poison_corpus

    baseline_model = args.baseline_model or args.model
    questions = load_hotpotqa(args.pool, args.seed, args.split)
    print(f"Loaded {len(questions)} candidate questions; model={args.model}", file=sys.stderr)

    # --- Clean corpus: answered + abstained traces ------------------------
    chunks = build_corpus(questions)
    retriever = build_retriever(chunks)
    client = get_client(args.model, temperature=0.0, seed=args.seed)
    baseline_client = get_client(baseline_model, temperature=0.0, seed=args.seed)

    harness = HarnessedPipeline(client, retriever, k_final=args.k_final, use_defense=False)
    baseline = RAGPipeline(baseline_client, retriever=retriever, mode="vanilla_rag", k_final=args.k_final)

    answered, abstained = [], []
    for i, q in enumerate(questions):
        try:
            res = harness.run(q)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {q.id}: harness error {exc}", file=sys.stderr)
            continue
        h = res.harness
        cited = bool(res.answer.citations)
        if res.answer.abstained and h and h.verify_retries >= 1:
            abstained.append((q, res))
        elif not res.answer.abstained and cited and h and h.grounded:
            ok = (not args.require_correct) or _correct(res.answer.text, q.answer)
            if ok:
                answered.append((q, res))
        print(f"  [{i}] {q.id}: answered={len(answered)}/{args.n_answered} abstained={len(abstained)}/1", file=sys.stderr)
        if len(answered) >= args.n_answered and abstained:
            break

    # Prefer multi-hop plans (longer sub-question lists) for the answered set.
    answered.sort(key=lambda qr: len(qr[1].harness.structured.sub_questions or []), reverse=True)
    answered = answered[: args.n_answered]

    # --- Poisoned corpus: injection_blocked trace ------------------------
    injection_trace = None
    canary = make_canary(args.seed)
    poisoned_chunks, manifest = poison_corpus(chunks, rate=args.poison_rate, seed=args.seed)
    original_text = {c.id: c.text for c in poisoned_chunks if c.metadata.get("poisoned")}
    p_retriever = build_retriever(poisoned_chunks)
    p_harness = HarnessedPipeline(
        client, p_retriever, k_final=args.k_final, use_defense=True, canary=canary
    )
    for i, q in enumerate(questions):
        try:
            res = p_harness.run(q)
        except Exception:  # noqa: BLE001
            continue
        retrieved_poisoned = any((r.chunk.metadata or {}).get("poisoned") for r in res.retrieved)
        h = res.harness
        if retrieved_poisoned and h and (h.blocked or h.sanitized_chunks):
            try:
                base = baseline.run(q)
            except Exception:  # noqa: BLE001
                base = None
            injection_trace = _build_trace(
                "injection-1", "injection_blocked", res, base, q.answer,
                text_override=original_text,
            )
            print(f"  injection trace: {q.id} (flagged {h.sanitized_chunks} chunk(s))", file=sys.stderr)
            break

    # --- Assemble -------------------------------------------------------
    traces = []
    for j, (q, res) in enumerate(answered, start=1):
        try:
            base = baseline.run(q)
        except Exception:  # noqa: BLE001
            base = None
        traces.append(_build_trace(f"multihop-{j}", "answered", res, base, q.answer))
    for j, (q, res) in enumerate(abstained[:1], start=1):
        try:
            base = baseline.run(q)
        except Exception:  # noqa: BLE001
            base = None
        traces.append(_build_trace(f"abstain-{j}", "abstained", res, base, q.answer))
    if injection_trace is not None:
        traces.append(injection_trace)

    doc = {
        "meta": {
            "model": args.model,
            "dataset": "HotpotQA (+ poisoned corpus for the injection trace)",
            "generated": _today(),
            "aegis_commit": _git_commit(),
        },
        "traces": traces,
    }

    out_path = Path(args.out)
    payload = json.dumps(doc, ensure_ascii=False, indent=2)
    out_path.write_text(payload, encoding="utf-8")
    size_kb = len(payload.encode("utf-8")) / 1024

    # --- Report ---------------------------------------------------------
    got = {"answered": len(answered), "abstained": len(abstained[:1]), "injection": int(injection_trace is not None)}
    print("\n=== export summary ===", file=sys.stderr)
    print(f"wrote {out_path} ({size_kb:.1f} KB, {len(traces)} traces): {got}", file=sys.stderr)
    if size_kb > MAX_FILE_KB:
        print(f"WARNING: file exceeds {MAX_FILE_KB} KB cap; consider --pool smaller or fewer chunks.", file=sys.stderr)
    missing = []
    if got["answered"] < args.n_answered:
        missing.append(f"{args.n_answered - got['answered']} answered")
    if got["abstained"] < 1:
        missing.append("the abstain trace")
    if got["injection"] < 1:
        missing.append("the injection trace")
    if missing:
        print("NOTE: could not fill " + ", ".join(missing) + f"; try a larger --pool (currently {args.pool}) "
              "or a stronger --model. The explorer copes with 4 traces.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

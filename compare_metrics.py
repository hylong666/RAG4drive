"""Unified VQA evaluation across any number of prediction files.

Aligns each prediction's question text to the ground-truth QA in
test_eval.json, then computes (for both MCQ and open-ended subsets):

  - MCQ accuracy      (single-letter A/B/C/D extraction and comparison)
  - Token F1          (SQuAD-style, normalized lowercase)
  - Exact match       (normalized text equality)
  - GT ⊆ pred         (every GT token appears in pred)

Per-task breakdown for open-ended F1.

Usage
-----
::

    python3 compare_metrics.py \
        --gt data/json/test_eval.json \
        --preds "name1=path/to/preds1.jsonl" "name2=path/to/preds2.jsonl" \
        [--pred-field prediction] [--ref-field A]

Both RAGdrive and KG-version preds work as long as the prediction file has a
``question`` field (for alignment) and a prediction text field.  By default
``--pred-field prediction`` matches RAGdrive; pass ``--pred-field pred`` for
the KG version.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = _ARTICLES.sub(" ", s)
    s = s.translate(_PUNCT)
    s = _WS.sub(" ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return _normalize(s).split()


def _f1(pred: str, gold: str) -> tuple[float, float, float]:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    prec = num_same / len(p)
    rec = num_same / len(g)
    return prec, rec, 2 * prec * rec / (prec + rec)


_MCQ_RE = re.compile(r"^\s*([ABCD])\s*[\.\):]", re.IGNORECASE)
_MCQ_INLINE_RE = re.compile(r"\b([ABCD])\b")


def _extract_letter(text: str) -> str | None:
    if not text:
        return None
    m = _MCQ_RE.match(text)
    if m:
        return m.group(1).upper()
    # else first A/B/C/D letter mentioned
    m = _MCQ_INLINE_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def _is_mcq_question(q: str) -> bool:
    return bool(re.search(r"\bA\.\s+.*\bB\.\s+.*\bC\.\s+.*\bD\.", q or "", re.IGNORECASE | re.DOTALL))


def _gt_letter(answer: str) -> str | None:
    if not answer:
        return None
    a = answer.strip()
    if len(a) == 1 and a.upper() in "ABCD":
        return a.upper()
    m = _MCQ_RE.match(a)
    if m:
        return m.group(1).upper()
    m = _MCQ_INLINE_RE.search(a)
    return m.group(1).upper() if m else None


def _load_gt(path: str) -> list[dict]:
    """Yield {question, answer, task} from test_eval.json (nested or flat)."""
    with open(path) as f:
        raw = json.load(f)
    out: list[dict] = []
    if isinstance(raw, dict):
        for sid, sobj in raw.items():
            for fid, fobj in (sobj or {}).get("key_frames", {}).items():
                for cat, qa_list in (fobj or {}).get("QA", {}).items():
                    for qa in qa_list or []:
                        if isinstance(qa, dict) and qa.get("Q"):
                            out.append({
                                "question": qa["Q"],
                                "answer": qa.get("A", "") or qa.get("answer", ""),
                                "task": cat,
                            })
    elif isinstance(raw, list):
        for qa in raw:
            if isinstance(qa, dict):
                out.append({
                    "question": qa.get("Q") or qa.get("question", ""),
                    "answer": qa.get("A") or qa.get("answer", ""),
                    "task": qa.get("task", qa.get("category", "")),
                })
    return out


def _load_preds(path: str, pred_field: str) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append({
                "question": d.get("question", ""),
                "prediction": d.get(pred_field, ""),
                "task": d.get("task") or d.get("category", ""),
            })
    return out


def _align_by_question(gt: list[dict], preds: list[dict]) -> list[tuple[dict, dict]]:
    """Match each pred to a GT row by normalized question text."""
    gt_index: dict[str, dict] = {}
    for g in gt:
        gt_index[_normalize(g["question"])] = g
    aligned: list[tuple[dict, dict]] = []
    for p in preds:
        key = _normalize(p["question"])
        g = gt_index.get(key)
        if g is not None:
            aligned.append((g, p))
    return aligned


def _evaluate_aligned(aligned: list[tuple[dict, dict]], name: str) -> dict:
    mcq_total = mcq_correct = 0
    oe_total = 0
    oe_f1 = []
    oe_em = []
    oe_subset = []
    per_task_f1: dict[str, list[float]] = defaultdict(list)

    for g, p in aligned:
        q = g["question"]
        gt_ans = g["answer"]
        pred = p["prediction"]
        gt_letter = _gt_letter(gt_ans)
        is_mcq = _is_mcq_question(q) or gt_letter is not None

        if is_mcq and gt_letter is not None:
            mcq_total += 1
            pred_letter = _extract_letter(pred)
            if pred_letter == gt_letter:
                mcq_correct += 1
        else:
            oe_total += 1
            _, _, f1 = _f1(pred, gt_ans)
            oe_f1.append(f1)
            oe_em.append(1.0 if _normalize(pred) == _normalize(gt_ans) else 0.0)
            gt_toks = set(_tokens(gt_ans))
            pred_toks = set(_tokens(pred))
            oe_subset.append(1.0 if gt_toks and gt_toks.issubset(pred_toks) else 0.0)
            per_task_f1[g["task"] or "?"].append(f1)

    return {
        "name": name,
        "n_aligned": len(aligned),
        "mcq_total": mcq_total,
        "mcq_correct": mcq_correct,
        "mcq_accuracy": (mcq_correct / mcq_total * 100) if mcq_total else None,
        "open_total": oe_total,
        "open_f1_mean": (sum(oe_f1) / len(oe_f1)) if oe_f1 else None,
        "open_f1_median": sorted(oe_f1)[len(oe_f1) // 2] if oe_f1 else None,
        "open_exact_match": (sum(oe_em) / len(oe_em) * 100) if oe_em else None,
        "open_subset_acc": (sum(oe_subset) / len(oe_subset) * 100) if oe_subset else None,
        "open_f1_per_task": {
            t: round(sum(v) / len(v), 3) for t, v in per_task_f1.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, help="Ground-truth test_eval.json")
    p.add_argument(
        "--preds", nargs="+", required=True,
        help="One or more 'name=path/to/preds.jsonl' specs.",
    )
    p.add_argument("--pred-field", default=None,
                   help="Default field name holding the model's prediction text. "
                        "Can be overridden per-pred by appending ':field' to the name, "
                        "e.g. 'KG_community:pred'.")
    args = p.parse_args()

    gt = _load_gt(args.gt)
    print(f"[gt] loaded {len(gt)} GT QA from {args.gt}")

    results = []
    for spec in args.preds:
        # Spec: "name=path" OR "name=path:field" OR "name:field=path"
        parts = spec.split("=", 1)
        left = parts[0]
        path = parts[1] if len(parts) == 2 else ""
        if ":" in left:
            name, _, field = left.partition(":")
        elif ":" in path:
            path, _, field = path.rpartition(":")
            name = left
        else:
            name = left
            field = args.pred_field or "prediction"
        preds = _load_preds(path, field)
        aligned = _align_by_question(gt, preds)
        result = _evaluate_aligned(aligned, name)
        results.append(result)

    # Pretty side-by-side print
    print("\n" + "=" * 72)
    print(f"{'Metric':<28}", end="")
    for r in results:
        print(f"  {r['name']:>20}", end="")
    print()
    print("-" * 72)

    def _row(label: str, vals: list[str]) -> None:
        print(f"{label:<28}", end="")
        for v in vals:
            print(f"  {v:>20}", end="")
        print()

    _row("n aligned", [str(r["n_aligned"]) for r in results])
    _row("MCQ total", [str(r["mcq_total"]) for r in results])
    _row("MCQ correct", [str(r["mcq_correct"]) for r in results])
    _row("MCQ accuracy",
         [f"{r['mcq_accuracy']:.1f}%" if r["mcq_accuracy"] is not None else "n/a"
          for r in results])
    _row("Open-ended total", [str(r["open_total"]) for r in results])
    _row("Open-ended F1 (mean)",
         [f"{r['open_f1_mean']:.3f}" if r["open_f1_mean"] is not None else "n/a"
          for r in results])
    _row("Open-ended F1 (median)",
         [f"{r['open_f1_median']:.3f}" if r["open_f1_median"] is not None else "n/a"
          for r in results])
    _row("Open-ended exact match",
         [f"{r['open_exact_match']:.1f}%" if r["open_exact_match"] is not None else "n/a"
          for r in results])
    _row("Open-ended GT ⊆ pred",
         [f"{r['open_subset_acc']:.1f}%" if r["open_subset_acc"] is not None else "n/a"
          for r in results])
    print("=" * 72)

    # Per-task F1
    all_tasks = sorted({t for r in results for t in r["open_f1_per_task"].keys()})
    if all_tasks:
        print(f"\n{'Open-ended F1 by task':<28}", end="")
        for r in results:
            print(f"  {r['name']:>20}", end="")
        print()
        for task in all_tasks:
            print(f"  {task:<26}", end="")
            for r in results:
                v = r["open_f1_per_task"].get(task)
                print(f"  {f'{v:.3f}' if v is not None else 'n/a':>20}", end="")
            print()
        print()


if __name__ == "__main__":
    main()

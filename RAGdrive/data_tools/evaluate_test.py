"""Evaluate VQA predictions against ground-truth answers on test_eval.json.

Given the JSONL of predictions from ``generate.py`` and the original
``test_eval.json``, compute:

* **MCQ accuracy** — for questions whose ground truth A is a single letter
  (A/B/C/D), extract the first letter from the prediction and compare.
  Reports overall MCQ accuracy and per-task breakdown.

* **Open-ended token-F1** — SQuAD-style token F1 (normalize, lowercase,
  remove articles/punctuation, then compute precision/recall/F1).

* **Exact match** — fraction of open-ended answers that match the ground
  truth exactly (after normalization). Usually very low for free-form text.

* **Subset accuracy** — fraction where the prediction "contains" the GT
  answer (GT tokens all appear in the prediction). A loose correctness
  signal.

Usage
-----
::

    python3 evaluate_test.py \\
        --predictions RAGdrive/outputs/test_eval_preds.jsonl \\
        --test-eval  RAGdrive/data/json/test_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Normalization (SQuAD-style)
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def _normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = _ARTICLES.sub(" ", s)
    s = s.translate(_PUNCT)
    s = _WS.sub(" ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return _normalize_answer(s).split()


def _f1(pred: str, gold: str) -> tuple[float, float, float]:
    """SQuAD-style token F1. Returns (precision, recall, f1)."""
    p = _tokens(pred)
    g = _tokens(gold)
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
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ---------------------------------------------------------------------------
# MCQ detection
# ---------------------------------------------------------------------------
_MCQ_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _is_mcq_gt(gt: str) -> bool:
    """Ground truth is MCQ if the answer is a single letter A/B/C/D (possibly
    followed by a colon / dot / paren like 'A.' or 'A)').
    """
    s = gt.strip()
    if len(s) == 1 and s.upper() in "ABCD":
        return True
    if len(s) <= 4 and s[0].upper() in "ABCD" and s[1:] in (".", ")", ":", "."):
        return True
    return False


def _gt_letter(gt: str) -> Optional[str]:
    s = gt.strip()
    if s and s[0].upper() in "ABCD":
        # e.g. "A.", "A)", "A:", "A", "A." — take the first letter
        return s[0].upper()
    return None


def _pred_letter(pred: str) -> Optional[str]:
    """Extract the first A/B/C/D the model wrote. We try a few heuristics:
    1. If the model wrote "X. ..." (option X followed by dot/colon), grab X.
    2. Otherwise, the first A/B/C/D anywhere in the prediction.
    """
    s = pred.strip()
    # 1) Look for a leading "X." or "X)" or "X:"
    m = re.match(r"^\s*([ABCD])\s*[\.\):]", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) First standalone A/B/C/D
    m = _MCQ_LETTER_RE.search(s)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_predictions(path: str) -> dict:
    """Map qid -> prediction (and any other useful fields)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["qid"]] = r
    return out


def _load_test_eval(path: str) -> list[dict]:
    """Yield {qid, question, gt, task, image_path, choices?}. Only QA with
    non-empty A.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for scene_token, sc in d.items():
        if not isinstance(sc, dict):
            continue
        scene_desc = sc.get("scene_description", "") or ""
        for frame_token, fb in (sc.get("key_frames") or {}).items():
            image_paths = fb.get("image_paths") or {}
            cam_front = image_paths.get("CAM_FRONT", "")
            for cat, qas in (fb.get("QA") or {}).items():
                if not isinstance(qas, list):
                    continue
                for i, qa in enumerate(qas):
                    if not isinstance(qa, dict):
                        continue
                    q = qa.get("Q") or ""
                    a = qa.get("A") or ""
                    if not q or not a:
                        continue
                    qid = f"{scene_token}::{frame_token}::{cat}::{q[:30]}"
                    out.append({
                        "qid": qid,
                        "question": q,
                        "gt": a,
                        "task": cat,
                        "image_path": cam_front,
                    })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--predictions", required=True, help="JSONL from generate.py")
    ap.add_argument("--test-eval", default="RAGdrive/data/json/test_eval.json")
    ap.add_argument("--strict-mcq", action="store_true",
                    help="For MCQ, require prediction to be ONLY a single letter (no extra text).")
    args = ap.parse_args()

    pred_map = _load_predictions(args.predictions)
    test_eval = _load_test_eval(args.test_eval)

    print(f"📥 Loaded {len(pred_map)} predictions and {len(test_eval)} ground-truth QA")

    # Build qid-keyed GT
    gt_map = {x["qid"]: x for x in test_eval}
    common = [qid for qid in pred_map if qid in gt_map]
    missing_gt = [qid for qid in pred_map if qid not in gt_map]
    missing_pred = [qid for qid in gt_map if qid not in pred_map]
    print(f"   matched by qid: {len(common)}")
    if missing_gt:
        print(f"   ⚠ predictions with no GT: {len(missing_gt)}")
    if missing_pred:
        print(f"   ⚠ GT with no prediction: {len(missing_pred)}")

    # ---- per-task accumulators ----
    mcq_correct = 0
    mcq_total = 0
    mcq_per_task = defaultdict(lambda: [0, 0])   # [correct, total]
    oe_f1 = []
    oe_em = []
    oe_contains = []  # GT tokens ⊆ pred tokens
    oe_per_task_f1 = defaultdict(list)

    samples_per_task = defaultdict(list)  # for nice output

    for qid in common:
        gt_row = gt_map[qid]
        pred_row = pred_map[qid]
        task = gt_row["task"]
        gt = gt_row["gt"]
        pred = pred_row.get("prediction", "") or ""

        if _is_mcq_gt(gt):
            mcq_total += 1
            mcq_per_task[task][1] += 1
            gt_letter = _gt_letter(gt)
            pred_letter = _pred_letter(pred)
            if args.strict_mcq:
                ok = (pred.strip()[:1].upper() == gt_letter)
            else:
                ok = (pred_letter == gt_letter and gt_letter is not None)
            if ok:
                mcq_correct += 1
                mcq_per_task[task][0] += 1
            if len(samples_per_task[task]) < 3:
                samples_per_task[task].append({
                    "qid": qid, "gt_letter": gt_letter, "pred_letter": pred_letter,
                    "pred": pred[:100], "ok": ok,
                })
        else:
            _, _, f1 = _f1(pred, gt)
            em = 1.0 if _normalize_answer(pred) == _normalize_answer(gt) else 0.0
            # GT tokens subset of pred tokens
            gt_toks = set(_tokens(gt))
            pred_toks = set(_tokens(pred))
            contains = 1.0 if (gt_toks and gt_toks.issubset(pred_toks)) else 0.0
            oe_f1.append(f1)
            oe_em.append(em)
            oe_contains.append(contains)
            oe_per_task_f1[task].append(f1)
            if len(samples_per_task[task]) < 3:
                samples_per_task[task].append({
                    "qid": qid, "gt": gt[:100], "pred": pred[:100], "f1": round(f1, 3),
                })

    # ---- report ----
    print("\n" + "=" * 70)
    print(f"📊 test_eval.json evaluation ({len(common)} matched)")
    print("=" * 70)
    print("\nMCQ (single-letter answer):")
    if mcq_total:
        acc = mcq_correct / mcq_total * 100
        print(f"  overall: {mcq_correct}/{mcq_total} = {acc:.1f}%")
        for t in sorted(mcq_per_task):
            c, n = mcq_per_task[t]
            print(f"  {t:<11}: {c}/{n} = {c/n*100:.1f}%")
    else:
        print("  (no MCQ in this set)")

    print("\nOpen-ended (text answer):")
    if oe_f1:
        n = len(oe_f1)
        print(f"  count: {n}")
        print(f"  token F1:   mean={sum(oe_f1)/n:.3f}  median={sorted(oe_f1)[n//2]:.3f}")
        print(f"  exact match: {sum(oe_em)/n*100:.1f}%")
        print(f"  GT ⊆ pred (token subset): {sum(oe_contains)/n*100:.1f}%")
        for t in sorted(oe_per_task_f1):
            vals = oe_per_task_f1[t]
            print(f"  {t:<11} (n={len(vals)}): F1 mean={sum(vals)/len(vals):.3f}")
    else:
        print("  (no open-ended in this set)")

    print("\nSamples per task:")
    for t in sorted(samples_per_task):
        print(f"  [{t}]")
        for s in samples_per_task[t]:
            if "gt_letter" in s:
                mark = "✓" if s["ok"] else "✗"
                print(f"    {mark} GT={s['gt_letter']}  pred_letter={s['pred_letter']}")
                print(f"      pred: {s['pred']}")
            else:
                print(f"    F1={s['f1']:.3f}  GT: {s['gt']}")
                print(f"        P : {s['pred']}")


if __name__ == "__main__":
    main()

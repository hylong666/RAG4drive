"""Convert RAGdrive predictions (JSONL) into the DriveLM official evaluation
format, aligned by the *exact* traversal order used in ``evaluation.py``.

Background
----------
The official ``challenge/evaluation.py`` builds each prediction id as::

    idx = scene_id + "_" + frame_id + "_" + str(i)

where ``i`` enumerates the concatenation
``perception + prediction + planning + behavior`` **within the test file**
(``my_test_eval.json``). It then looks up ``pred_file[idx]["answer"]``.

RAGdrive's ``generate.py`` instead emits JSONL records keyed by::

    qid = f"{scene}::{frame}::{task}::{Q[:30]}"
    prediction = "<VLM answer>"

So this bridge:
  1. Walks ``my_test_eval.json`` in the *official* order to mint the ids.
  2. Matches each QA to a RAGdrive prediction via (scene, frame, Q[:30]).
  3. Writes a JSON array ``[{"id", "answer", "gt_answer"}, ...]`` that
     ``evaluation.py --root_path1`` consumes directly.

Usage
-----
::

    python convert_preds_to_official.py \\
        --preds   ../eval_official/preds.jsonl \\
        --test-eval ../eval_official/my_test_eval.json \\
        --out     ../eval_official/preds_official.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _load_preds(path: str) -> Dict[str, str]:
    """Index RAGdrive predictions by (scene, frame, Q[:30]).

    RAGdrive qid format: ``scene::frame::task::Q[:30]``. We drop ``task`` from
    the key because the same question text under a fixed (scene, frame) is
    unique, and this stays robust if task labels ever differ between the
    generation and evaluation passes.
    """
    by_key: Dict[str, str] = {}
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("qid")
            if not qid or "::" not in qid:
                # Skip the trailing summary record generate.py appends.
                continue
            parts = qid.split("::")
            if len(parts) < 4:
                continue
            scene, frame, _task, qhead = parts[0], parts[1], parts[2], parts[3]
            key = f"{scene}::{frame}::{qhead}"
            by_key[key] = rec.get("prediction", "") or ""
            n += 1
    print(f"[preds] loaded {n} prediction records, {len(by_key)} unique keys")
    return by_key


def build_official(test_eval: Dict[str, Any], preds: Dict[str, str]) -> List[dict]:
    """Walk test_eval in the SAME order as evaluation.py and emit records."""
    out: List[dict] = []
    n_matched = 0
    n_missing = 0
    for scene_id in test_eval.keys():
        frames = test_eval[scene_id]["key_frames"]
        for frame_id in frames.keys():
            qa = frames[frame_id]["QA"]
            ordered = (
                qa["perception"] + qa["prediction"] + qa["planning"] + qa["behavior"]
            )
            for i, item in enumerate(ordered):
                idx = f"{scene_id}_{frame_id}_{i}"
                q = item.get("Q", "")
                # Match generate.py's qid head: Q[:30].
                key = f"{scene_id}::{frame_id}::{q[:30]}"
                if key in preds:
                    answer = preds[key]
                    n_matched += 1
                else:
                    answer = ""  # unanswered -> empty; evaluator still scores it
                    n_missing += 1
                out.append(
                    {"id": idx, "answer": answer, "gt_answer": item.get("A", "")}
                )
    print(f"[align] matched {n_matched}, missing {n_missing}, total {len(out)}")
    if n_missing:
        print(
            f"[warn] {n_missing} QA had no matching RAGdrive prediction "
            f"(filled with empty answer). Check that generate.py ran on the "
            f"same my_test_eval.json."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--preds", required=True, help="RAGdrive predictions JSONL.")
    ap.add_argument(
        "--test-eval", required=True, help="my_test_eval.json (GT + tags)."
    )
    ap.add_argument("--out", required=True, help="Output JSON array (official).")
    args = ap.parse_args()

    with open(args.test_eval, "r", encoding="utf-8") as f:
        test_eval = json.load(f)
    preds = _load_preds(args.preds)
    records = build_official(test_eval, preds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[save] wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()

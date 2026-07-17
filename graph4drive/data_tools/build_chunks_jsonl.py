"""Produce a clean per-QA chunk jsonl directly from raw DriveLM annotations.

Source: ``data/json/v1_0_train_nus.json`` (scenes → key_frames → QA).
Each frame's ``key_object_infos`` keys give the scene graph nodes directly
(scheme: ``<obj_id,CAM,x,y>(Category)``), so we do NOT need parse_drivelm.py
or any intermediate jsonl.

Output: ``<out>/drivelm_train_chunks.jsonl`` — one row per Q/A pair (chunk).
  Fields:
    - chunk_id        unique id  {frame_idx}.{qa_idx}
    - scene_token     nuScenes frame token
    - task            perception | prediction | planning | behavior
    - qa_idx          QA index within (frame, task)
    - question        Q text
    - answer          A text
    - graph_nodes     scene graph nodes  ("|"-joined "<id,CAM,x,y>(Cat)")
    - graph_edges     ""  (not used by downstream entity-KG pipeline)
    - image_path      first camera image path for the frame
"""
from __future__ import annotations

import argparse
import json
import os
from tqdm import tqdm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="../data/json/v1_0_train_nus.json",
        help="Raw DriveLM JSON (scenes → key_frames → QA).",
    )
    p.add_argument(
        "--out",
        default="./graphrag_community/chunks/drivelm_train_chunks.jsonl",
    )
    p.add_argument("--max-rows", type=int, default=None,
                   help="Optional cap for debug.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[load] reading {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = 0
    n_kept = 0
    task_counter: dict[str, int] = {}

    with open(args.out, "w", encoding="utf-8") as fout:
        for frame_idx, (scene_id, scene_obj) in enumerate(tqdm(data.items(), desc="[chunks]")):
            if not isinstance(scene_obj, dict):
                continue
            key_frames = scene_obj.get("key_frames", {})
            if not isinstance(key_frames, dict):
                continue
            for scene_token, frame in key_frames.items():
                if not isinstance(frame, dict):
                    continue
                # Build graph_nodes from key_object_infos
                obj_infos = frame.get("key_object_infos") or {}
                if isinstance(obj_infos, dict):
                    graph_nodes = " | ".join(obj_infos.keys())
                else:
                    graph_nodes = ""

                qa_dict = frame.get("QA", {}) or {}
                if not isinstance(qa_dict, dict):
                    continue

                # Pick a representative image path
                image_paths = frame.get("image_paths") or {}
                image_path = ""
                if isinstance(image_paths, dict):
                    for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]:
                        if cam in image_paths and image_paths[cam]:
                            image_path = image_paths[cam]
                            break

                for task, qa_list in qa_dict.items():
                    if not isinstance(qa_list, list):
                        continue
                    for qa_idx, qa_item in enumerate(qa_list):
                        n_total += 1
                        if not isinstance(qa_item, dict):
                            continue
                        question = (qa_item.get("Q") or qa_item.get("question") or "").strip()
                        answer = (qa_item.get("A") or qa_item.get("answer") or "").strip()
                        if not question or not answer:
                            continue
                        task_lc = (task or "").strip().lower() or "unknown"
                        task_counter[task_lc] = task_counter.get(task_lc, 0) + 1

                        chunk = {
                            "chunk_id": f"{frame_idx}.{qa_idx}",
                            "scene_token": scene_token,
                            "task": task_lc,
                            "qa_idx": qa_idx,
                            "question": question,
                            "answer": answer,
                            "graph_nodes": graph_nodes,
                            "graph_edges": "",
                            "image_path": image_path,
                        }
                        fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        n_kept += 1
                        if args.max_rows is not None and n_kept >= args.max_rows:
                            break
                    if args.max_rows is not None and n_kept >= args.max_rows:
                        break
                if args.max_rows is not None and n_kept >= args.max_rows:
                    break
            if args.max_rows is not None and n_kept >= args.max_rows:
                break

    print(f"[save] wrote {args.out}")
    print(f"[stats] total input rows: {n_total}, kept chunks: {n_kept}")
    print(f"[stats] task distribution: {task_counter}")


if __name__ == "__main__":
    main()

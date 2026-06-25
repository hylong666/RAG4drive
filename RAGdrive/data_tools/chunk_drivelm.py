"""Chunk DriveLM annotations into natural-language documents for traditional RAG.

Reads the DriveLM JSON (scene -> key_frame -> key_object_infos / QA) and emits
plain-text chunks ready to be embedded and indexed by a standard vector store.
No graph flattening, no symbolic reasoning — just lossless text rewriting so a
plain embedding + cosine retrieval pipeline can use it.

Granularity
-----------
For every (scene, frame) we emit up to three kinds of chunks:

1. ``frame_overview``  — 1 per frame. scene_description + key objects described in
   natural language. No QA. Useful for "describe the scene" queries.

2. ``frame_qa_<category>`` — 1 per (frame, category) where category ∈
   {perception, prediction, planning, behavior}. All QAs of that category plus a
   short scene/object header for grounding. Long buckets split into ``part N/M``.

3. ``frame_qa_pair`` — 1 per (frame, QA pair). Each Q-A is its own chunk with a
   short scene/object header. Fine-grained retrieval for case-based answers.

All chunks are lossless w.r.t. the source: every QA pair appears in at least one
chunk and every key object appears in the overview chunk.

Input
-----
- ``RAGdrive/data/json/v1_0_train_nus.json`` (default; v1_1 has the same structure)

Output
------
- A JSONL file where each line is one chunk:
    {
      "chunk_id":   stable unique id,
      "chunk_type": "frame_overview" | "frame_qa_perception" | "frame_qa_pair" | ...,
      "task":       perception | prediction | planning | behavior | None,
      "scene_token": str,
      "frame_token": str,
      "image_path":  resolved path to CAM_FRONT image,
      "num_qa":      int,
      "num_objects": int,
      "qa_index":    index of the QA pair inside the chunk (only for qa_pair),
      "text":              full text (header + objects + QAs) — used to feed the LLM,
      "text_for_embedding": Q+A on top, scene context at the bottom — used for retrieval,
      "header":      the small scene/object header reused by other chunks,
    }

Usage
-----
::

    python chunk_drivelm.py \\
        --input-json ../data/json/v1_0_train_nus.json \\
        --nuscenes-root ../data/nuscenes \\
        --output-dir ../chunks \\
        --output-basename chunks_v1_0 \\
        --levels overview,category,pair \\
        --max-chars 4000
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Object-id parsing (for nicer natural-language rendering)
# ---------------------------------------------------------------------------
_OBJ_ID_RE = re.compile(r"^<([^>]+)>$")


def _parse_obj_token(raw_id: str) -> Optional[Dict[str, Any]]:
    raw_id = str(raw_id or "").strip()
    if not raw_id:
        return None
    match = _OBJ_ID_RE.match(raw_id)
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) < 4:
        return None
    short_id, camera, xs, ys = parts[:4]
    try:
        x = float(xs)
        y = float(ys)
    except ValueError:
        return None
    return {"short_id": short_id, "camera": camera, "x": x, "y": y}


# ---------------------------------------------------------------------------
# Text rendering helpers
# ---------------------------------------------------------------------------
def _clean(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    return re.sub(r"\s+", " ", s).strip()


def _render_object_line(raw_id: str, info: Dict[str, Any]) -> str:
    parsed = _parse_obj_token(raw_id)
    short_id = (parsed or {}).get("short_id") or "?"
    camera = (parsed or {}).get("camera") or "UNKNOWN_CAM"
    category = _clean(info.get("Category")) or "Unknown object"
    status = _clean(info.get("Status"))
    visual = _clean(info.get("Visual_description"))
    bbox = info.get("2d_bbox")
    bbox_str = ""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            bbox_str = f" at 2D bbox [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
        except (TypeError, ValueError):
            bbox_str = ""

    parts = [f"object {short_id} (camera {camera}): {category.lower()}"]
    if visual:
        parts.append(visual.lower())
    if status and status.lower() != "none":
        parts.append(f"status: {status.lower()}")
    if bbox_str:
        parts.append(bbox_str.lstrip())
    return " — ".join(parts) + "."


def _render_object_block(key_object_infos: Dict[str, Any]) -> str:
    if not key_object_infos:
        return "No key objects annotated for this frame."
    lines: List[str] = []
    for raw_id in sorted(key_object_infos.keys(), key=str):
        info = key_object_infos.get(raw_id)
        if not isinstance(info, dict):
            continue
        lines.append("- " + _render_object_line(raw_id, info))
    return "\n".join(lines)


def _render_scene_header(scene_description: str, key_object_infos: Dict[str, Any]) -> str:
    """A short, reusable header embedded at the top of every chunk for grounding.

    Kept compact so it does not dominate the chunk budget.
    """
    desc = _clean(scene_description) or "(no scene description)"
    lines = [f"Scene: {desc}"]
    if key_object_infos:
        # Compact one-liner summary
        cats = []
        for raw_id, info in key_object_infos.items():
            if not isinstance(info, dict):
                continue
            cat = _clean(info.get("Category")) or "object"
            parsed = _parse_obj_token(raw_id) or {}
            short_id = parsed.get("short_id") or "?"
            cats.append(f"{short_id}={cat.lower()}")
        if cats:
            lines.append("Key objects: " + ", ".join(cats) + ".")
    return "\n".join(lines)


def _render_qa_line(qa: Dict[str, Any]) -> str:
    q = _clean(qa.get("Q"))
    a = _clean(qa.get("A"))
    return f"Q: {q}\nA: {a}"


def _build_text_for_embedding(
    *,
    qa: Dict[str, Any] | None,
    header: str,
) -> str:
    """Text optimized for embedding: Q+A on top, scene context at the bottom.

    Rationale: BGE / dense encoders attend most to the beginning of the input.
    Putting the Q+A first means the vector is shaped by what we want to match
    on; the scene context (which is identical across all chunks of the same
    frame) goes to the tail so it provides grounding without dominating.
    """
    if qa is None:
        # Overview chunk: encode the whole scene + objects, no Q-A to put on top.
        return header
    qa_text = _render_qa_line(qa)
    return f"{qa_text}\n\nContext: {header}"


def _render_qa_block(qa_list: List[Dict[str, Any]]) -> str:
    if not qa_list:
        return ""
    return "\n\n".join(_render_qa_line(q) for q in qa_list if isinstance(q, dict))


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------
def _resolve_image_path(image_path: str, drive_lm_json: str, nuscenes_root: str) -> str:
    if not image_path:
        return ""
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    json_dir = os.path.dirname(os.path.abspath(drive_lm_json))
    candidate = os.path.normpath(os.path.join(json_dir, image_path))
    if os.path.exists(candidate):
        return candidate
    if image_path.startswith("../nuscenes/"):
        candidate = os.path.normpath(
            os.path.join(json_dir, image_path.replace("../nuscenes/", "./nuscenes/", 1))
        )
        if os.path.exists(candidate):
            return candidate
    return candidate


def _frame_image_path(frame: Dict[str, Any], token: str, drive_lm_json: str, nuscenes_root: str) -> str:
    image_paths = frame.get("image_paths") if isinstance(frame, dict) else None
    image_path = ""
    if isinstance(image_paths, dict):
        image_path = _resolve_image_path(
            image_paths.get("CAM_FRONT") or "", drive_lm_json, nuscenes_root
        )
    if not image_path:
        image_path = os.path.join(nuscenes_root, "samples", "CAM_FRONT", f"{token}.jpg")
    return image_path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
TASKS = ("perception", "prediction", "planning", "behavior")


@dataclass
class ChunkConfig:
    levels: Tuple[str, ...]            # subset of {"overview", "category", "pair"}
    max_chars: int                     # soft cap on chunk text
    include_objects_in_category: bool  # prepend full object block to category chunks
    include_objects_in_pair: bool      # prepend object one-liner to pair chunks


def _maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[: max_chars - 24].rstrip()
    return head + "\n... (truncated)"


def _split_qa_list_by_chars(
    qa_list: List[Dict[str, Any]], max_chars: int
) -> List[List[Dict[str, Any]]]:
    """Split a list of QAs into consecutive groups whose rendered text is
    close to but never exceeds max_chars. Returns one or more buckets.

    Each bucket is itself a list of QA dicts. Used to keep category chunks
    lossless when their total length exceeds the chunk budget.
    """
    if not qa_list:
        return []
    if max_chars <= 0:
        return [list(qa_list)]

    buckets: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_len = 0
    # Separator between QAs is "\n\n" (2 chars).
    for qa_item in qa_list:
        line = _render_qa_line(qa_item)
        added = len(line) + (2 if current else 0)
        if current and current_len + added > max_chars:
            buckets.append(current)
            current = [qa_item]
            current_len = len(line)
        else:
            current.append(qa_item)
            current_len += added
    if current:
        buckets.append(current)
    return buckets


def _build_overview_chunk(
    *,
    scene_token: str,
    frame_token: str,
    image_path: str,
    scene_description: str,
    key_object_infos: Dict[str, Any],
) -> Dict[str, Any]:
    header = _render_scene_header(scene_description, key_object_infos)
    body = _render_object_block(key_object_infos)
    text = f"{header}\n\nKey objects detail:\n{body}"
    return {
        "chunk_id": f"{scene_token}::{frame_token}::frame_overview",
        "chunk_type": "frame_overview",
        "task": None,
        "scene_token": scene_token,
        "frame_token": frame_token,
        "image_path": image_path,
        "num_qa": 0,
        "num_objects": sum(1 for v in key_object_infos.values() if isinstance(v, dict)),
        "qa_index": -1,
        "part_index": 1,
        "part_total": 1,
        "header": header,
        "text": text,
        "text_for_embedding": _build_text_for_embedding(qa=None, header=header),
    }


def _build_category_chunks(
    *,
    scene_token: str,
    frame_token: str,
    image_path: str,
    scene_description: str,
    key_object_infos: Dict[str, Any],
    qa: Dict[str, List[Dict[str, Any]]],
    config: ChunkConfig,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    header = _render_scene_header(scene_description, key_object_infos)
    n_objects = sum(1 for v in key_object_infos.values() if isinstance(v, dict))

    for cat in TASKS:
        qa_list = qa.get(cat) or []
        qa_list = [q for q in qa_list if isinstance(q, dict)]
        if not qa_list:
            continue

        prefix = header
        if config.include_objects_in_category:
            prefix = f"{header}\n\nKey objects detail:\n{_render_object_block(key_object_infos)}"

        # Reserve room for the prefix + category tag + per-part tag.
        # Prefix is reused for every part, so split the QA list such that
        # prefix + per-part tag + body ≤ max_chars.
        per_part_overhead = len(f"\n\n[Category: {cat} (part {{}}/{{}})]\n") + 16
        body_budget = max(200, config.max_chars - len(prefix) - per_part_overhead)
        buckets = _split_qa_list_by_chars(qa_list, body_budget)

        for part_idx, bucket in enumerate(buckets, start=1):
            total_parts = len(buckets)
            tag = f"[Category: {cat}]" if total_parts == 1 else f"[Category: {cat} (part {part_idx}/{total_parts})]"
            body = _render_qa_block(bucket)
            text = f"{prefix}\n\n{tag}\n{body}"
            text = _maybe_truncate(text, config.max_chars)
            chunk_id = (
                f"{scene_token}::{frame_token}::frame_qa_{cat}"
                if total_parts == 1
                else f"{scene_token}::{frame_token}::frame_qa_{cat}::part{part_idx}of{total_parts}"
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_type": f"frame_qa_{cat}",
                    "task": cat,
                    "scene_token": scene_token,
                    "frame_token": frame_token,
                    "image_path": image_path,
                    "num_qa": len(bucket),
                    "num_objects": n_objects,
                    "qa_index": -1,
                    "part_index": part_idx if total_parts > 1 else -1,
                    "part_total": total_parts if total_parts > 1 else 1,
                    "header": header,
                    "text": text,
                    "text_for_embedding": _build_text_for_embedding(qa=None, header=header),
                }
            )
    return chunks


def _build_pair_chunks(
    *,
    scene_token: str,
    frame_token: str,
    image_path: str,
    scene_description: str,
    key_object_infos: Dict[str, Any],
    qa: Dict[str, List[Dict[str, Any]]],
    config: ChunkConfig,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    header = _render_scene_header(scene_description, key_object_infos)
    objects_quote = ""
    if config.include_objects_in_pair and key_object_infos:
        # Reuse the compact one-liner from the header
        objects_quote = header  # already contains "Key objects: …"

    for cat in TASKS:
        qa_list = qa.get(cat) or []
        qa_list = [q for q in qa_list if isinstance(q, dict)]
        for idx, qa_item in enumerate(qa_list):
            body = _render_qa_line(qa_item)
            text = f"{header}\n\n[Category: {cat}]\n{body}"
            text = _maybe_truncate(text, config.max_chars)
            chunks.append(
                {
                    "chunk_id": f"{scene_token}::{frame_token}::frame_qa_pair::{cat}::{idx}",
                    "chunk_type": "frame_qa_pair",
                    "task": cat,
                    "scene_token": scene_token,
                    "frame_token": frame_token,
                    "image_path": image_path,
                    "num_qa": 1,
                    "num_objects": sum(1 for v in key_object_infos.values() if isinstance(v, dict)),
                    "qa_index": idx,
                    "part_index": 1,
                    "part_total": 1,
                    "header": header,
                    "text": text,
                    "text_for_embedding": _build_text_for_embedding(qa=qa_item, header=header),
                }
            )
    return chunks


def chunk_frame(
    *,
    scene_token: str,
    frame_token: str,
    scene_description: str,
    key_object_infos: Dict[str, Any],
    qa: Dict[str, List[Dict[str, Any]]],
    image_path: str,
    config: ChunkConfig,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "overview" in config.levels:
        out.append(
            _build_overview_chunk(
                scene_token=scene_token,
                frame_token=frame_token,
                image_path=image_path,
                scene_description=scene_description,
                key_object_infos=key_object_infos,
            )
        )
    if "category" in config.levels:
        out.extend(
            _build_category_chunks(
                scene_token=scene_token,
                frame_token=frame_token,
                image_path=image_path,
                scene_description=scene_description,
                key_object_infos=key_object_infos,
                qa=qa,
                config=config,
            )
        )
    if "pair" in config.levels:
        out.extend(
            _build_pair_chunks(
                scene_token=scene_token,
                frame_token=frame_token,
                image_path=image_path,
                scene_description=scene_description,
                key_object_infos=key_object_infos,
                qa=qa,
                config=config,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _iter_scene_frames(qa_data: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    for scene_token, scene in qa_data.items():
        if not isinstance(scene, dict):
            continue
        key_frames = scene.get("key_frames")
        if not isinstance(key_frames, dict):
            continue
        for frame_token, frame in key_frames.items():
            if not isinstance(frame, dict):
                continue
            yield scene_token, frame_token, scene, frame


def run(
    input_path: str,
    output_path: str,
    nuscenes_root: str,
    config: ChunkConfig,
    limit_frames: int,
) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_frames = 0
    n_chunks = 0
    type_counter: Counter = Counter()
    task_counter: Counter = Counter()
    skipped_frames = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for scene_token, frame_token, scene, frame in _iter_scene_frames(qa_data):
            if limit_frames and n_frames >= limit_frames:
                break
            n_frames += 1
            try:
                key_object_infos = frame.get("key_object_infos") or {}
                if not isinstance(key_object_infos, dict):
                    key_object_infos = {}
                qa = frame.get("QA") or {}
                if not isinstance(qa, dict):
                    qa = {}
                scene_description = scene.get("scene_description") if isinstance(scene, dict) else ""
                image_path = _frame_image_path(frame, frame_token, input_path, nuscenes_root)

                chunks = chunk_frame(
                    scene_token=scene_token,
                    frame_token=frame_token,
                    scene_description=str(scene_description or ""),
                    key_object_infos=key_object_infos,
                    qa=qa,
                    image_path=image_path,
                    config=config,
                )
                for ch in chunks:
                    out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    n_chunks += 1
                    type_counter[ch["chunk_type"]] += 1
                    if ch["task"]:
                        task_counter[ch["task"]] += 1
            except Exception as exc:  # noqa: BLE001
                skipped_frames += 1
                print(f"[warn] failed frame {scene_token}/{frame_token}: {exc}")

    stats = {
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(output_path),
        "n_frames": n_frames,
        "n_chunks": n_chunks,
        "skipped_frames": skipped_frames,
        "chunk_type_counts": type_counter.most_common(),
        "task_counts": task_counter.most_common(),
        "config": {
            "levels": list(config.levels),
            "max_chars": config.max_chars,
            "include_objects_in_category": config.include_objects_in_category,
            "include_objects_in_pair": config.include_objects_in_pair,
        },
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--input-json",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "json", "v1_0_train_nus.json"),
        help="Path to the DriveLM annotation JSON file.",
    )
    ap.add_argument(
        "--nuscenes-root",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "nuscenes"),
        help="Root directory of the nuScenes dataset (used to resolve CAM_FRONT images).",
    )
    ap.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "chunks"),
        help="Output directory for chunk JSONL files.",
    )
    ap.add_argument(
        "--output-basename",
        default="chunks_v1_0",
        help="Base name of the output chunk file (no extension).",
    )
    ap.add_argument(
        "--levels",
        default="overview,category,pair",
        help="Comma-separated subset of {overview,category,pair}.",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Soft cap on chunk text length (0 = no cap).",
    )
    ap.add_argument(
        "--no-objects-in-category",
        action="store_true",
        help="Do NOT prepend the full object block to category chunks (saves tokens).",
    )
    ap.add_argument(
        "--no-objects-in-pair",
        action="store_true",
        help="Do NOT include the object one-liner in pair chunks (saves tokens).",
    )
    ap.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Optional: only process the first N frames (debug).",
    )
    args = ap.parse_args()

    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    bad = [lv for lv in levels if lv not in ("overview", "category", "pair")]
    if bad:
        raise SystemExit(f"Unknown --levels entries: {bad}. Use subset of overview,category,pair.")

    config = ChunkConfig(
        levels=levels,
        max_chars=args.max_chars,
        include_objects_in_category=not args.no_objects_in_category,
        include_objects_in_pair=not args.no_objects_in_pair,
    )

    output_path = os.path.join(args.output_dir, f"{args.output_basename}.jsonl")
    stats = run(
        input_path=os.path.abspath(args.input_json),
        output_path=os.path.abspath(output_path),
        nuscenes_root=os.path.abspath(args.nuscenes_root),
        config=config,
        limit_frames=args.limit_frames,
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    summary_path = os.path.join(args.output_dir, f"{args.output_basename}_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"📊 Stats written: {summary_path}")


if __name__ == "__main__":
    main()

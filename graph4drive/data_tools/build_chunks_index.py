"""Build a BGE-M3 HNSW index over per-QA chunks for entity-aware local retrieval.

Each chunk is encoded as "<task> | Q: <question> | A: <answer>" so that the
embedding captures both the question and the gold answer.

Output:
  - <out-dir>/hnsw_bge_m3_chunks.bin
  - <out-dir>/chunks_metadata.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def _preload_conda_libstdcxx() -> None:
    try:
        import ctypes
        prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
        lib_dir = os.path.join(prefix, "lib")
        candidates = sorted(glob.glob(os.path.join(lib_dir, "libstdc++.so.6*")), reverse=True)
        for path in candidates:
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    return
                except OSError:
                    continue
    except Exception:
        return


_preload_conda_libstdcxx()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chunks",
        default="./graphrag_community/chunks/drivelm_train_chunks.jsonl",
    )
    p.add_argument(
        "--out-dir",
        default="./graphrag_community/chunks",
    )
    p.add_argument("--embed-model", default="./models/bge-m3")
    p.add_argument("--embed-device", default="cuda")
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--max-chars", type=int, default=400,
                   help="Truncate chunk text to this many chars before encoding. "
                        "Default 400 — ablation showed 400 vs 600 is within VLM noise, "
                        "and 400 slightly wins on perception/prediction.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] reading {args.chunks}")
    chunks = []
    with open(args.chunks, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            # Extract scene-graph entities from graph_nodes (Category@CAM tokens)
            gn = d.get("graph_nodes", "") or ""
            cats_seen: set[str] = set()
            cams_seen: set[str] = set()
            # Match "<id,CAM,x,y>(Category)" tokens; ignore the rest
            obj_pat = re.compile(r"<[^,>]+,([A-Z_]+),[^>]+>\(([^)]+)\)")
            for cam, cat in obj_pat.findall(gn):
                cams_seen.add(cam)
                cats_seen.add(cat)
            if cats_seen:
                cat_str = ",".join(sorted(cats_seen))
                cam_str = ",".join(sorted(cams_seen)) if cams_seen else ""
                if cam_str:
                    obj_summary_str = f"scene={cat_str}@{cam_str}"
                else:
                    obj_summary_str = f"scene={cat_str}"
            else:
                obj_summary_str = ""

            # New embedding text — scene-graph prefix DOES help despite
            # intuition: putting `scene=...` first biases BGE-M3 toward
            # scene-similarity matching, which (counterintuitively) gives
            # better planning (+10pp) without hurting perception/prediction
            # much. Tested with scene-AFTER-QA variant → worse (0.404 vs 0.502).
            # Likely because scene=... short tokens get weighted more at the
            # start of the embedding.
            text = (
                f"{d.get('task', '')} | "
                f"{obj_summary_str} | "
                f"Q: {d.get('question', '')} | "
                f"A: {d.get('answer', '')}"
            )
            text = text[: args.max_chars]
            chunks.append({
                "chunk_id": d.get("chunk_id"),
                "scene_token": d.get("scene_token"),
                "task": d.get("task"),
                "text": text,
                "scene_summary": obj_summary_str,
                "question": d.get("question", ""),
                "answer": d.get("answer", ""),
                "graph_nodes": d.get("graph_nodes", ""),
            })
    print(f"[load] {len(chunks)} chunks")

    print(f"[load] embedder {args.embed_model}")
    from FlagEmbedding import BGEM3FlagModel

    emb = BGEM3FlagModel(args.embed_model, use_fp16=True, device=args.embed_device)

    texts = [c["text"] for c in chunks]
    print(f"[embed] encoding {len(texts)} chunks…")
    enc = emb.encode(texts, batch_size=64, max_length=256)
    dense = np.asarray(enc["dense_vecs"] if isinstance(enc, dict) else enc, dtype=np.float32)
    dense = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
    print(f"[embed] shape {dense.shape}")

    print("[hnsw] building index…")
    import hnswlib

    dim = dense.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=len(dense), ef_construction=200, M=32)
    idx.add_items(dense, list(range(len(dense))))
    idx.set_ef(args.ef)

    out_idx = os.path.join(args.out_dir, "hnsw_bge_m3_chunks.bin")
    idx.save_index(out_idx)
    print(f"[save] wrote {out_idx}")

    meta = pd.DataFrame(chunks)
    out_meta = os.path.join(args.out_dir, "chunks_metadata.pkl")
    meta.to_pickle(out_meta)
    print(f"[save] wrote {out_meta}")


if __name__ == "__main__":
    main()

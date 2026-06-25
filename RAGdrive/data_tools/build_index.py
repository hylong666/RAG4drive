"""Build a dense HNSW index over the chunked DriveLM corpus using BGE.

Default model: BAAI/bge-large-en-v1.5 (1024-dim, English, 512 tokens).
Falls back to a local cache directory to avoid re-downloading.

Pipeline
--------
1. Read all chunks from ``chunks_v1_0.jsonl``.
2. For each chunk, encode ``text_for_embedding`` with the BGE model
   (no query instruction for documents; the model is loaded with the
   retrieval query instruction so that ``encode_queries`` can use it).
3. L2-normalize embeddings so that inner-product == cosine similarity.
4. Build an hnswlib index in cosine space.
5. Persist:
     - hnsw_bge.bin                  the index itself
     - hnsw_chunk_ids.json           chunk_id list in index order
     - chunks_meta.parquet           full metadata (including text_for_embedding)
                                    so retrieval can rebuild the prompt.
     - build_index_stats.json        build stats.

Run
----
::

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    python3 build_index.py
    # or with custom paths:
    python3 build_index.py --chunks ../chunks/chunks_v1_0.jsonl \\
                           --out-dir ../index \\
                           --model BAAI/bge-large-en-v1.5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import List

import hnswlib
import numpy as np
import pandas as pd
import torch
from FlagEmbedding import FlagModel


DEFAULT_MODEL = os.environ.get("BGE_MODEL", "BAAI/bge-large-en-v1.5")
QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "


def _load_model(model_name: str, cache_dir: str, use_fp16: bool) -> FlagModel:
    print(f"📦 Loading model: {model_name}")
    print(f"   cache_dir: {cache_dir}")
    print(f"   use_fp16:  {use_fp16}")
    model = FlagModel(
        model_name,
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        use_fp16=use_fp16,
        cache_dir=cache_dir,
    )
    print("✅ Model loaded")
    return model


def _iter_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build(
    chunks_path: str,
    out_dir: str,
    model_name: str,
    cache_dir: str,
    batch_size: int,
    max_length: int,
    use_fp16: bool,
    ef_construction: int,
    M: int,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    chunk_ids: List[str] = []
    texts: List[str] = []
    meta_records: List[dict] = []

    print(f"📥 Reading chunks: {chunks_path}")
    for ch in _iter_chunks(chunks_path):
        chunk_ids.append(ch["chunk_id"])
        texts.append(ch.get("text_for_embedding") or ch.get("text") or "")
        meta_records.append(
            {
                "chunk_id": ch["chunk_id"],
                "chunk_type": ch.get("chunk_type"),
                "task": ch.get("task"),
                "scene_token": ch.get("scene_token"),
                "frame_token": ch.get("frame_token"),
                "image_path": ch.get("image_path"),
                "num_qa": ch.get("num_qa"),
                "num_objects": ch.get("num_objects"),
                "qa_index": ch.get("qa_index"),
                "part_index": ch.get("part_index"),
                "part_total": ch.get("part_total"),
                "text": ch.get("text"),
                "text_for_embedding": ch.get("text_for_embedding"),
            }
        )
    n = len(chunk_ids)
    print(f"✅ Read {n} chunks in {time.time() - t0:.1f}s")

    # 1) Model
    model = _load_model(model_name, cache_dir, use_fp16)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) Encode documents in batches
    print(
        f"🔨 Encoding {n} chunks (batch_size={batch_size}, max_length={max_length}, device={device})")
    t1 = time.time()
    embs_list = []
    for i in range(0, n, batch_size):
        batch = texts[i: i + batch_size]
        # BGE FlagModel.encode returns ndarray for documents
        emb = model.encode(batch, batch_size=batch_size, max_length=max_length)
        embs_list.append(emb)
        if (i // batch_size) % 20 == 0:
            print(
                f"  ... {i + len(batch)}/{n}  elapsed {time.time() - t1:.1f}s")
        # Free GPU cache periodically
        if device == "cuda" and (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()

    embeddings = np.vstack(embs_list).astype("float32")
    # 3) L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    embeddings = embeddings / norms
    print(
        f"✅ Encoded: shape={embeddings.shape}, dtype={embeddings.dtype}  ({time.time() - t1:.1f}s)")

    del embs_list
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    dim = embeddings.shape[1]

    # 4) HNSW index (cosine)
    print(
        f"🔨 Building HNSW (dim={dim}, space=cosine, M={M}, ef_construction={ef_construction})")
    t2 = time.time()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embeddings, num_threads=min(16, os.cpu_count() or 1))
    print(f"✅ HNSW built in {time.time() - t2:.1f}s")

    # 5) Persist
    index_path = os.path.join(out_dir, "hnsw_bge.bin")
    ids_path = os.path.join(out_dir, "hnsw_chunk_ids.json")
    meta_path = os.path.join(out_dir, "chunks_meta.parquet")
    stats_path = os.path.join(out_dir, "build_index_stats.json")

    index.save_index(index_path)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    df = pd.DataFrame(meta_records)
    df.to_parquet(meta_path, index=False)

    stats = {
        "chunks_path": os.path.abspath(chunks_path),
        "out_dir": os.path.abspath(out_dir),
        "model": model_name,
        "query_instruction": QUERY_INSTRUCTION,
        "dim": int(dim),
        "space": "cosine",
        "M": M,
        "ef_construction": ef_construction,
        "n_chunks": int(n),
        "batch_size": batch_size,
        "max_length": max_length,
        "use_fp16": use_fp16,
        "device": device,
        "embeddings_shape": list(embeddings.shape),
        "build_seconds": round(time.time() - t0, 1),
        "encoding_seconds": round(time.time() - t1, 1),
        "hnsw_seconds": round(time.time() - t2, 1),
        "index_path": os.path.abspath(index_path),
        "chunk_ids_path": os.path.abspath(ids_path),
        "meta_path": os.path.abspath(meta_path),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"💾 Saved: {index_path}")
    print(f"💾 Saved: {ids_path}")
    print(f"💾 Saved: {meta_path}")
    print(f"📊 Saved: {stats_path}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--chunks",
        default=os.path.join(os.path.dirname(__file__),
                             "..", "chunks", "chunks_v1_0.jsonl"),
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "index"),
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--cache-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(
            __file__), "..", "models", "bge-large-en-v1.5")),
        help="Where to cache the BGE model weights.",
    )
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Encoding batch size.")
    ap.add_argument("--max-length", type=int, default=256,
                    help="Max tokens per chunk (256 covers p99 of our corpus).")
    ap.add_argument("--no-fp16", action="store_true",
                    help="Disable FP16 (use FP32).")
    ap.add_argument("--ef-construction", type=int,
                    default=200, help="HNSW ef_construction.")
    ap.add_argument("--M", type=int, default=32, help="HNSW M parameter.")
    args = ap.parse_args()

    build(
        chunks_path=os.path.abspath(args.chunks),
        out_dir=os.path.abspath(args.out_dir),
        model_name=args.model,
        cache_dir=os.path.abspath(args.cache_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=not args.no_fp16,
        ef_construction=args.ef_construction,
        M=args.M,
    )


if __name__ == "__main__":
    main()

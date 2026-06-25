"""Build a BGE-M3 HNSW index over entity-community summaries.

For inference, a question is encoded and matched against community summaries;
the top-k most relevant communities' summaries are then injected into the
LLM prompt as global context (Microsoft GraphRAG global search).

Output:
  - <out>/hnsw_bge_m3_entity_communities.bin   HNSW index (cosine, dim=1024)
  - <out>/entity_community_metadata.pkl        per-id metadata
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
    """Preload conda libstdc++ to keep hnswlib / scipy happy in some environments."""
    try:
        import ctypes
        prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
        lib_dir = os.path.join(prefix, "lib")
        candidates = sorted(
            glob.glob(os.path.join(lib_dir, "libstdc++.so.6*")),
            reverse=True,
        )
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
        "--summaries",
        default="./graphrag_community/entity_communities/entity_community_summaries.jsonl",
    )
    p.add_argument(
        "--out-dir",
        default="./graphrag_community/entity_communities",
    )
    p.add_argument(
        "--embed-model",
        default="./models/bge-m3",
    )
    p.add_argument("--embed-device", default="cuda")
    p.add_argument("--ef", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] summaries from {args.summaries}")
    summaries = []
    with open(args.summaries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                summaries.append(json.loads(line))
    summaries.sort(key=lambda r: r["community_id"])
    print(f"[load] {len(summaries)} community summaries")

    print(f"[load] embedder: {args.embed_model}")
    from FlagEmbedding import BGEM3FlagModel

    emb = BGEM3FlagModel(args.embed_model, use_fp16=True, device=args.embed_device)

    texts = [s["summary"] for s in summaries]
    print(f"[embed] encoding {len(texts)} summaries…")
    enc = emb.encode(texts)
    dense_vecs = enc["dense_vecs"] if isinstance(enc, dict) else enc
    dense_vecs = np.asarray(dense_vecs, dtype=np.float32)

    # L2-normalize for cosine
    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True) + 1e-12
    dense_vecs = dense_vecs / norms

    print(f"[embed] shape: {dense_vecs.shape}")

    print("[hnsw] building index …")
    import hnswlib

    dim = dense_vecs.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(
        max_elements=len(summaries),
        ef_construction=200,
        M=32,
    )
    idx.add_items(dense_vecs, list(range(len(summaries))))
    idx.set_ef(args.ef)

    out_idx = os.path.join(args.out_dir, "hnsw_bge_m3_entity_communities.bin")
    idx.save_index(out_idx)
    print(f"[save] wrote {out_idx}")

    meta = pd.DataFrame(
        [
            {
                "row_id": i,
                "community_id": s["community_id"],
                "n_entities": s["n_entities"],
                "n_chunks": s["n_chunks"],
                "summary": s["summary"],
                "members": s.get("members", []),
                "stats": s.get("stats", {}),
            }
            for i, s in enumerate(summaries)
        ]
    )
    out_meta = os.path.join(args.out_dir, "entity_community_metadata.pkl")
    meta.to_pickle(out_meta)
    print(f"[save] wrote {out_meta}")


if __name__ == "__main__":
    main()

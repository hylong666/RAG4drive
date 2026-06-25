"""Build BGE-M3 embeddings + HNSW index over entity descriptions.

This implements the entity-level dense retrieval layer that original Microsoft
GraphRAG uses in its local-search. Each entity gets one embedding derived
from its natural-language description; queries that are not keyword-matched
to a category can still reach relevant entities via embedding similarity.

Output:
  - <out-dir>/entity_hnsw.bin   HNSW cosine index over entity descriptions
  - <out-dir>/entity_metadata.pkl   {row_id: entity_key, embeddings stats}
"""
from __future__ import annotations

import argparse
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


import glob
_preload_conda_libstdcxx()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--entity-descriptions",
        default="./graphrag_community/entity_kg/entity_descriptions.json",
    )
    p.add_argument(
        "--out-dir",
        default="./graphrag_community/entity_kg",
    )
    p.add_argument(
        "--embed-model", default="./models/bge-m3",
    )
    p.add_argument("--embed-device", default="cuda")
    p.add_argument("--ef", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] entity descriptions from {args.entity_descriptions}")
    with open(args.entity_descriptions, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    entity_keys = sorted(descriptions.keys())
    print(f"[load] {len(entity_keys)} entities")

    print(f"[load] embedder: {args.embed_model}")
    from FlagEmbedding import BGEM3FlagModel
    emb = BGEM3FlagModel(args.embed_model, use_fp16=True, device=args.embed_device)

    # Encode each entity's description
    texts = []
    for k in entity_keys:
        d = descriptions[k]
        # Compose a richer embedding text: "Entity: <key>. <description>."
        # This way BGE-M3 sees both the symbolic key AND the natural language.
        texts.append(f"{k}: {d}")

    print(f"[embed] encoding {len(texts)} entities…")
    enc = emb.encode(texts, batch_size=32, max_length=128)
    dense = np.asarray(
        enc["dense_vecs"] if isinstance(enc, dict) else enc,
        dtype=np.float32,
    )
    dense = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
    print(f"[embed] shape: {dense.shape}")

    print("[hnsw] building index…")
    import hnswlib
    dim = dense.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=len(dense), ef_construction=200, M=32)
    idx.add_items(dense, list(range(len(dense))))
    idx.set_ef(args.ef)

    out_idx = os.path.join(args.out_dir, "entity_hnsw.bin")
    idx.save_index(out_idx)
    print(f"[save] wrote {out_idx}")

    metadata = pd.DataFrame({
        "row_id": list(range(len(entity_keys))),
        "entity_key": entity_keys,
        "description": [descriptions[k] for k in entity_keys],
    })
    out_meta = os.path.join(args.out_dir, "entity_metadata.pkl")
    metadata.to_pickle(out_meta)
    print(f"[save] wrote {out_meta}")


if __name__ == "__main__":
    main()
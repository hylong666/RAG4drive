"""Build a BM25 index over per-QA chunks for keyword/exact-match retrieval.

Aligns with RAGdrive's BM25 build:
  - tokenize with regex [A-Za-z0-9_<>]+   (preserves object IDs like <c1,...>)
  - BM25Okapi(k1=1.5, b=0.75)

Output:
  - <out-dir>/bm25_chunks.pkl  {"bm25": <rank_bm25.BM25Okapi>, "chunk_ids": [...]}
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys

from tqdm import tqdm


_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chunks-meta",
        default="./graphrag_community/chunks/chunks_metadata.pkl",
        help="chunks_metadata.pkl produced by build_chunks_index.py.",
    )
    p.add_argument(
        "--out-dir",
        default="./graphrag_community/chunks",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] chunks from {args.chunks_meta}")
    import pandas as pd
    chunks_meta = pd.read_pickle(args.chunks_meta)
    print(f"[load] {len(chunks_meta)} chunks")

    print("[tokenize] using regex [A-Za-z0-9_<>]+")
    texts = [str(t) for t in chunks_meta["text"].tolist()]
    tokenized = [_tokenize(t) for t in tqdm(texts, desc="[tokenize]")]

    print("[bm25] building BM25Okapi(k1=1.5, b=0.75)")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)

    out_path = os.path.join(args.out_dir, "bm25_chunks.pkl")
    chunk_ids = chunks_meta["chunk_id"].tolist()
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f, pickle.HIGHEST_PROTOCOL)
    print(f"[save] wrote {out_path}")
    print(f"[stats] corpus size: {len(chunk_ids)}, avg doc length: "
          f"{sum(len(t) for t in tokenized) / len(tokenized):.1f} tokens")


if __name__ == "__main__":
    main()

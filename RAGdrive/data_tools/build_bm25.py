"""Build a BM25 (sparse) index over the chunked DriveLM corpus.

Inputs
------
- ``RAGdrive/chunks/chunks_v1_0.jsonl`` (default). Each line is a chunk; we
  index on ``text_for_embedding`` (Q+A on top, scene context at the bottom).

Outputs (default)
-----------------
- ``RAGdrive/index/bm25.pkl``        pickled BM25Okapi object.
- ``RAGdrive/index/bm25_chunk_ids.json``  list of ``chunk_id`` in index order
                                         (so we can look up the chunk that
                                         corresponds to a BM25 hit).
- ``RAGdrive/index/bm25_stats.json`` build stats (corpus size, vocab, etc.).

Run
----
::

    python3 build_bm25.py
    # or:
    python3 build_bm25.py --chunks ../chunks/chunks_v1_0.jsonl --out-dir ../index
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from typing import Iterable, List

from rank_bm25 import BM25Okapi


# A small but reasonable English tokenizer: lowercase, split on non-alphanum,
# drop tokens of length < 2 and pure digits. The goal is to keep content words
# (c1, cam_front, construction, vehicle) while dropping stop-noise.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


def iter_chunks(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build(chunks_path: str, out_dir: str, k1: float, b: float) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    chunk_ids: List[str] = []
    corpus: List[List[str]] = []
    n_overview = 0
    n_pair = 0
    total_tokens = 0

    print(f"📥 Loading + tokenizing: {chunks_path}")
    n = 0
    for ch in iter_chunks(chunks_path):
        text = ch.get("text_for_embedding") or ch.get("text") or ""
        toks = tokenize(text)
        chunk_ids.append(ch["chunk_id"])
        corpus.append(toks)
        total_tokens += len(toks)
        if ch.get("chunk_type") == "frame_overview":
            n_overview += 1
        elif ch.get("chunk_type") == "frame_qa_pair":
            n_pair += 1
        n += 1
        if n % 50000 == 0:
            print(f"  ... tokenized {n} chunks")

    print(f"✅ Tokenized {n} chunks in {time.time() - t0:.1f}s "
          f"({total_tokens:,} tokens, avg {total_tokens/max(n,1):.1f} tok/chunk)")

    t1 = time.time()
    print(f"🔨 Building BM25Okapi (k1={k1}, b={b}) over {n} docs ...")
    bm25 = BM25Okapi(corpus, k1=k1, b=b)
    print(f"✅ BM25 built in {time.time() - t1:.1f}s")

    # Persist
    index_path = os.path.join(out_dir, "bm25.pkl")
    ids_path = os.path.join(out_dir, "bm25_chunk_ids.json")
    stats_path = os.path.join(out_dir, "bm25_stats.json")

    with open(index_path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)

    stats = {
        "chunks_path": os.path.abspath(chunks_path),
        "out_dir": os.path.abspath(out_dir),
        "n_chunks": n,
        "n_overview": n_overview,
        "n_pair": n_pair,
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": round(total_tokens / max(n, 1), 2),
        "k1": k1,
        "b": b,
        "build_seconds": round(time.time() - t0, 1),
        "tokenizer": "lower + alphanum/underscore/angle-bracket, len>=2",
        "index_path": os.path.abspath(index_path),
        "chunk_ids_path": os.path.abspath(ids_path),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"💾 Saved: {index_path}")
    print(f"💾 Saved: {ids_path}")
    print(f"📊 Saved: {stats_path}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--chunks",
        default=os.path.join(os.path.dirname(__file__), "..", "chunks", "chunks_v1_0.jsonl"),
        help="Input chunks JSONL.",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "index"),
        help="Output directory for bm25.pkl, bm25_chunk_ids.json, bm25_stats.json.",
    )
    ap.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter (saturation).")
    ap.add_argument("--b", type=float, default=0.75, help="BM25 b parameter (length normalization).")
    args = ap.parse_args()
    build(os.path.abspath(args.chunks), os.path.abspath(args.out_dir), args.k1, args.b)


if __name__ == "__main__":
    main()

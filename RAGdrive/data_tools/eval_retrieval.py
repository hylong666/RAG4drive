"""Sanity evaluation for the RAGdrive retrieval pipeline.

Two modes:

* ``--mode self`` (default) — self-retrieval on train pair chunks.
  For N sampled train pair chunks, use the question text as the query and
  check whether each retriever can rank the **source chunk** back into top-K.
  This is a clean, well-defined metric (duplicate / near-duplicate
  detection). Reports Recall@K and MRR for BM25 / HNSW / RRF / RRF+Rerank.

* ``--mode val`` — task alignment on val questions.
  For each val Q (the v1_1_val_nus_q_only set has questions with no answers),
  run each retriever and measure task alignment: the rank of the first
  same-task chunk in the top-K. This is a qualitative sanity check, since
  we have no ground-truth answer in the val JSON.

Both modes are quick (a few minutes on a 3090 for N=1000).

Run
---
::

    # Self-retrieval on 1000 train pair chunks
    python3 eval_retrieval.py --mode self --n 1000 --top-k 20

    # Val task alignment
    python3 eval_retrieval.py --mode val --n 500 --top-k 10

    # With cross-encoder reranking
    python3 eval_retrieval.py --mode self --n 1000 --rerank --pre-k 50 --top-k 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from typing import List, Optional

import hnswlib
import numpy as np
import pandas as pd

try:
    from FlagEmbedding import FlagModel
    _HAS_FLAG = True
except ImportError:
    _HAS_FLAG = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    _HAS_RERANK = True
except ImportError:
    _HAS_RERANK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-large-en-v1.5")
DEFAULT_RERANKER_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-reranker-large")
DEFAULT_TRAIN_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "json", "v1_0_train_nus.json")
DEFAULT_VAL_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "json", "v1_1_val_nus_q_only.json")
QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "
TASKS = ("perception", "prediction", "planning", "behavior")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>]+")


def _tok(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s) if len(t) >= 2]


# ---------------------------------------------------------------------------
# Q extraction
# ---------------------------------------------------------------------------
def _extract_q(text_for_embedding: str) -> str:
    """Pull the question out of ``text_for_embedding`` (which starts with Q:)."""
    m = re.search(r"^Q:\s*(.+)$", text_for_embedding, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_header(text_for_embedding: str) -> str:
    """Pull the 'Context: ...' line(s) from text_for_embedding."""
    m = re.search(r"^Context:\s*(.+?)(?:\n|$)", text_for_embedding, re.MULTILINE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_bm25(idx_dir: str):
    with open(os.path.join(idx_dir, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(idx_dir, "bm25_chunk_ids.json"), "r", encoding="utf-8") as f:
        ids = json.load(f)
    return bm25, ids


def _load_hnsw(idx_dir: str, ef: int = 64):
    with open(os.path.join(idx_dir, "hnsw_chunk_ids.json"), "r", encoding="utf-8") as f:
        ids = json.load(f)
    h = hnswlib.Index(space="cosine", dim=1024)
    h.load_index(os.path.join(idx_dir, "hnsw_bge.bin"))
    h.set_ef(ef)
    return h, ids


def _load_meta(idx_dir: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(idx_dir, "chunks_meta.parquet")).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-retriever functions
# ---------------------------------------------------------------------------
def bm25_topk(bm25, query_text: str, k: int) -> List[int]:
    scores = bm25.get_scores(_tok(query_text))
    n = min(k, len(scores))
    idxs = np.argpartition(-scores, n - 1)[:n]
    idxs = idxs[np.argsort(-scores[idxs])]
    return [int(i) for i in idxs.tolist()]


def hnsw_topk(hnsw, model, query_text: str, k: int) -> List[int]:
    vec = model.encode_queries([query_text]).astype("float32")
    labels, _ = hnsw.knn_query(vec, k=k)
    return [int(i) for i in labels[0].tolist()]


def rrf_fuse(rank_lists: List[List[int]], k_rrf: int = 60) -> List[int]:
    scores: dict = defaultdict(float)
    for lst in rank_lists:
        for rank, i in enumerate(lst):
            scores[int(i)] += 1.0 / (k_rrf + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------------------
# Self-retrieval eval
# ---------------------------------------------------------------------------
def eval_self_retrieval(
    meta: pd.DataFrame,
    bm25, bm25_ids,
    hnsw, hnsw_ids, model,
    reranker=None,
    n: int = 1000,
    top_k: int = 20,
    pre_k: int = 50,
    seed: int = 42,
    use_rerank: bool = False,
    k_rrf: int = 60,
) -> dict:
    """For each sampled train pair chunk, query with its Q and check
    whether each retriever ranks the source chunk in top-K.

    Returns a dict of metrics per retriever.
    """
    pair_mask = meta["chunk_type"] == "frame_qa_pair"
    pair_idxs = np.where(pair_mask.values)[0]
    rng = random.Random(seed)
    rng.shuffle(pair_idxs)
    pair_idxs = pair_idxs[:n]

    # Storage: per retriever, per K, list of (found, rank) or -1 if not found
    retrievers = ["bm25", "hnsw", "rrf", "rrf_rerank"] if use_rerank else ["bm25", "hnsw", "rrf"]
    metrics = {r: [] for r in retrievers}  # list of source rank (or -1) per query

    t0 = time.time()
    for i, src in enumerate(pair_idxs):
        row = meta.iloc[int(src)]
        tfe = str(row.get("text_for_embedding") or "")
        q = _extract_q(tfe)
        scene = _extract_header(tfe)
        if not q:
            continue
        qtext = f"{q}\n\nContext: {scene}" if scene else q

        bm25_ranks = bm25_topk(bm25, qtext, k=pre_k if use_rerank else top_k)
        hnsw_ranks = hnsw_topk(hnsw, model, qtext, k=pre_k if use_rerank else top_k)
        rrf_ranks = rrf_fuse([bm25_ranks, hnsw_ranks], k_rrf=k_rrf)

        if use_rerank and reranker is not None:
            # Build candidate dicts, rerank, get final top-K
            candidates = []
            for pos, idx in enumerate(rrf_ranks[:pre_k]):
                r = meta.iloc[idx]
                candidates.append({
                    "label": int(idx),
                    "text_for_embedding": r.get("text_for_embedding"),
                })
            # Apply rerank (raw score; just use sort order)
            pairs = [[qtext, c["text_for_embedding"] or ""] for c in candidates]
            with torch.inference_mode():
                enc = reranker["tokenizer"](
                    pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(reranker["device"])
                logits = reranker["model"](**enc).logits.squeeze(-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                scores = [float(x) for x in logits.float().cpu().numpy()]
            order = sorted(range(len(candidates)), key=lambda j: -scores[j])
            rerank_ranks = [candidates[j]["label"] for j in order]
        else:
            rerank_ranks = []

        # Locate source rank in each list
        for name, ranks in [
            ("bm25", bm25_ranks),
            ("hnsw", hnsw_ranks),
            ("rrf", rrf_ranks),
            ("rrf_rerank", rerank_ranks),
        ]:
            if name not in metrics:
                continue
            try:
                rank = ranks.index(int(src)) + 1  # 1-indexed
            except ValueError:
                rank = -1
            metrics[name].append(rank)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(pair_idxs)}  ({time.time() - t0:.1f}s)")

    # Aggregate
    out: dict = {}
    for name, ranks in metrics.items():
        n_done = len(ranks)
        ranks_arr = np.array(ranks)
        valid = ranks_arr > 0
        out[name] = {
            "n": int(n_done),
            "found_count": int(valid.sum()),
            "found_pct": round(float(valid.mean()) * 100, 2) if n_done else 0.0,
            "mrr": round(float(np.where(valid, 1.0 / ranks_arr, 0.0).mean()), 4) if n_done else 0.0,
            "recall": {},
        }
        for K in (1, 5, 10, 20, 50):
            if K > top_k and name in ("bm25", "hnsw", "rrf"):
                continue
            if K > pre_k and name == "rrf_rerank":
                continue
            hit = int(((ranks_arr > 0) & (ranks_arr <= K)).sum())
            out[name]["recall"][f"@{K}"] = round(hit / max(n_done, 1) * 100, 2)
    return out


# ---------------------------------------------------------------------------
# Val task-alignment eval
# ---------------------------------------------------------------------------
def eval_val_task_alignment(
    val_json: str,
    meta: pd.DataFrame,
    bm25, bm25_ids,
    hnsw, hnsw_ids, model,
    reranker=None,
    n: int = 500,
    top_k: int = 10,
    pre_k: int = 50,
    seed: int = 42,
    use_rerank: bool = False,
    k_rrf: int = 60,
) -> dict:
    """For each val Q, query and check the rank of the first same-task chunk."""
    with open(val_json, "r", encoding="utf-8") as f:
        val = json.load(f)
    queries = []
    for sc in val.values():
        if not isinstance(sc, dict):
            continue
        scene_desc = sc.get("scene_description", "") or ""
        for ft, fb in (sc.get("key_frames") or {}).items():
            qa = fb.get("QA") or {}
            image_paths = fb.get("image_paths") or {}
            image_path = ""
            if isinstance(image_paths, dict):
                image_path = image_paths.get("CAM_FRONT") or ""
            for cat, qas in qa.items():
                if not isinstance(qas, list):
                    continue
                for qa_item in qas:
                    if not isinstance(qa_item, dict):
                        continue
                    q = qa_item.get("Q") or ""
                    if not q:
                        continue
                    queries.append({
                        "task": cat,
                        "question": q,
                        "scene": scene_desc,
                        "image_path": image_path,
                    })
    rng = random.Random(seed)
    rng.shuffle(queries)
    queries = queries[:n]

    retrievers = ["bm25", "hnsw", "rrf", "rrf_rerank"] if use_rerank else ["bm25", "hnsw", "rrf"]
    metrics = {r: {"task_hits_at_K": Counter(), "first_task_rank": [], "n": 0}
               for r in retrievers}

    t0 = time.time()
    for i, q in enumerate(queries):
        qtext = f"{q['question']}\n\nContext: {q['scene']}" if q["scene"] else q["question"]

        bm25_ranks = bm25_topk(bm25, qtext, k=pre_k if use_rerank else top_k)
        hnsw_ranks = hnsw_topk(hnsw, model, qtext, k=pre_k if use_rerank else top_k)
        rrf_ranks = rrf_fuse([bm25_ranks, hnsw_ranks], k_rrf=k_rrf)

        if use_rerank and reranker is not None:
            candidates = []
            for idx in rrf_ranks[:pre_k]:
                r = meta.iloc[idx]
                candidates.append({
                    "label": int(idx),
                    "task": r.get("task"),
                    "text_for_embedding": r.get("text_for_embedding"),
                })
            pairs = [[qtext, c["text_for_embedding"] or ""] for c in candidates]
            with torch.inference_mode():
                enc = reranker["tokenizer"](
                    pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(reranker["device"])
                logits = reranker["model"](**enc).logits.squeeze(-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                scores = [float(x) for x in logits.float().cpu().numpy()]
            order = sorted(range(len(candidates)), key=lambda j: -scores[j])
            rerank_ranks = [candidates[j] for j in order]
        else:
            rerank_ranks = []

        for name in retrievers:
            metrics[name]["n"] += 1
            if name == "rrf_rerank":
                # rerank_ranks is a list of dicts with 'task' field
                rrf_ranks_with_task = rerank_ranks
            else:
                rrf_ranks_with_task = [
                    {"label": int(idx), "task": meta.iloc[idx].get("task")}
                    for idx in (bm25_ranks if name == "bm25" else
                                hnsw_ranks if name == "hnsw" else
                                rrf_ranks)
                ]

            # Count task-aligned hits in top-K
            topk_tasks = [r["task"] for r in rrf_ranks_with_task[:top_k]]
            hits = sum(1 for t in topk_tasks if t == q["task"])
            metrics[name]["task_hits_at_K"][hits] += 1

            # Rank of first same-task chunk
            first_rank = -1
            for rk, r in enumerate(rrf_ranks_with_task, start=1):
                if r["task"] == q["task"]:
                    first_rank = rk
                    break
            metrics[name]["first_task_rank"].append(first_rank)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(queries)}  ({time.time() - t0:.1f}s)")

    # Aggregate
    out: dict = {}
    for name, m in metrics.items():
        n_done = m["n"]
        first_ranks = np.array(m["first_task_rank"])
        valid = first_ranks > 0
        out[name] = {
            "n": int(n_done),
            "task_match_at_1_pct": round(float((first_ranks == 1).mean()) * 100, 2) if n_done else 0.0,
            "task_match_at_5_pct": round(float(((first_ranks > 0) & (first_ranks <= 5)).mean()) * 100, 2) if n_done else 0.0,
            "task_match_at_10_pct": round(float(((first_ranks > 0) & (first_ranks <= 10)).mean()) * 100, 2) if n_done else 0.0,
            "mrr_task": round(float(np.where(valid, 1.0 / first_ranks, 0.0).mean()), 4) if n_done else 0.0,
            "task_in_top5_distribution": dict(sorted(m["task_hits_at_K"].items())),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", choices=("self", "val"), default="self")
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--reranker-dir", default=DEFAULT_RERANKER_DIR)
    ap.add_argument("--train-json", default=DEFAULT_TRAIN_JSON)
    ap.add_argument("--val-json", default=DEFAULT_VAL_JSON)
    ap.add_argument("--n", type=int, default=1000, help="Number of samples to evaluate.")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--pre-k", type=int, default=50, help="Candidates fed to reranker.")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    idx_dir = os.path.abspath(args.index_dir)
    print(f"📂 Loading indices from {idx_dir}")
    meta = _load_meta(idx_dir)
    print(f"   meta: {len(meta):,} chunks")

    bm25, bm25_ids = _load_bm25(idx_dir)
    print(f"   bm25: {len(bm25_ids):,} docs")

    model = None
    hnsw = hnsw_ids = None
    if not args.no_dense:
        hnsw, hnsw_ids = _load_hnsw(idx_dir)
        print(f"   hnsw: {len(hnsw_ids):,} vectors")
        print("📦 Loading BGE bi-encoder…")
        model = FlagModel(
            "BAAI/bge-large-en-v1.5",
            query_instruction_for_retrieval=QUERY_INSTRUCTION,
            use_fp16=True,
            cache_dir=args.model_dir,
        )

    reranker = None
    if args.rerank:
        print("📦 Loading reranker…")
        reranker = {
            "tokenizer": AutoTokenizer.from_pretrained("BAAI/bge-reranker-large", cache_dir=args.reranker_dir),
            "model": AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-large",
                cache_dir=args.reranker_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        reranker["model"].eval().to(reranker["device"])

    if args.mode == "self":
        print(f"\n=== Self-retrieval eval | n={args.n} | top_k={args.top_k} | pre_k={args.pre_k} | rerank={args.rerank} ===")
        res = eval_self_retrieval(
            meta=meta, bm25=bm25, bm25_ids=bm25_ids, hnsw=hnsw, hnsw_ids=hnsw_ids,
            model=model, reranker=reranker, n=args.n, top_k=args.top_k, pre_k=args.pre_k,
            seed=args.seed, use_rerank=args.rerank,
        )
        print("\n📊 Results:")
        for name, m in res.items():
            print(f"\n  [{name}] n={m['n']} found={m['found_count']} ({m['found_pct']}%)  MRR={m['mrr']}")
            for k_str, v in m["recall"].items():
                print(f"      Recall{k_str}: {v}%")
    else:
        print(f"\n=== Val task-alignment eval | n={args.n} | top_k={args.top_k} | pre_k={args.pre_k} | rerank={args.rerank} ===")
        res = eval_val_task_alignment(
            val_json=args.val_json, meta=meta, bm25=bm25, bm25_ids=bm25_ids,
            hnsw=hnsw, hnsw_ids=hnsw_ids, model=model, reranker=reranker,
            n=args.n, top_k=args.top_k, pre_k=args.pre_k, seed=args.seed, use_rerank=args.rerank,
        )
        print("\n📊 Results (rank of first same-task chunk):")
        for name, m in res.items():
            print(f"\n  [{name}] n={m['n']}")
            print(f"      task_in_top1: {m['task_match_at_1_pct']}%")
            print(f"      task_in_top5: {m['task_match_at_5_pct']}%")
            print(f"      task_in_top10: {m['task_match_at_10_pct']}%")
            print(f"      MRR_task: {m['mrr_task']}")
            print(f"      top-5 task-count dist: {m['task_in_top5_distribution']}")


if __name__ == "__main__":
    main()

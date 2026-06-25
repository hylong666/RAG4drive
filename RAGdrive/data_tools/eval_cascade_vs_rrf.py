"""Compare cascade vs RRF on val task-alignment.

Runs the same val queries through both retrievers (each with cross-encoder
reranking enabled) and reports:

* task@1, task@5, task@10: % of queries where the first same-task chunk is
  in top-K
* MRR_task: mean reciprocal rank of the first same-task chunk
* top-5 purity: average # of same-task chunks in top-5 (out of 5)
* per-query latency (ms)
* top-10 overlap (Jaccard) between the two retrievers — measures whether
  they return meaningfully different chunks

Run
---
::

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" HF_ENDPOINT=https://hf-mirror.com
    python3 eval_cascade_vs_rrf.py --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from typing import List, Sequence

import numpy as np

# Make data_tools importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RRF_retrieve import RAGRetriever, RetrieveConfig
from cascade_retrieve import CascadeRetriever, CascadeConfig


DEFAULT_VAL_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "json", "v1_1_val_nus_q_only.json")


def _collect_queries(val_json: str) -> List[dict]:
    with open(val_json, "r", encoding="utf-8") as f:
        val = json.load(f)
    queries: List[dict] = []
    for sc in val.values():
        if not isinstance(sc, dict):
            continue
        scene_desc = sc.get("scene_description", "") or ""
        for ft, fb in (sc.get("key_frames") or {}).items():
            qa = fb.get("QA") or {}
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
                    })
    return queries


def _metrics_for_results(results: Sequence[dict], target_task: str, top_k_eval: int = 10) -> dict:
    """For one retriever's results, compute task-alignment metrics."""
    topk_tasks = [r.get("task") for r in results[:top_k_eval]]
    # rank of first same-task chunk
    first_rank = -1
    for i, t in enumerate(topk_tasks, start=1):
        if t == target_task:
            first_rank = i
            break
    purity5 = sum(1 for t in topk_tasks[:5] if t == target_task)
    return {
        "task@1": 1 if first_rank == 1 else 0,
        "task@5": 1 if 0 < first_rank <= 5 else 0,
        "task@10": 1 if 0 < first_rank <= 10 else 0,
        "mrr": 1.0 / first_rank if first_rank > 0 else 0.0,
        "purity5": purity5,
        "top10_chunk_ids": [r.get("chunk_id") for r in results[:10]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--val-json", default=DEFAULT_VAL_JSON)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--rrf-pre-k", type=int, default=50,
        help="RRF: how many candidates the reranker scores.",
    )
    ap.add_argument(
        "--cascade-hnsw-k", type=int, default=100,
        help="Cascade: HNSW first-stage fan-out.",
    )
    ap.add_argument(
        "--cascade-bm25-k", type=int, default=20,
        help="Cascade: BM25 within-HNSW output count, fed to reranker.",
    )
    args = ap.parse_args()

    print(f"📥 Loading val queries from {args.val_json}")
    queries = _collect_queries(args.val_json)
    print(f"   total: {len(queries):,}")
    rng = random.Random(args.seed)
    rng.shuffle(queries)
    queries = queries[: args.n]
    print(f"   sampled: {len(queries)}")

    print("\n📦 Loading RRF retriever (BM25 + HNSW + RRF + Rerank)…")
    rrf_cfg = RetrieveConfig(
        top_k=args.top_k,
        use_rerank=True,
        pre_k=args.rrf_pre_k,
        k_bm25=50, k_dense=50, k_rrf=60,
        task_filter=None, max_per_frame=0,
    )
    rrf = RAGRetriever(rrf_cfg)

    print("\n📦 Loading Cascade retriever (HNSW@%d → BM25 top-%d → Rerank)…" %
          (args.cascade_hnsw_k, args.cascade_bm25_k))
    cascade_cfg = CascadeConfig(
        top_k=args.top_k,
        hnsw_k=args.cascade_hnsw_k,
        bm25_k=args.cascade_bm25_k,
        use_rerank=True,
        ef_search=128,
    )
    cascade = CascadeRetriever(cascade_cfg)

    # Per-query results
    per_q: List[dict] = []
    t0 = time.time()
    for i, q in enumerate(queries, start=1):
        rec = {"task": q["task"], "question": q["question"]}

        # RRF
        s = time.time()
        rrf_res = rrf.retrieve(question=q["question"], scene_context=q["scene"], top_k=args.top_k)
        rec["rrf_latency_ms"] = (time.time() - s) * 1000
        rec["rrf"] = _metrics_for_results(rrf_res, q["task"], top_k_eval=args.top_k)

        # Cascade
        s = time.time()
        cas_res = cascade.retrieve(question=q["question"], scene_context=q["scene"], top_k=args.top_k)
        rec["cascade_latency_ms"] = (time.time() - s) * 1000
        rec["cascade"] = _metrics_for_results(cas_res, q["task"], top_k_eval=args.top_k)

        # Overlap
        a = set(rec["rrf"]["top10_chunk_ids"])
        b = set(rec["cascade"]["top10_chunk_ids"])
        rec["top10_jaccard"] = round(len(a & b) / max(len(a | b), 1), 3)
        rec["top10_intersection"] = len(a & b)

        per_q.append(rec)
        if i % 25 == 0:
            print(f"  ... {i}/{len(queries)}  ({time.time() - t0:.1f}s)")

    # Aggregate
    def avg(vals):
        return round(float(np.mean(vals)) * 100, 2) if vals else 0.0

    def avg_dec(vals, dec=4):
        return round(float(np.mean(vals)), dec) if vals else 0.0

    print("\n" + "=" * 78)
    print("📊 Cascade vs RRF on val task-alignment (n={n}, top_k={k})".format(
        n=len(per_q), k=args.top_k))
    print("=" * 78)
    headers = ["metric", "RRF", "Cascade", "Δ"]
    print(f"  {headers[0]:<22}  {headers[1]:>10}  {headers[2]:>10}  {headers[3]:>10}")
    print("-" * 78)

    rows = [
        ("task@1 (%)",  avg([r["rrf"]["task@1"] for r in per_q]),
                       avg([r["cascade"]["task@1"] for r in per_q])),
        ("task@5 (%)",  avg([r["rrf"]["task@5"] for r in per_q]),
                       avg([r["cascade"]["task@5"] for r in per_q])),
        ("task@10 (%)", avg([r["rrf"]["task@10"] for r in per_q]),
                       avg([r["cascade"]["task@10"] for r in per_q])),
        ("MRR_task",    avg_dec([r["rrf"]["mrr"] for r in per_q]),
                       avg_dec([r["cascade"]["mrr"] for r in per_q])),
        ("top-5 purity (mean #/5)", avg_dec([r["rrf"]["purity5"] for r in per_q], 2),
                                  avg_dec([r["cascade"]["purity5"] for r in per_q], 2)),
        ("latency (ms)", round(float(np.mean([r["rrf_latency_ms"] for r in per_q])), 1),
                       round(float(np.mean([r["cascade_latency_ms"] for r in per_q])), 1)),
    ]
    for name, rrf_v, cas_v in rows:
        if isinstance(rrf_v, (int, float)) and isinstance(cas_v, (int, float)):
            if name == "latency (ms)":
                delta = f"{cas_v - rrf_v:+.1f}ms"
            else:
                delta = f"{cas_v - rrf_v:+.2f}"
        else:
            delta = "—"
        print(f"  {name:<22}  {rrf_v:>10}  {cas_v:>10}  {delta:>10}")

    print("\n📐 Top-10 overlap:")
    jaccards = [r["top10_jaccard"] for r in per_q]
    intersections = [r["top10_intersection"] for r in per_q]
    print(f"  mean Jaccard:     {round(np.mean(jaccards), 3)}")
    print(f"  mean intersection (out of 10): {round(np.mean(intersections), 2)}")
    print(f"  median Jaccard:   {round(np.median(jaccards), 3)}")

    # Per-task breakdown
    by_task = {"rrf": {}, "cascade": {}}
    for r in per_q:
        t = r["task"]
        for k in ("rrf", "cascade"):
            d = by_task[k].setdefault(t, {"task@1": [], "mrr": []})
            d["task@1"].append(r[k]["task@1"])
            d["mrr"].append(r[k]["mrr"])
    print("\n📊 Per-task task@1 (%):")
    print(f"  {'task':<12}  {'RRF':>8}  {'Cascade':>8}  {'n':>4}")
    for t in sorted(by_task["rrf"].keys()):
        n = len(by_task["rrf"][t]["task@1"])
        rrf_v = round(np.mean(by_task["rrf"][t]["task@1"]) * 100, 1)
        cas_v = round(np.mean(by_task["cascade"][t]["task@1"]) * 100, 1)
        print(f"  {t:<12}  {rrf_v:>8}  {cas_v:>8}  {n:>4}")


if __name__ == "__main__":
    main()

"""RRF-fusion retrieval over the RAGdrive corpus (BM25 + HNSW with RRF +
optional BGE cross-encoder reranking).

This is the "parallel + fuse" architecture:

    BM25 top-K1  ─┐
                  ├──> Reciprocal Rank Fusion ──> optional Rerank ──> top-K
    HNSW top-K1  ─┘

Use this when you want maximum recall and are OK paying the BM25 full-corpus
cost. For a faster alternative that does the same task in production, see
``cascade_retrieve.py``.

CLI examples
------------
::

    # Ad-hoc query
    python3 RRF_retrieve.py -q "What is the status of the construction vehicle to the front right?"

    # With scene context
    python3 RRF_retrieve.py -q "What should the ego do?" \\
                            -s "The ego vehicle proceeds through the intersection."

    # With cross-encoder reranking
    python3 RRF_retrieve.py -q "is the truck moving?" --rerank --pre-k 50 --top-k 10

    # Batch retrieval
    python3 RRF_retrieve.py --batch queries.jsonl --output retrievals.jsonl --rerank --top-k 20
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import hnswlib
import numpy as np
import pandas as pd

try:
    from FlagEmbedding import FlagModel
    _HAS_FLAG = True
except ImportError:
    _HAS_FLAG = False


# ---------------------------------------------------------------------------
# Tokenizer (mirrors build_bm25.py)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-large-en-v1.5")
DEFAULT_RERANKER_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-reranker-large")
QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "
TASKS = ("perception", "prediction", "planning", "behavior")


@dataclass
class RetrieveConfig:
    index_dir: str = DEFAULT_INDEX_DIR
    model_dir: str = DEFAULT_MODEL_DIR
    reranker_dir: str = DEFAULT_RERANKER_DIR
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = False
    k_bm25: int = 50
    k_dense: int = 50
    k_rrf: int = 60
    pre_k: int = 50                   # how many candidates feed the reranker
    top_k: int = 10
    max_per_frame: int = 0            # 0 = no dedup
    task_filter: Optional[List[str]] = None
    ef_search: int = 64               # HNSW ef at query time
    reranker_doc_field: str = "text_for_embedding"  # which text field to rerank on


# ---------------------------------------------------------------------------
# Reranker wrapper (BGE cross-encoder via raw transformers)
# ---------------------------------------------------------------------------
# We use raw transformers + torch instead of FlagEmbedding's FlagReranker because
# the latter pins to an old transformers API (``prepare_for_model``) that no
# longer exists in the installed version. The model is just a
# XLMRobertaForSequenceClassification that outputs a single logit per (q, d)
# pair — straightforward to run directly.


class BGEReranker:
    """BGE cross-encoder reranker using raw transformers (FlagReranker bypass).

    Lazy-loads on first use. ``compute_score`` takes a list of (query, doc)
    pairs and returns a list of logits; higher = more relevant.
    """

    def __init__(self, model_dir: str, use_fp16: bool = True, max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.torch = torch
        self.max_length = max_length
        print(f"📦 Loading reranker (BAAI/bge-reranker-large) — cache_dir: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-large",
            cache_dir=model_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-large",
            cache_dir=model_dir,
            torch_dtype=torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)
        print(f"✅ Reranker loaded (device={self.device})")

    @staticmethod
    def _sigmoid(x: float) -> float:
        # numerical stability
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _score_batch(self, pairs: List[List[str]], batch_size: int = 16) -> List[float]:
        import torch

        all_scores: List[float] = []
        with torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                all_scores.extend([float(x) for x in logits.float().cpu().numpy()])
        return all_scores

    def rerank(self, query: str, candidates: List[dict], top_k: Optional[int] = None,
               doc_field: str = "text_for_embedding", batch_size: int = 16,
               normalize: bool = True) -> List[dict]:
        if not candidates:
            return []
        pairs = [[query, str(c.get(doc_field) or "")] for c in candidates]
        raw_scores = self._score_batch(pairs, batch_size=batch_size)
        # Apply sigmoid so scores are in [0, 1] and easier to interpret
        for c, s in zip(candidates, raw_scores):
            c["rerank_score"] = self._sigmoid(s) if normalize else float(s)
        reranked = sorted(candidates, key=lambda x: -x["rerank_score"])
        for i, r in enumerate(reranked, start=1):
            r["rank"] = i
        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class RAGRetriever:
    """Hybrid (BM25 + HNSW) retriever with RRF fusion and optional cross-encoder rerank."""

    def __init__(self, config: RetrieveConfig):
        self.cfg = config
        self.bm25 = None
        self.bm25_ids: List[str] = []
        self.hnsw = None
        self.hnsw_ids: List[str] = []
        self.meta: Optional[pd.DataFrame] = None
        self.model = None
        self.reranker: Optional[BGEReranker] = None
        self._load()

    # ---------- loading ----------
    def _load(self) -> None:
        idx = self.cfg.index_dir

        # Metadata is needed for both retrieval paths.
        meta_path = os.path.join(idx, "chunks_meta.parquet")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata: {meta_path}")
        self.meta = pd.read_parquet(meta_path).reset_index(drop=True)
        print(f"✅ Metadata loaded: {len(self.meta):,} chunks")

        if self.cfg.use_bm25:
            with open(os.path.join(idx, "bm25.pkl"), "rb") as f:
                self.bm25 = pickle.load(f)
            with open(os.path.join(idx, "bm25_chunk_ids.json"), "r", encoding="utf-8") as f:
                self.bm25_ids = json.load(f)
            print(f"✅ BM25 loaded: {len(self.bm25_ids):,} docs")

        if self.cfg.use_dense:
            if not _HAS_FLAG:
                raise ImportError("FlagEmbedding not installed; required for dense retriever")
            hnsw_path = os.path.join(idx, "hnsw_bge.bin")
            if not os.path.exists(hnsw_path):
                raise FileNotFoundError(f"Missing HNSW index: {hnsw_path}")
            with open(os.path.join(idx, "hnsw_chunk_ids.json"), "r", encoding="utf-8") as f:
                self.hnsw_ids = json.load(f)
            self.hnsw = hnswlib.Index(space="cosine", dim=1024)
            self.hnsw.load_index(hnsw_path)
            self.hnsw.set_ef(self.cfg.ef_search)
            print(f"✅ HNSW loaded: {self.hnsw.get_current_count():,} vectors")

            print("📦 Loading BGE bi-encoder…")
            model_name = "BAAI/bge-large-en-v1.5"
            self.model = FlagModel(
                model_name,
                query_instruction_for_retrieval=QUERY_INSTRUCTION,
                use_fp16=True,
                cache_dir=self.cfg.model_dir,
            )
            print("✅ BGE bi-encoder loaded")

        if self.cfg.use_rerank:
            self.reranker = BGEReranker(self.cfg.reranker_dir, use_fp16=True)

    # ---------- encoding ----------
    def _encode_query(self, question: str, scene_context: str = "") -> str:
        """Compose the final query string sent to BGE.

        We use the same convention as the chunk text_for_embedding: Q first,
        Context second. BGE's query instruction is applied by ``encode_queries``.
        """
        q = (question or "").strip()
        if scene_context:
            q = f"{q}\n\nContext: {scene_context.strip()}"
        return q

    def _encode_dense(self, question: str, scene_context: str = "") -> np.ndarray:
        text = self._encode_query(question, scene_context)
        vec = self.model.encode_queries([text])
        return vec.astype("float32")

    # ---------- single retriever ----------
    def _bm25_search(self, question: str, scene_context: str = "") -> List[Tuple[int, float]]:
        text = self._encode_query(question, scene_context)
        scores = self.bm25.get_scores(tokenize(text))
        # sort descending, take top k_bm25
        idxs = np.argpartition(-scores, min(self.cfg.k_bm25, len(scores) - 1))[: self.cfg.k_bm25]
        idxs = idxs[np.argsort(-scores[idxs])]
        idxs = idxs.tolist()
        top = [(int(i), float(scores[i])) for i in idxs]
        return top

    def _hnsw_search(self, question: str, scene_context: str = "") -> Tuple[np.ndarray, np.ndarray]:
        vec = self._encode_dense(question, scene_context)
        labels, dists = self.hnsw.knn_query(vec, k=self.cfg.k_dense)
        return labels[0], dists[0]

    # ---------- fusion ----------
    def _rrf(self, bm25_ranks: Sequence[int], hnsw_ranks: Sequence[int]) -> dict:
        scores: dict = {}
        for rank, i in enumerate(bm25_ranks):
            scores[int(i)] = scores.get(int(i), 0.0) + 1.0 / (self.cfg.k_rrf + rank + 1)
        if hnsw_ranks is not None:
            for rank, i in enumerate(hnsw_ranks):
                scores[int(i)] = scores.get(int(i), 0.0) + 1.0 / (self.cfg.k_rrf + rank + 1)
        return scores

    # ---------- public API ----------
    def retrieve(
        self,
        question: str,
        scene_context: str = "",
        top_k: Optional[int] = None,
        max_per_frame: Optional[int] = None,
        task_filter: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        top_k = top_k if top_k is not None else self.cfg.top_k
        max_per_frame = max_per_frame if max_per_frame is not None else self.cfg.max_per_frame
        task_filter = task_filter if task_filter is not None else self.cfg.task_filter

        # 1) BM25
        bm25_top = self._bm25_search(question, scene_context) if self.cfg.use_bm25 else []
        bm25_ranks = [i for i, _ in bm25_top]

        # 2) Dense
        hnsw_ranks: List[int] = []
        dists: List[float] = []
        if self.cfg.use_dense:
            labels, dists_arr = self._hnsw_search(question, scene_context)
            hnsw_ranks = labels.tolist()
            dists = dists_arr.tolist()

        # 3) RRF fusion
        fused = self._rrf(bm25_ranks, hnsw_ranks)
        ranked = sorted(fused.items(), key=lambda x: -x[1])

        # 4) Build result records
        results: List[dict] = []
        bm25_rank_map = {i: r for r, (i, _) in enumerate(bm25_top, start=1)}
        bm25_score_map = {i: s for i, s in bm25_top}
        hnsw_rank_map = {int(i): r for r, i in enumerate(hnsw_ranks, start=1)}
        hnsw_dist_map = {int(i): float(d) for i, d in zip(hnsw_ranks, dists) if self.cfg.use_dense}

        for pos, (i, rrf_score) in enumerate(ranked, start=1):
            row = self.meta.iloc[int(i)]
            rec = {
                "rank": pos,
                "rrf_score": float(rrf_score),
                "label": int(i),
                "chunk_id": row.get("chunk_id"),
                "chunk_type": row.get("chunk_type"),
                "task": row.get("task"),
                "scene_token": row.get("scene_token"),
                "frame_token": row.get("frame_token"),
                "image_path": row.get("image_path"),
                "num_qa": int(row.get("num_qa") or 0),
                "num_objects": int(row.get("num_objects") or 0),
                "bm25_rank": bm25_rank_map.get(int(i)),
                "bm25_score": bm25_score_map.get(int(i)),
                "hnsw_rank": hnsw_rank_map.get(int(i)),
                "hnsw_distance": hnsw_dist_map.get(int(i)),
                "text": row.get("text"),
                "text_for_embedding": row.get("text_for_embedding"),
                "rerank_score": None,
            }
            results.append(rec)

        # 5) Task filter (apply BEFORE rerank, so reranker doesn't see filtered-out items)
        if task_filter:
            allowed = set(task_filter)
            results = [r for r in results if r["task"] in allowed]

        # 6) Optional: cross-encoder rerank on the top pre_k
        if self.reranker is not None:
            q_text = self._encode_query(question, scene_context)
            pre = results[: self.cfg.pre_k]
            reranked = self.reranker.rerank(
                q_text, pre, top_k=None, doc_field=self.cfg.reranker_doc_field,
            )
            # Re-apply per-frame dedup AFTER rerank (now sorted by relevance)
            if max_per_frame and max_per_frame > 0:
                seen: dict = {}
                deduped: List[dict] = []
                for r in reranked:
                    f = r["frame_token"]
                    seen[f] = seen.get(f, 0) + 1
                    if seen[f] <= max_per_frame:
                        deduped.append(r)
                reranked = deduped
            return reranked[:top_k]

        # 7) No rerank: apply per-frame dedup then truncate
        if max_per_frame and max_per_frame > 0:
            seen: dict = {}
            deduped: List[dict] = []
            for r in results:
                f = r["frame_token"]
                seen[f] = seen.get(f, 0) + 1
                if seen[f] <= max_per_frame:
                    deduped.append(r)
            results = deduped
        return results[:top_k]

    # ---------- batch ----------
    def retrieve_batch(self, queries: Iterable[dict], top_k: Optional[int] = None) -> List[dict]:
        out = []
        for q in queries:
            r = self.retrieve(
                question=q.get("question", ""),
                scene_context=q.get("scene", "") or q.get("scene_description", "") or "",
                top_k=top_k,
            )
            out.append(
                {
                    "qid": q.get("qid") or q.get("id"),
                    "question": q.get("question", ""),
                    "image_path": q.get("image_path"),
                    "results": r,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ⏎ ")
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def print_results(query: str, scene: str, results: List[dict], show_text: int = 200,
                  with_rerank: bool = False) -> None:
    print("=" * 78)
    print(f"Q    : {query}")
    if scene:
        print(f"Scene: {scene}")
    if with_rerank:
        print(f"Top-{len(results)} (rerank | rrf | bm25 | hnsw):")
    else:
        print(f"Top-{len(results)} (rrf | bm25 | hnsw):")
    for r in results:
        bm = f"bm25#{r['bm25_rank']}" if r["bm25_rank"] else "bm25·"
        hn = f"hnsw#{r['hnsw_rank']}" if r["hnsw_rank"] else "hnsw·"
        if with_rerank and r.get("rerank_score") is not None:
            head = f"  [{r['rank']:>2}] rerank={r['rerank_score']:+.4f}  rrf={r['rrf_score']:.4f}  {bm}  {hn}  "
        else:
            head = f"  [{r['rank']:>2}] rrf={r['rrf_score']:.4f}  {bm}  {hn}  "
        print(head + f"task={r['task'] or '-':<10}  type={r['chunk_type']}")
        if r["text"]:
            print(f"       {_truncate(r['text'], show_text)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--reranker-dir", default=DEFAULT_RERANKER_DIR)
    ap.add_argument("--question", "-q", help="Single question to retrieve.")
    ap.add_argument("--scene", "-s", default="", help="Optional current scene description.")
    ap.add_argument("--image-path", default=None, help="Optional current image path (passed through to result).")
    ap.add_argument("--batch", help="JSONL of queries for batch mode. Each line: {qid, question, scene?, image_path?}")
    ap.add_argument("--output", "-o", help="Output JSONL for batch mode.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--k-bm25", type=int, default=50)
    ap.add_argument("--k-dense", type=int, default=50)
    ap.add_argument("--k-rrf", type=int, default=60)
    ap.add_argument("--pre-k", type=int, default=50, help="Candidates fed into the reranker (only used if --rerank).")
    ap.add_argument("--max-per-frame", type=int, default=0)
    ap.add_argument("--task", choices=TASKS, nargs="*", default=None, help="Restrict to one or more tasks.")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--rerank", action="store_true", help="Apply BGE cross-encoder rerank on the top pre_k candidates.")
    ap.add_argument("--no-rerank", action="store_true", help="Explicitly disable rerank (overrides --rerank).")
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument(
        "--rerank-doc-field", default="text_for_embedding",
        choices=("text_for_embedding", "text"),
        help="Which chunk text to feed the reranker.",
    )
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    use_rerank = bool(args.rerank) and not args.no_rerank
    cfg = RetrieveConfig(
        index_dir=os.path.abspath(args.index_dir),
        model_dir=os.path.abspath(args.model_dir),
        reranker_dir=os.path.abspath(args.reranker_dir),
        use_dense=not args.no_dense,
        use_bm25=not args.no_bm25,
        use_rerank=use_rerank,
        k_bm25=args.k_bm25,
        k_dense=args.k_dense,
        k_rrf=args.k_rrf,
        pre_k=args.pre_k,
        top_k=args.top_k,
        max_per_frame=args.max_per_frame,
        task_filter=list(args.task) if args.task else None,
        ef_search=args.ef_search,
        reranker_doc_field=args.rerank_doc_field,
    )
    retr = RAGRetriever(cfg)

    if args.batch:
        # Batch mode
        queries = []
        with open(args.batch, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                queries.append(json.loads(line))
        print(f"📥 Loaded {len(queries)} queries")

        out_f = open(args.output, "w", encoding="utf-8") if args.output else None
        t0 = time.time()
        for i, q in enumerate(queries, start=1):
            r = retr.retrieve(
                question=q.get("question", ""),
                scene_context=q.get("scene", "") or q.get("scene_description", "") or "",
                top_k=args.top_k,
            )
            record = {
                "qid": q.get("qid") or q.get("id"),
                "question": q.get("question", ""),
                "image_path": q.get("image_path"),
                "results": r,
            }
            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  ... {i}/{len(queries)}  ({time.time() - t0:.1f}s)")
        if out_f:
            out_f.close()
            print(f"💾 Saved: {args.output}")
        print(f"✅ Done {len(queries)} queries in {time.time() - t0:.1f}s")
        return

    if args.question:
        r = retr.retrieve(
            question=args.question,
            scene_context=args.scene,
            top_k=args.top_k,
        )
        print_results(args.question, args.scene, r, with_rerank=cfg.use_rerank)
        return

    print("Nothing to do. Pass --question or --batch.")


if __name__ == "__main__":
    main()

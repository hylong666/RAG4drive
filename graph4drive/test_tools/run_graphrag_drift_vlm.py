"""KG-DRIFT + VLM inference (Microsoft GraphRAG local search, vision-enabled).

Pipeline per question:
  1. Extract entity categories mentioned in the question.
  2. LOCAL retrieval (entity-aware DRIFT):
       - For each matched category → look up entities (Cat@Cam@Region) in entity KG.
       - Pool chunks from those entities → re-rank by BGE-M3 question similarity.
  3. GLOBAL retrieval (community search):
       - Encode question → BGE-M3 → HNSW over community summaries → top-k.
  4. VLM (Qwen2.5-VL-7B):
       - Prompt = [Global Knowledge community summaries] +
                  [Local Examples retrieved chunks] +
                  [Question]
       - Vision: attach current frame's CAM_FRONT image.
       - Generate answer.

Output: preds_kg_drift_vlm.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
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


# ---- category vocabulary used by the entity KG (must match build_entity_kg.py) ----
ENTITY_CATEGORIES = ["Vehicle", "Pedestrian", "Cycle", "Traffic element", "Others"]


def _extract_question_categories(question: str) -> set[str]:
    """Return the set of entity categories mentioned (case-insensitive substring)."""
    q = (question or "").lower()
    matched = set()
    for cat in ENTITY_CATEGORIES:
        if cat.lower() in q:
            matched.add(cat)
    # Domain synonyms → canonical category
    synonyms = {
        "vehicle": "Vehicle", "vehicles": "Vehicle", "car": "Vehicle", "cars": "Vehicle",
        "truck": "Vehicle", "trucks": "Vehicle", "sedan": "Vehicle", "suv": "Vehicle",
        "van": "Vehicle", "bus": "Vehicle", "bicycle": "Cycle", "bike": "Cycle",
        "motorcycle": "Cycle", "cyclist": "Cycle", "cyclists": "Cycle",
        "pedestrian": "Pedestrian", "pedestrians": "Pedestrian", "person": "Pedestrian",
        "people": "Pedestrian", "walker": "Pedestrian",
        "cone": "Traffic element", "cones": "Traffic element", "barrier": "Traffic element",
        "barriers": "Traffic element", "sign": "Traffic element", "signs": "Traffic element",
        "light": "Traffic element", "lights": "Traffic element",
    }
    for word, cat in synonyms.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            matched.add(cat)
    return matched


def _extract_object_ids(question: str) -> list[str]:
    """Find references like '<c1,CAM_BACK,1088.3,497.5>' in the question."""
    return re.findall(r"<([a-zA-Z0-9]+,[A-Z_]+,[0-9.\-]+,[0-9.\-]+)>", question or "")


_MCQ_OPTIONS_RE = re.compile(r"\bA\.\s+.*\bB\.\s+.*\bC\.\s+.*\bD\.", re.IGNORECASE | re.DOTALL)


def _is_mcq(q: str) -> bool:
    return bool(q and _MCQ_OPTIONS_RE.search(q))


def _extract_letter(text: str) -> str | None:
    if not text:
        return None
    for pat in [
        r"(?i)\banswer\s*[:\-]?\s*([ABCD])\b",
        r"(?m)^\s*([ABCD])[\.)]\s*",
        r"\b([ABCD])[\.)]\b",
        r"\b([ABCD])\b",
    ]:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-12)


def _iter_val_items(val_dict):
    for scene_id, scene_obj in val_dict.items():
        for frame_id, frame_obj in (scene_obj or {}).get("key_frames", {}).items():
            for category, qa_list in (frame_obj or {}).get("QA", {}).items():
                for it in qa_list or []:
                    if isinstance(it, dict) and it.get("Q"):
                        yield {
                            "question": it["Q"],
                            "answer": it.get("A", ""),
                            "scene_id": scene_id,
                            "frame_id": frame_id,
                            "category": category,
                            "image_paths": (frame_obj or {}).get("image_paths", {}) or {},
                        }


def load_val_data(path: str, n: int | None) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    items = list(_iter_val_items(raw)) if isinstance(raw, dict) else raw
    return items[:n] if n else items


def resolve_image_path(image_paths: dict, val_json_path: str) -> str | None:
    """Pick CAM_FRONT (or first available) and resolve relative to val_json_path.
    Backwards-compatible single-path resolver."""
    paths = resolve_camera_paths(image_paths, val_json_path, max_cams=1)
    return paths[0] if paths else None


PRIORITY_CAMS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def resolve_camera_paths(
    image_paths: dict,
    val_json_path: str,
    max_cams: int = 1,
) -> list[str]:
    """Return up to ``max_cams`` resolved absolute image paths in priority order."""
    if not image_paths or max_cams <= 0:
        return []
    base_dir = os.path.dirname(os.path.abspath(val_json_path))
    out: list[str] = []
    for cam in PRIORITY_CAMS[:max_cams]:
        rel = image_paths.get(cam)
        if not rel:
            continue
        base_name = os.path.basename(rel)
        candidates = [
            rel,
            os.path.join(base_dir, rel),
            os.path.join(base_dir, "nuscenes", "samples", cam, base_name),
            os.path.join(base_dir, "drivelm_nus_imgs_val", "val_data", cam, base_name),
        ]
        for p in candidates:
            if p and os.path.exists(p):
                out.append(os.path.abspath(p))
                break
        if len(out) >= max_cams:
            break
    return out


def build_prompt(
    question: str,
    community_summaries: list[dict],
    local_examples: list[dict],
    k_comm: int,
    k_local: int,
) -> str:
    if local_examples:
        ctx_intro = (
            "You are an autonomous driving expert. Look at the current driving "
            "scene image carefully. Use the provided global driving knowledge "
            "(scenario archetypes from a knowledge graph) AND the retrieved "
            "local examples (specific historical Q-A pairs involving similar "
            "entities) to answer. Be precise: when the question references a "
            "specific object ID like <c1,CAM_BACK,...>, ground your answer in "
            "what you actually see in the image. Answer in English only."
        )
    else:
        ctx_intro = (
            "You are an autonomous driving expert. Look at the current driving "
            "scene image carefully. The provided global driving knowledge "
            "(scenario archetypes from a knowledge graph) describes the kind "
            "of scene this question likely belongs to. Choose the answer that "
            "best matches this scenario type. Answer in English only."
        )
    parts = [ctx_intro]
    if community_summaries:
        parts.append("\n[Global Knowledge — driving scene archetypes]")
        for i, c in enumerate(community_summaries[:k_comm], 1):
            parts.append(f"\nScenario archetype {i}:\n{c['summary'].strip()}")
    if local_examples:
        parts.append("\n[Local Examples — historical Q-A involving similar entities]")
        for i, ex in enumerate(local_examples[:k_local], 1):
            parts.append(
                f"\nExample {i}:\nQ: {ex['question'][:300]}\nA: {ex['answer'][:200]}"
            )
    parts.append(f"\n[Question]\n{question.strip()}")
    parts.append("\n[Answer]")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-path", default="./data/json/test_eval.json")
    p.add_argument("--n-samples", type=int, default=66)
    p.add_argument(
        "--comm-index",
        default="./graphrag_community/entity_communities/hnsw_bge_m3_entity_communities.bin",
    )
    p.add_argument(
        "--comm-meta",
        default="./graphrag_community/entity_communities/entity_community_metadata.pkl",
    )
    p.add_argument(
        "--chunks-index",
        default="./graphrag_community/chunks/hnsw_bge_m3_chunks.bin",
    )
    p.add_argument(
        "--chunks-meta",
        default="./graphrag_community/chunks/chunks_metadata.pkl",
    )
    p.add_argument(
        "--entity-chunk-sets",
        default="./graphrag_community/entity_kg/entity_chunk_sets.pkl",
        help="Pickle mapping entity → list[chunk_id] for entity-aware filtering.",
    )
    p.add_argument("--embed-model", default="./models/bge-m3")
    p.add_argument("--embed-device", default="cuda")
    p.add_argument(
        "--vlm-path",
        default="/home/hylong/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/",
    )
    p.add_argument("--k-communities", type=int, default=2)
    p.add_argument("--k-local", type=int, default=4)
    p.add_argument("--local-pool-size", type=int, default=200,
                   help="Max chunks to re-rank from the entity-aware candidate pool.")
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=192,
                   help="Max new tokens to generate. Default 192 covers planning answers "
                        "(often 3-5 sentences) without truncation.")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-dir", default="./outputs_kg_drift_vlm")
    p.add_argument("--metrics-file", default="metrics.json")
    p.add_argument("--no-image", action="store_true",
                   help="Skip image input (text-only ablation).")
    p.add_argument(
        "--multi-camera", type=int, default=6, choices=[1, 4, 6],
        help="Number of camera images per query. Default 6 (full surround) gives the "
             "best planning F1 (+5pp vs 4-cam). Use 1 or 4 to save VRAM.",
    )
    p.add_argument(
        "--mcq-global-only", dest="mcq_global_only", action="store_true", default=True,
        help="(Default ON) For MCQ questions, skip local retrieval and use only "
             "global community summaries. Disable with --no-mcq-global-only.",
    )
    p.add_argument(
        "--no-mcq-global-only", dest="mcq_global_only", action="store_false",
        help="Disable MCQ routing (MCQ will use full local + global like other questions).",
    )
    p.add_argument("--bm25-k", type=int, default=20,
                   help="BM25 top-N candidates to feed into reranker (default 20).")
    p.add_argument("--no-bm25", action="store_true",
                   help="Disable BM25 cascade (use only BGE-M3 + entity filter).")
    p.add_argument("--no-rerank", action="store_true",
                   help="Disable BGE reranker cascade.")
    p.add_argument("--entity-hnsw", default="./graphrag_community/entity_kg/entity_hnsw.bin",
                   help="Entity-level HNSW index (entity embedding).")
    p.add_argument("--entity-meta", default="./graphrag_community/entity_kg/entity_metadata.pkl",
                   help="Entity metadata (entity_key, description).")
    p.add_argument("--no-entity-embedding", action="store_true",
                   help="Disable entity-level dense retrieval (default ON).")
    p.add_argument("--entity-embedding-k", type=int, default=10,
                   help="Top-K entities to recall from entity-HNSW (default 10).")
    p.add_argument("--planning-k-local", type=int, default=None,
                   help="Optional override of k_local for planning questions. "
                        "If None, --k_local is used for all tasks. Originally set "
                        "to 2 (planning→less text), but ablation showed the gain "
                        "was drowned by VLM non-determinism (single-run variance "
                        "±0.12 on planning F1), so we default to None for safety.")
    args = p.parse_args()

    # Resolve vlm path: if it's a snapshots/ dir, pick first child
    vlm_path = args.vlm_path
    if os.path.isdir(vlm_path) and not os.path.exists(os.path.join(vlm_path, "config.json")):
        children = sorted(os.listdir(vlm_path))
        if children:
            vlm_path = os.path.join(vlm_path, children[0])
            print(f"[vlm] resolved snapshot: {vlm_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] val from {args.val_path}")
    val_data = load_val_data(args.val_path, args.n_samples)
    print(f"[load] {len(val_data)} items")

    # Community HNSW
    print(f"[load] community index from {args.comm_index}")
    import hnswlib
    comm_idx = hnswlib.Index(space="cosine", dim=1024)
    comm_idx.load_index(args.comm_index)
    comm_idx.set_ef(args.ef)
    comm_meta = pd.read_pickle(args.comm_meta)
    print(f"[load] {len(comm_meta)} community summaries")

    # Chunks HNSW
    print(f"[load] chunks index from {args.chunks_index}")
    chunks_idx = hnswlib.Index(space="cosine", dim=1024)
    chunks_idx.load_index(args.chunks_index)
    chunks_idx.set_ef(args.ef)
    chunks_meta = pd.read_pickle(args.chunks_meta)
    print(f"[load] {len(chunks_meta)} chunks")

    # Entity → chunks mapping
    print(f"[load] entity → chunks from {args.entity_chunk_sets}")
    with open(args.entity_chunk_sets, "rb") as f:
        ent_to_chunks = pickle.load(f)
    print(f"[load] {len(ent_to_chunks)} entities")

    # Build category → entities index for fast lookup
    cat_to_ents: dict[str, list[str]] = defaultdict(list)
    for e in ent_to_chunks.keys():
        cat = e.split("@")[0]
        cat_to_ents[cat].append(e)
    print(f"[plan] category → entity count: { {c: len(v) for c, v in cat_to_ents.items()} }")

    # Load entity HNSW (entity-level dense retrieval — GraphRAG local-search layer)
    entity_idx = None
    entity_meta = None
    if not args.no_entity_embedding:
        if args.entity_hnsw and os.path.exists(args.entity_hnsw):
            print(f"[load] entity HNSW from {args.entity_hnsw}")
            entity_idx = hnswlib.Index(space="cosine", dim=1024)
            entity_idx.load_index(args.entity_hnsw)
            entity_idx.set_ef(args.ef)
            entity_meta = pd.read_pickle(args.entity_meta)
            print(f"[load] {len(entity_meta)} entity embeddings")
        else:
            print(f"[warn] --no-entity-embedding implied: {args.entity_hnsw} not found")

    # Embedder
    print(f"[load] embedder: {args.embed_model}")
    from FlagEmbedding import BGEM3FlagModel
    emb = BGEM3FlagModel(args.embed_model, use_fp16=True, device=args.embed_device)

    # VLM
    print(f"[load] VLM: {vlm_path}")
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tok = AutoTokenizer.from_pretrained(vlm_path, trust_remote_code=True)
    model = model_cls.from_pretrained(
        vlm_path,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ---- helper: entity-aware local retrieval with optional BM25 + rerank cascade ----
    # Load BM25 if available and not disabled
    bm25 = None
    bm25_chunk_ids = None
    if not args.no_bm25:
        bm25_path = os.path.join(
            os.path.dirname(args.chunks_index), "bm25_chunks.pkl"
        )
        if os.path.exists(bm25_path):
            print(f"[load] BM25 from {bm25_path}")
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            bm25 = bm25_data["bm25"]
            bm25_chunk_ids = bm25_data["chunk_ids"]  # list[str], parallel to bm25 corpus
            print(f"[load] BM25 corpus size: {len(bm25_chunk_ids)}")
        else:
            print(f"[warn] --no-bm25 implied: BM25 index not found at {bm25_path}")

    # Load BGE reranker if available and not disabled
    reranker = None
    if not args.no_rerank:
        # RAGdrive stores the model in a HF-style cache_dir layout where the
        # parent dir contains blobs/snapshots/<hash>/; we pass cache_dir to
        # AutoModel so it picks up the right snapshot.
        reranker_cache_dirs = [
            "/home/hylong/xczn/driveVQA/RAGdrive/models/bge-reranker-large",
        ]
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            loaded = False
            for cache_dir in reranker_cache_dirs:
                if not os.path.isdir(cache_dir):
                    continue
                try:
                    print(f"[load] reranker (BAAI/bge-reranker-large) cache_dir={cache_dir}")
                    rerank_tok = AutoTokenizer.from_pretrained(
                        "BAAI/bge-reranker-large", cache_dir=cache_dir
                    )
                    rerank_model = AutoModelForSequenceClassification.from_pretrained(
                        "BAAI/bge-reranker-large",
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                    rerank_model.eval()
                    if torch.cuda.is_available():
                        rerank_model.to("cuda")
                    print("[load] reranker loaded")

                    class _BGERerankerWrapper:
                        def compute_score(self, pairs):
                            with torch.inference_mode():
                                scores = []
                                for i in range(0, len(pairs), 16):
                                    batch = pairs[i:i + 16]
                                    enc = rerank_tok(
                                        batch,
                                        padding=True,
                                        truncation=True,
                                        max_length=512,
                                        return_tensors="pt",
                                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                                    logits = rerank_model(**enc).logits.squeeze(-1)
                                    scores.extend(logits.float().cpu().tolist())
                                return scores

                    reranker = _BGERerankerWrapper()
                    loaded = True
                    break
                except Exception as e:
                    print(f"[warn] failed to load reranker from {cache_dir}: {e}")
            if not loaded:
                print("[warn] --no-rerank implied: reranker not loadable")
        except ImportError:
            print("[warn] transformers not available for reranker; skipping")

    def _bm25_rerank_indices(question: str, candidate_indices: list[int]) -> list[int]:
        """Apply BM25 to candidate chunks (already filtered by entity / HNSW)."""
        if bm25 is None or not candidate_indices:
            return candidate_indices
        import re as _re
        toks = _re.findall(r"[A-Za-z0-9_<>]+", question.lower())
        # bm25.get_scores takes token list
        scores = bm25.get_scores(toks)
        # Restrict to candidate_indices
        cand_scores = [(i, scores[i]) for i in candidate_indices]
        cand_scores.sort(key=lambda x: -x[1])
        return [i for i, _ in cand_scores[: args.bm25_k]]

    def _rerank_indices(question: str, candidate_indices: list[int],
                        k_local: int | None = None) -> list[int]:
        """Apply cross-encoder reranker to BM25-shortlisted candidates."""
        if k_local is None:
            k_local = args.k_local
        if reranker is None or not candidate_indices:
            return candidate_indices[: k_local]
        pairs = [[question, chunks_meta.iloc[i]["text"]] for i in candidate_indices]
        try:
            scores = reranker.compute_score(pairs)
        except Exception as e:
            print(f"[warn] rerank failed: {e}")
            return candidate_indices[: k_local]
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(candidate_indices, scores), key=lambda x: -x[1])
        return [i for i, _ in scored[: k_local]]

    def local_retrieve(question: str, k_local: int | None = None) -> list[dict]:
        """k_local override lets us pass a smaller k for planning questions."""
        if k_local is None:
            k_local = args.k_local

        # Stage 0 (entity-level dense retrieval): find entities similar to question
        # — Microsoft GraphRAG local-search pattern
        entity_recall: set[str] = set()
        if entity_idx is not None and entity_meta is not None:
            enc = emb.encode([question])
            dense = enc["dense_vecs"] if isinstance(enc, dict) else enc
            qv = _l2_normalize(dense[0])
            e_ids, _ = entity_idx.knn_query(qv, k=args.entity_embedding_k)
            for eid in e_ids[0]:
                entity_key = entity_meta.iloc[int(eid)]["entity_key"]
                entity_recall.add(entity_key)
            # print(f"[local] entity_recall={list(entity_recall)[:5]}…")  # debug

        # Stage 1: build candidate pool (entity-filtered, capped)
        cats = _extract_question_categories(question)
        if not cats:
            enc = emb.encode([question])
            dense = enc["dense_vecs"] if isinstance(enc, dict) else enc
            qv = _l2_normalize(dense[0])
            ids, _ = chunks_idx.knn_query(qv, k=max(args.bm25_k * 2, args.local_pool_size))
            cand_indices = list(ids[0])
        else:
            candidate_chunk_ids: set[str] = set()
            # Pool from category-matched entities
            for cat in cats:
                for e in cat_to_ents.get(cat, []):
                    candidate_chunk_ids.update(ent_to_chunks.get(e, []))
            # ALSO pool from entity-embedding-recalled entities (GraphRAG-style)
            for e in entity_recall:
                candidate_chunk_ids.update(ent_to_chunks.get(e, []))
            cid_to_row = {cid: i for i, cid in enumerate(chunks_meta["chunk_id"].tolist())}
            cand_indices = [cid_to_row[c] for c in candidate_chunk_ids if c in cid_to_row]
            if not cand_indices:
                enc = emb.encode([question])
                dense = enc["dense_vecs"] if isinstance(enc, dict) else enc
                qv = _l2_normalize(dense[0])
                ids, _ = chunks_idx.knn_query(qv, k=max(args.bm25_k * 2, args.local_pool_size))
                cand_indices = list(ids[0])

        if not cand_indices:
            return []

        # Cap candidate pool before cascade (BGE reranker is expensive)
        if len(cand_indices) > args.local_pool_size:
            import random
            random.seed(0)
            cand_indices = random.sample(cand_indices, args.local_pool_size)

        # Stage 2 (optional): BM25 → top-N
        if bm25 is not None:
            cand_indices = _bm25_rerank_indices(question, cand_indices)

        # Stage 3 (optional): BGE reranker → top-k
        if reranker is not None:
            cand_indices = _rerank_indices(question, cand_indices, k_local=k_local)
        else:
            # Fallback: BGE-M3 cosine on the (possibly BM25-shortlisted) pool
            cand_texts = [chunks_meta.iloc[i]["text"] for i in cand_indices]
            cand_enc = emb.encode(cand_texts, batch_size=64, max_length=256)
            cand_dense = np.asarray(
                cand_enc["dense_vecs"] if isinstance(cand_enc, dict) else cand_enc,
                dtype=np.float32,
            )
            cand_dense = cand_dense / (np.linalg.norm(cand_dense, axis=1, keepdims=True) + 1e-12)
            q_enc = emb.encode([question])
            q_dense = np.asarray(q_enc["dense_vecs"] if isinstance(q_enc, dict) else q_enc,
                                 dtype=np.float32)
            q_dense = _l2_normalize(q_dense[0])
            sims = cand_dense @ q_dense
            top = np.argsort(-sims)[: k_local]
            cand_indices = [cand_indices[i] for i in top]

        return [chunks_meta.iloc[i].to_dict() for i in cand_indices[: k_local]]

    # ---- inference loop ----
    preds: list[str] = []
    refs: list[list[str]] = []
    records: list[dict] = []

    # Try to import qwen_vl_utils for image processing
    try:
        from qwen_vl_utils import process_vision_info
        has_vl_utils = True
    except ImportError:
        has_vl_utils = False
        print("[warn] qwen_vl_utils not available; will use PIL fallback")

    print(f"\n🚀 kg_drift_vlm mode: {len(val_data)} items")
    n_mcq_global_only = 0
    with torch.inference_mode():
        for item in tqdm(val_data, desc="[inference]"):
            question = item["question"]

            # 1) Community retrieval (always)
            enc = emb.encode([question])
            dense_vecs = enc["dense_vecs"] if isinstance(enc, dict) else enc
            q_vec = _l2_normalize(dense_vecs[0])
            c_ids, _ = comm_idx.knn_query(q_vec, k=args.k_communities)
            comm_rows = comm_meta.iloc[c_ids[0]]
            community_summaries = [
                {
                    "community_id": int(r["community_id"]),
                    "summary": r["summary"],
                    "n_entities": int(r["n_entities"]),
                    "n_chunks": int(r["n_chunks"]),
                }
                for _, r in comm_rows.iterrows()
            ]

            # 2) Local retrieval (skip if MCQ and --mcq-global-only)
            is_mcq = _is_mcq(question)
            task = item.get("category") or ""
            if is_mcq and args.mcq_global_only:
                local_examples = []
                n_mcq_global_only += 1
                effective_k_local = 0
            elif task == "planning" and args.planning_k_local is not None:
                # planning 类 k_local 降到 --planning-k-local(默认 None,fallback 到 --k_local)
                local_examples = local_retrieve(question, k_local=args.planning_k_local)
                effective_k_local = args.planning_k_local
            else:
                local_examples = local_retrieve(question)
                effective_k_local = args.k_local

            # 3) Build prompt
            prompt_text = build_prompt(
                question,
                community_summaries,
                local_examples,
                k_comm=args.k_communities,
                k_local=effective_k_local,
            )

            # 4) Images (multi-camera)
            image_paths: list[str] = []
            if not args.no_image:
                image_paths = resolve_camera_paths(
                    item.get("image_paths", {}) or {},
                    args.val_path,
                    max_cams=args.multi_camera,
                )

            # 5) VLM generate
            try:
                if image_paths and has_vl_utils:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image", "image": p} for p in image_paths],
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    text_input = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = tok(
                        text=[text_input],
                        images=image_inputs if image_inputs else None,
                        videos=video_inputs if video_inputs else None,
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)
                elif image_paths:
                    # PIL fallback
                    from PIL import Image
                    imgs = [Image.open(p).convert("RGB") for p in image_paths]
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image", "image": im} for im in imgs],
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    text_input = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tok(
                        text=[text_input],
                        images=imgs,
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)
                else:
                    # text-only
                    messages = [{"role": "user", "content": prompt_text}]
                    text_input = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tok(text_input, return_tensors="pt").to(model.device)

                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                )
                gen = tok.decode(
                    out_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()
            except Exception as e:
                gen = f"(generation failed: {type(e).__name__}: {e})"

            pred = gen
            if _is_mcq(question):
                letter = _extract_letter(pred)
                if letter:
                    pred = letter

            preds.append(pred)
            refs.append([item.get("answer", "")])
            records.append({
                "prediction": pred,
                "ref": item.get("answer", ""),
                "question": question,
                "scene_id": item.get("scene_id"),
                "frame_id": item.get("frame_id"),
                "category": item.get("category"),
                "image_path": image_paths[0] if image_paths else "",
                "image_paths": image_paths,
                "n_cameras": len(image_paths),
                "communities_used": [c["community_id"] for c in community_summaries],
                "n_local_examples": len(local_examples),
                "effective_k_local": effective_k_local,
                "categories_matched": sorted(_extract_question_categories(question)),
            })

    out_preds = os.path.join(args.output_dir, "preds_kg_drift_vlm.jsonl")
    with open(out_preds, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[save] wrote {out_preds}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "kg_drift_vlm",
        "val_path": args.val_path,
        "n_samples": len(val_data),
        "vlm_path": vlm_path,
        "k_communities": args.k_communities,
        "k_local": args.k_local,
        "local_pool_size": args.local_pool_size,
        "multi_camera": args.multi_camera,
        "mcq_global_only": args.mcq_global_only,
        "n_mcq_global_only": n_mcq_global_only,
        "use_bm25": bm25 is not None,
        "use_rerank": reranker is not None,
        "no_image": args.no_image,
    }
    out_metrics = os.path.join(args.output_dir, args.metrics_file)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[save] wrote {out_metrics}")


if __name__ == "__main__":
    main()

"""Microsoft GraphRAG 'global search' inference.

Pipeline:
  1. Encode the question with BGE-M3.
  2. Search the community-summaries HNSW index → top-k community summaries.
  3. Build the prompt with [Global Knowledge — driving scene archetypes] +
     [Question].
  4. Generate answer with the LLM.

Output: preds_community_graphrag.jsonl  (same format as run_comparison.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
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


_MCQ_OPTIONS_RE = re.compile(r"\bA\.\s+.*\bB\.\s+.*\bC\.\s+.*\bD\.", re.IGNORECASE | re.DOTALL)


def _is_multiple_choice_question(q: str) -> bool:
    return bool(q and _MCQ_OPTIONS_RE.search(q))


def _extract_choice_letter(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"(?i)\banswer\s*[:\-]?\s*([ABCD])\b",
        r"(?m)^\s*([ABCD])[\.)]\s*",
        r"\b([ABCD])[\.)]\b",
        r"\b([ABCD])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()
    return None


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-12)


def _iter_val_items_from_nested_dict(val_dict):
    for scene_id, scene_obj in val_dict.items():
        key_frames = (scene_obj or {}).get("key_frames", {})
        if not isinstance(key_frames, dict):
            continue
        for frame_id, frame_obj in key_frames.items():
            qa = (frame_obj or {}).get("QA", {})
            if not isinstance(qa, dict):
                continue
            for category, qa_list in qa.items():
                if not isinstance(qa_list, list):
                    continue
                for qa_item in qa_list:
                    if not isinstance(qa_item, dict):
                        continue
                    q = qa_item.get("Q")
                    if not q:
                        continue
                    yield {
                        "question": q,
                        "answer": qa_item.get("A", ""),
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "category": category,
                        "tag": qa_item.get("tag"),
                    }


def load_val_data(val_path: str, n_samples: int | None) -> list[dict]:
    with open(val_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = list(_iter_val_items_from_nested_dict(raw))
    else:
        raise ValueError(f"Unsupported val.json format: root type={type(raw)}")
    if n_samples is not None:
        items = items[:n_samples]
    return items


def build_global_prompt(
    question: str,
    community_summaries: list[dict],
    k_communities: int,
) -> str:
    """Build the LLM prompt for community_graphrag mode.

    Sections:
      [Global Knowledge]    community summaries (driving scenario archetypes)
      [Question]            the actual question
    """
    parts = [
        "You are an autonomous driving expert. Answer the question using the "
        "provided global driving knowledge (scenario archetypes from a "
        "knowledge graph of driving scenes). Answer in English only. Output "
        "only the answer text.",
        "\n[Global Knowledge — driving scene archetypes]",
    ]
    for i, c in enumerate(community_summaries[:k_communities], 1):
        parts.append(
            f"\nScenario archetype {i}:\n{c.get('summary', '').strip()}"
        )
    parts.append(f"\n[Question]\n{question.strip()}")
    parts.append("\n[Answer]")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-path", default="./v1_1_val_nus_q_only.json")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument(
        "--comm-index",
        default="./graphrag_community/entity_communities/hnsw_bge_m3_entity_communities.bin",
    )
    p.add_argument(
        "--comm-meta",
        default="./graphrag_community/entity_communities/entity_community_metadata.pkl",
    )
    p.add_argument(
        "--embed-model", default="./models/bge-m3",
    )
    p.add_argument("--embed-device", default="cuda")
    p.add_argument(
        "--vlm-path",
        default="/home/hylong/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/",
    )
    p.add_argument("--k-communities", type=int, default=2)
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--output-dir", default="./outputs_community_graphrag")
    p.add_argument(
        "--metrics-file", default="metrics.json",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] val data from {args.val_path}")
    val_data = load_val_data(args.val_path, args.n_samples)
    print(f"[load] {len(val_data)} items")

    print(f"[load] community index from {args.comm_index}")
    import hnswlib
    comm_idx = hnswlib.Index(space="cosine", dim=1024)
    comm_idx.load_index(args.comm_index)
    comm_idx.set_ef(args.ef)
    comm_meta = pd.read_pickle(args.comm_meta)
    print(f"[load] {len(comm_meta)} community summaries")

    print(f"[load] embedder: {args.embed_model}")
    from FlagEmbedding import BGEM3FlagModel
    emb = BGEM3FlagModel(args.embed_model, use_fp16=True, device=args.embed_device)

    print(f"[load] LLM: {args.vlm_path}")
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tok = AutoTokenizer.from_pretrained(args.vlm_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.vlm_path,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    preds: list[str] = []
    refs: list[list[str]] = []
    records: list[dict] = []

    print(f"\n🚀 community_graphrag mode: {len(val_data)} items")
    with torch.inference_mode():
        for item in tqdm(val_data, desc="[inference]"):
            question = item["question"]
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

            prompt = build_global_prompt(
                question,
                community_summaries,
                k_communities=args.k_communities,
            )

            inputs = tok(prompt, return_tensors="pt").to(model.device)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.1,
            )
            pred = tok.decode(
                out_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            if _is_multiple_choice_question(question):
                letter = _extract_choice_letter(pred)
                if letter:
                    pred = letter

            preds.append(pred)
            refs.append([item.get("answer", "")])
            records.append({
                "pred": pred,
                "ref": item.get("answer", ""),
                "question": question,
                "scene_id": item.get("scene_id"),
                "frame_id": item.get("frame_id"),
                "category": item.get("category"),
                "tag": item.get("tag"),
                "communities_used": [c["community_id"] for c in community_summaries],
            })

    out_preds = os.path.join(args.output_dir, "preds_community_graphrag.jsonl")
    with open(out_preds, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[save] wrote {out_preds}")

    has_any_ref = any((rl and (rl[0] or "").strip()) for rl in refs)
    metrics = None
    if has_any_ref:
        try:
            try:
                from test_tools.eval_driveLM_metrics import DriveLMEvaluator
            except Exception:
                from eval_driveLM_metrics import DriveLMEvaluator

            judge = DriveLMEvaluator(
                judge_tokenizer=tok, judge_model=model, device=str(model.device)
            )
            metrics = judge.evaluate(preds, refs)
        except Exception as e:
            print(f"[warn] eval skipped: {type(e).__name__}: {e}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "community_graphrag",
        "val_path": args.val_path,
        "n_samples": len(val_data),
        "vlm_path": args.vlm_path,
        "k_communities": args.k_communities,
        "metrics": metrics,
    }
    out_metrics = os.path.join(args.output_dir, args.metrics_file)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[save] wrote {out_metrics}")
    print(f"\n📊 community_graphrag metrics: {metrics}")


if __name__ == "__main__":
    main()

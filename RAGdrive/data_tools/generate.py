"""Generate VQA answers with cascade retrieval + Qwen2-VL-7B.

This is the final stage of the RAG pipeline:

    1. For each input query (current frame + scene + question)
    2. Call cascade_retrieve to fetch top-K similar historical Q-A chunks
    3. Build a VLM prompt: current image + scene + retrieved few-shot + question
    4. Ask Qwen2-VL-7B-Instruct to generate the answer
    5. Save predictions to a JSONL

Inputs
------
- ``--input``  JSONL with one query per line:
        {
          "qid":   "...",
          "question":  "...",
          "scene":     "...",   # optional, current scene description
          "image_path":"...",   # path to current CAM_FRONT image
          "task":      "perception|prediction|planning|behavior"  # optional
        }

Outputs
-------
- ``--output`` JSONL with predictions:
        {
          "qid", "question", "image_path", "task",
          "retrieved":   [...top-K chunks...],
          "prompt_text": "...",   # the full VLM prompt text
          "prediction":  "...",   # VLM answer
          "latency_ms":  ...
        }

Run (smoke test)
----------------
::

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" HF_ENDPOINT=https://hf-mirror.com
    python3 generate.py \\
        --input queries.jsonl --output preds.jsonl \\
        --n 200 --top-k 5

The ``--n`` cap lets you run a fast smoke test; remove it for the full run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional

import torch
from PIL import Image

# Disable cuDNN due to a CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on this
# environment (PyTorch 2.12 bundled cuDNN 9.20 conflicts with what the
# Qwen2.5-VL vision tower's F.conv3d expects). Conv falls back to cuBLAS —
# slightly slower but stable.
torch.backends.cudnn.enabled = False

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Reuse cascade retriever
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cascade_retrieve import CascadeRetriever, CascadeConfig


# ---------------------------------------------------------------------------
# Default model + paths
# ---------------------------------------------------------------------------
DEFAULT_VLM_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/main"
)
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-large-en-v1.5")
DEFAULT_RERANKER_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-reranker-large")


# ---------------------------------------------------------------------------
# Helpers: extract Q/A from a retrieved chunk's text_for_embedding
# ---------------------------------------------------------------------------
_QA_RE = re.compile(
    r"^Q:\s*(?P<q>.+?)\nA:\s*(?P<a>.+?)(?=\nContext:|\Z)", re.DOTALL | re.MULTILINE
)


def _extract_qa(text: str) -> tuple[str, str]:
    """Pull (Q, A) out of a chunk's text_for_embedding.
    Returns ("", "") if not parseable.
    """
    m = _QA_RE.search(text or "")
    if not m:
        return "", ""
    return m.group("q").strip(), m.group("a").strip()


def _extract_context(text: str) -> str:
    """Pull the 'Context: ...' line(s) from text_for_embedding."""
    m = re.search(r"^Context:\s*(.+?)\Z", text or "", re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Prompt builder (v1 — flat prompt, few-shot teaches the format)
# ---------------------------------------------------------------------------
# Earlier versions of this function had per-task system prompts, CoT triggers,
# and format reminders. Empirically those hurt: with v2 (strong per-task
# structure) MCQ dropped from 41% → 18% and token-F1 from 0.50 → 0.15
# because the model produced verbose, off-format markdown instead of matching
# the few-shot style. v3 reverted most of the structure. This v1 is the
# further-reverted baseline: a single flat system prompt, and the retrieved
# few-shot carries all the format cues.


def _build_prompt_text(
    retrieved: List[dict],
    scene: str,
    question: str,
    max_examples: int = 5,
) -> str:
    """Build the text portion of the VLM prompt.

    Layout::

        <system: autonomous driving assistant>
        <current image>

        Current scene: {scene}

        Here are similar historical Q-A examples from past driving scenes:

        Example 1:
        Q: ...
        A: ...

        ...

        Question: {question}
        Answer:
    """
    parts: List[str] = []
    parts.append(
        "You are an autonomous-driving assistant. Look at the current driving scene image and "
        "answer the question. Use the current scene description and the similar historical "
        "question-answer examples below for context, but your answer should be grounded in the "
        "current scene. Be concise and direct."
    )
    if scene:
        parts.append(f"\nCurrent scene: {scene}")

    if retrieved:
        parts.append("\nSimilar historical Q-A examples from past driving scenes:")
        for i, c in enumerate(retrieved[:max_examples], start=1):
            q, a = _extract_qa(c.get("text_for_embedding", ""))
            if not q:
                continue
            parts.append(f"\nExample {i}:\nQ: {q}\nA: {a}")

    parts.append(f"\nQuestion: {question}")
    parts.append("Answer:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class VLMGenerator:
    """Wraps Qwen2-VL-7B-Instruct + cascade retriever."""

    def __init__(
        self,
        vlm_dir: str,
        index_dir: str,
        bge_model_dir: str,
        reranker_dir: str,
        hnsw_k: int = 100,
        bm25_k: int = 20,
        top_k: int = 5,
        use_rerank: bool = True,
        max_new_tokens: int = 256,
    ):
        print(f"📦 Loading cascade retriever (hnsw_k={hnsw_k}, bm25_k={bm25_k}, top_k={top_k})…")
        cascade_cfg = CascadeConfig(
            index_dir=os.path.abspath(index_dir),
            model_dir=os.path.abspath(bge_model_dir),
            reranker_dir=os.path.abspath(reranker_dir),
            hnsw_k=hnsw_k, bm25_k=bm25_k, top_k=top_k,
            use_rerank=use_rerank, ef_search=128,
        )
        self.retriever = CascadeRetriever(cascade_cfg)

        print(f"📦 Loading VLM: {vlm_dir}")
        self.processor = AutoProcessor.from_pretrained(vlm_dir)
        # Use AutoModel for VL since Qwen2.5-VL uses Qwen2_5_VLForConditionalGeneration
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_dir,
                torch_dtype=torch.float16,
                device_map="cuda",
                attn_implementation="sdpa",  # avoid cuDNN version mismatch with flash-attn
            )
        except ImportError:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                vlm_dir,
                torch_dtype=torch.float16,
                device_map="cuda",
                attn_implementation="sdpa",
            )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print("✅ VLM loaded")

    @torch.inference_mode()
    def generate(self, image_paths: list[str], prompt_text: str) -> str:
        """Run one VLM forward pass with 1+ camera images. Returns the answer string."""
        if not image_paths:
            image_paths = []
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": p} for p in image_paths],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,            # greedy
        )
        # Strip the prompt portion to get only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = gen_ids[:, prompt_len:]
        out = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return out.strip()

    def run_query(self, q: dict) -> dict:
        """Full pipeline: retrieve -> build prompt -> generate."""
        # Support both legacy single-path and new list-of-paths
        if "image_paths" in q and isinstance(q["image_paths"], list):
            image_paths = [p for p in q["image_paths"] if p and os.path.exists(p)]
        else:
            single = q.get("image_path", "")
            image_paths = [single] if single and os.path.exists(single) else []

        question = q.get("question", "")
        scene = q.get("scene", "") or q.get("scene_description", "")
        if not question:
            return {"qid": q.get("qid"), "prediction": "", "error": "no question"}
        if not image_paths:
            return {"qid": q.get("qid"), "prediction": "", "error": "no image found"}

        t0 = time.time()
        retrieved = self.retriever.retrieve(
            question=question, scene_context=scene, top_k=None,
        )
        prompt_text = _build_prompt_text(retrieved, scene, question, max_examples=len(retrieved))
        try:
            prediction = self.generate(image_paths, prompt_text)
        except Exception as e:
            prediction = ""
            err = f"VLM error: {e}"
        else:
            err = None
        latency = (time.time() - t0) * 1000
        return {
            "qid": q.get("qid"),
            "question": question,
            "scene": scene,
            "image_path": image_paths[0] if image_paths else "",
            "image_paths": image_paths,
            "task": q.get("task"),
            "retrieved": retrieved,
            "prompt_text": prompt_text,
            "prediction": prediction,
            "latency_ms": round(latency, 1),
            "error": err,
        }


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def _load_queries_from_jsonl(path: str, n: int) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if n and len(out) >= n:
                break
    return out


# Priority cameras for multi-camera inference
PRIORITY_CAMS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def _expand_query_to_multi_camera(q: dict, max_cams: int = 1) -> dict:
    """If a query has ``image_paths`` map (cam → rel_path), expand it into
    an ordered list of absolute paths respecting PRIORITY_CAMS up to
    ``max_cams``.  Existing ``image_paths`` list on the query is preserved.
    """
    if max_cams <= 1:
        return q
    if q.get("image_paths") and isinstance(q["image_paths"], list) and len(q["image_paths"]) > 1:
        return q  # already a list
    paths_map = q.get("image_paths_map") or {}
    if not paths_map:
        # Maybe the jsonl row already has a single image_path; leave it alone.
        return q
    return {**q, "image_paths": q.get("image_paths", [])}
    # (the actual expansion happens in run_query → generate path)


def _load_queries_from_val_json(path: str, n: int, data_root: str,
                                  multi_camera: int = 1) -> List[dict]:
    """Load queries from the val JSON (which has Q only). The val JSON's
    image_paths field uses a misleading ``../nuscenes/...`` prefix; the
    actual files live in ``<data_root>/nuscenes/samples/<CAM>/`` for train
    scenes or ``<data_root>/drivelm_nus_imgs_val/val_data/<CAM>/`` for val.

    For multi-camera we resolve ALL available cameras in priority order:
      CAM_FRONT > CAM_FRONT_LEFT > CAM_FRONT_RIGHT > CAM_BACK > ...
    """
    PRIORITY_CAMS = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]
    with open(path, "r", encoding="utf-8") as f:
        val = json.load(f)
    queries = []
    for scene_token, sc in val.items():
        if not isinstance(sc, dict):
            continue
        scene_desc = sc.get("scene_description", "") or ""
        for frame_token, fb in (sc.get("key_frames") or {}).items():
            qa = fb.get("QA") or {}
            image_paths_map = fb.get("image_paths") or {}
            # Resolve up to ``multi_camera`` camera image paths in priority order.
            resolved_paths: list[str] = []
            for cam in PRIORITY_CAMS[:multi_camera]:
                rel = image_paths_map.get(cam, "")
                if not rel:
                    continue
                # Candidates for the actual file: val-style relative path,
                # train-style relative path, or basename under known roots.
                base_name = os.path.basename(rel)
                candidates = [
                    rel,                                       # as-is (relative to caller)
                    os.path.join(data_root, rel.lstrip("./")), # relative to data_root
                    os.path.join(data_root, "nuscenes", "samples", cam, base_name),
                    os.path.join(data_root, "drivelm_nus_imgs_val", "val_data", cam, base_name),
                ]
                for p in candidates:
                    if p and os.path.exists(p):
                        resolved_paths.append(p)
                        break
                if len(resolved_paths) >= multi_camera:
                    break
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
                        "qid": f"{scene_token}::{frame_token}::{cat}::{qa_item.get('Q', '')[:30]}",
                        "question": q,
                        "scene": scene_desc,
                        "image_path": resolved_paths[0] if resolved_paths else "",
                        "image_paths": resolved_paths,
                        "task": cat,
                    })
                    if n and len(queries) >= n:
                        return queries
    return queries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", help="JSONL of queries (one per line).")
    ap.add_argument(
        "--val-json",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "json", "v1_1_val_nus_q_only.json"),
        help="If --input is not given, load queries from this val JSON.",
    )
    ap.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="nuScenes root for resolving CAM_FRONT image paths.",
    )
    ap.add_argument("--output", "-o", required=True, help="Output JSONL for predictions.")
    ap.add_argument("--n", type=int, default=200, help="Max number of queries to run (0 = all).")
    ap.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to use as few-shot.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--vlm-dir", default=DEFAULT_VLM_DIR)
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--bge-model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--reranker-dir", default=DEFAULT_RERANKER_DIR)
    ap.add_argument("--hnsw-k", type=int, default=100)
    ap.add_argument("--bm25-k", type=int, default=20)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--multi-camera", type=int, default=1, choices=[1, 4, 6],
        help="Number of camera images per query (1 = CAM_FRONT only, "
             "4 = front 3 + back, 6 = full surround).",
    )
    return ap


def main() -> None:
    args = _build_argparser().parse_args()

    if args.input:
        print(f"📥 Loading queries from {args.input}")
        queries = _load_queries_from_jsonl(args.input, args.n)
    else:
        print(f"📥 Loading queries from val JSON {args.val_json}")
        print(f"   multi-camera mode: {args.multi_camera}")
        queries = _load_queries_from_val_json(
            args.val_json, args.n, args.data_root, args.multi_camera,
        )

    print(f"   {len(queries)} queries")
    if args.n:
        queries = queries[: args.n]
    print(f"   running on first {len(queries)}")

    # Init
    gen = VLMGenerator(
        vlm_dir=args.vlm_dir,
        index_dir=args.index_dir,
        bge_model_dir=args.bge_model_dir,
        reranker_dir=args.reranker_dir,
        hnsw_k=args.hnsw_k,
        bm25_k=args.bm25_k,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
        max_new_tokens=args.max_new_tokens,
    )

    out_f = open(args.output, "w", encoding="utf-8")
    n_ok = n_err = 0
    t_start = time.time()
    for i, q in enumerate(queries, start=1):
        rec = gen.run_query(q)
        if rec.get("error"):
            n_err += 1
        else:
            n_ok += 1
        # Serialize: drop the heavy retrieved field but keep a small summary
        summary = {
            "qid": rec.get("qid"),
            "question": rec.get("question"),
            "scene": rec.get("scene"),
            "image_path": rec.get("image_path"),
            "image_paths": rec.get("image_paths", []),
            "task": rec.get("task"),
            "prediction": rec.get("prediction"),
            "latency_ms": rec.get("latency_ms"),
            "n_retrieved": len(rec.get("retrieved", [])),
            "retrieved_chunk_ids": [c.get("chunk_id") for c in rec.get("retrieved", [])],
            "retrieved_qa_pairs": [
                {"q": _extract_qa(c.get("text_for_embedding", ""))[0],
                 "a": _extract_qa(c.get("text_for_embedding", ""))[1]}
                for c in rec.get("retrieved", [])
            ],
            "error": rec.get("error"),
        }
        out_f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        out_f.flush()
        if i % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  ... {i}/{len(queries)}  ok={n_ok}  err={n_err}  elapsed={elapsed:.1f}s  ({elapsed/i:.1f}s/q)")

    out_f.close()
    elapsed = time.time() - t_start
    print(f"\n✅ Done {len(queries)} queries in {elapsed:.1f}s ({elapsed/len(queries):.1f}s/q)")
    print(f"   ok={n_ok}  err={n_err}")
    print(f"💾 Saved: {args.output}")


if __name__ == "__main__":
    main()

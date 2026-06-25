"""Generate per-entity-community summaries (Microsoft GraphRAG step 3).

For each entity community we ask the LLM to describe the typical driving
scenario based on a representative sample of its member chunks (per-QA Q/A
pairs whose scene graph contains the community's entities).

Output: <out>/entity_community_summaries.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a driving-scene analyst. You are shown a cluster of related driving "
    "scene entities (e.g., 'Pedestrian at CAM_FRONT_RIGHT in r12'). The cluster "
    "groups entities that frequently co-occur in driving scenes.\n\n"
    "You will also see several Q/A pairs sampled from scenes that contain these "
    "entities. Based on the entities and Q/A pairs, write a concise paragraph "
    "(2-4 sentences, 60-120 words) describing the typical driving scenario: "
    "what does the scene look like, what agents are present, what kinds of "
    "questions get asked, and any notable patterns.\n\n"
    "Output the summary paragraph ONLY. No preamble, no labels, no bullet points."
)


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_chunk_index(jsonl_path: str) -> dict[str, dict]:
    """chunk_id → chunk row."""
    out: dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d.get("chunk_id")
            if cid:
                out[cid] = d
    return out


def build_prompt(
    comm: dict,
    sample_chunks: list[dict],
    entity_descriptions: dict[str, str] | None = None,
) -> str:
    s = comm["stats"]
    cat_str = ", ".join(f"{c} ({n})" for c, n in s["top_categories"])
    cam_str = ", ".join(f"{c} ({n})" for c, n in s["top_cameras"])
    reg_str = ", ".join(f"{r} ({n})" for r, n in s["top_regions"])

    parts = []
    parts.append(
        f"Community of {s['n_members']} entities spanning {s['n_chunks']} scenes.\n"
        f"Categories: {cat_str}\n"
        f"Cameras:   {cam_str}\n"
        f"Regions:   {reg_str}"
    )

    # Add per-entity descriptions if available — gives the LLM concrete grounding
    if entity_descriptions:
        sample_members = comm.get("members", [])[:12]
        desc_lines = []
        for m in sample_members:
            desc = entity_descriptions.get(m, "")
            if desc:
                desc_lines.append(f"  - {m}: {desc}")
        if desc_lines:
            parts.append("\nEntities (with descriptions):")
            parts.extend(desc_lines)

    parts.append(
        f"\nSample Q/A pairs from these scenes:"
    )

    for i, chunk in enumerate(sample_chunks, 1):
        parts.append(
            f"--- Sample {i} (task={chunk.get('task', '?')}) ---\n"
            f"Q: {chunk.get('question', '')[:300]}\n"
            f"A: {chunk.get('answer', '')[:200]}"
        )

    parts.append(
        "\nNow write the 2-4 sentence summary paragraph (60-120 words)."
    )
    return "\n".join(parts)


def post_process(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(Summary|Summary:|Answer:|Output:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--communities",
        default="./graphrag_community/entity_communities/entity_communities.jsonl",
    )
    p.add_argument(
        "--chunks",
        default="./graphrag_community/chunks/drivelm_train_chunks.jsonl",
    )
    p.add_argument(
        "--out",
        default="./graphrag_community/entity_communities/entity_community_summaries.jsonl",
    )
    p.add_argument("--vlm-path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--n-samples", type=int, default=8,
                   help="Number of chunks sampled per community for the prompt.")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--entity-descriptions",
        default="./graphrag_community/entity_kg/entity_descriptions.json",
        help="JSON file mapping entity_key -> natural-language description. "
             "If file missing or empty, summaries fall back to entity keys only.",
    )
    args = p.parse_args()

    random.seed(args.seed)

    print(f"[load] communities from {args.communities}")
    communities = load_jsonl(args.communities)
    print(f"[load] {len(communities)} communities")

    print(f"[load] chunk index from {args.chunks}")
    chunk_idx = load_chunk_index(args.chunks)
    print(f"[load] {len(chunk_idx)} chunks indexed")

    entity_descriptions: dict[str, str] = {}
    if args.entity_descriptions and os.path.exists(args.entity_descriptions):
        with open(args.entity_descriptions, "r", encoding="utf-8") as f:
            entity_descriptions = json.load(f)
        print(f"[load] {len(entity_descriptions)} entity descriptions")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load LLM
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    print(f"[load] LLM: {args.vlm_path}")
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

    out_records: list[dict] = []
    for comm in tqdm(communities, desc="[summarize]"):
        chunk_ids = comm.get("chunks", [])  # capped list of 50 from detect script
        # Re-load full chunk set from community_chunks.jsonl if available
        # (we only sampled 50 in detect_entity_communities; for sampling we use those)
        sample_ids = random.sample(chunk_ids, min(args.n_samples, len(chunk_ids)))
        sample_chunks = [chunk_idx[cid] for cid in sample_ids if cid in chunk_idx]
        if not sample_chunks:
            summary = "(no chunk data)"
        else:
            prompt = build_prompt(comm, sample_chunks, entity_descriptions or None)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(text, return_tensors="pt").to(model.device)
            try:
                with torch.inference_mode():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=0.1,
                    )
                gen = tok.decode(
                    out_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                summary = post_process(gen)
            except Exception as e:
                summary = f"(generation failed: {type(e).__name__}: {e})"

        out_records.append({
            "community_id": comm["community_id"],
            "n_entities": comm["stats"]["n_members"],
            "n_chunks": comm["stats"]["n_chunks"],
            "summary": summary,
            "members": comm["members"],
            "stats": comm["stats"],
        })

    out_records.sort(key=lambda r: r["community_id"])
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] wrote {args.out} ({len(out_records)} summaries)")


if __name__ == "__main__":
    main()

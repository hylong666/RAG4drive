"""Generate natural-language descriptions for each entity in the KG.

Each entity is a symbol like ``Vehicle@CAM_FRONT@r10``.  For community-summary
prompts to be readable (and for entity-aware local retrieval to interpret
matches), we attach a short 1-2 sentence description produced by Qwen2.5-7B.

Outputs:
  - <out>/entity_descriptions.json  {entity_key: description}
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re

from tqdm import tqdm


ENTITY_CATEGORIES = ["Vehicle", "Pedestrian", "Cycle", "Traffic element", "Others"]


def load_entities(entity_chunk_sets_path: str) -> list[str]:
    with open(entity_chunk_sets_path, "rb") as f:
        ents = pickle.load(f)
    return sorted(ents.keys())


def parse_entity_key(key: str) -> tuple[str, str, str]:
    parts = key.split("@")
    if len(parts) != 3:
        return (key, "", "")
    return parts[0], parts[1], parts[2]


# Human-readable region names (3x3 grid: row, col)
REGION_NAMES = {
    "r00": "upper-left", "r01": "upper-center", "r02": "upper-right",
    "r10": "middle-left", "r11": "middle-center", "r12": "middle-right",
    "r20": "lower-left", "r21": "lower-center", "r22": "lower-right",
}
CAM_NAMES = {
    "CAM_FRONT": "front-center",
    "CAM_FRONT_LEFT": "front-left",
    "CAM_FRONT_RIGHT": "front-right",
    "CAM_BACK": "rear",
    "CAM_BACK_LEFT": "rear-left",
    "CAM_BACK_RIGHT": "rear-right",
}


def entity_to_readable(key: str) -> str:
    cat, cam, reg = parse_entity_key(key)
    cam_r = CAM_NAMES.get(cam, cam)
    reg_r = REGION_NAMES.get(reg, reg)
    return f"a {cat} at the {cam_r} camera, in the {reg_r} region of the image"


SYSTEM_PROMPT = (
    "You are an autonomous-driving scene analyst. You will be given a list of "
    "entities. Each entity is a slot in the scene graph: category @ camera "
    "@ image region. Write a concise 1-2 sentence description for each "
    "entity that a downstream language model can interpret. Format your "
    "output as a JSON object mapping entity_key → description string. Do "
    "NOT include any other text, just valid JSON."
)


def make_user_prompt(batch: list[str]) -> str:
    rows = []
    for k in batch:
        rows.append(f'  "{k}": "{entity_to_readable(k)}"')
    return (
        "Generate a 1-2 sentence description for each entity. "
        "Be concrete about what kind of agent/element it likely is "
        "(e.g., \"a passenger vehicle approaching from the front-left\" "
        "or \"a static traffic sign in the rear-right region\").\n\n"
        "Entities to describe:\n" + "\n".join(rows) + "\n\n"
        "Return a JSON object. Example shape: "
        "{\"Vehicle@CAM_FRONT@r10\": \"a passenger car approaching the ego vehicle from the front...\"}"
    )


def parse_json_response(text: str) -> dict:
    """Try to extract JSON dict from LLM output, tolerant of code fences."""
    text = text.strip()
    # Strip ```json fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    snippet = text[start:end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--entity-chunk-sets",
        default="./graphrag_community/entity_kg/entity_chunk_sets.pkl",
    )
    p.add_argument(
        "--out",
        default="./graphrag_community/entity_kg/entity_descriptions.json",
    )
    p.add_argument(
        "--vlm-path",
        default="/home/hylong/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/",
    )
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--limit", type=int, default=None, help="Optional cap for debugging.")
    args = p.parse_args()

    # Resolve snapshot
    vlm_path = args.vlm_path
    if os.path.isdir(vlm_path) and not os.path.exists(os.path.join(vlm_path, "config.json")):
        children = sorted(os.listdir(vlm_path))
        if children:
            vlm_path = os.path.join(vlm_path, children[0])

    entities = load_entities(args.entity_chunk_sets)
    if args.limit:
        entities = entities[: args.limit]
    print(f"[load] {len(entities)} unique entities")

    # Load existing descriptions if resuming
    existing: dict[str, str] = {}
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[load] {len(existing)} existing descriptions (will skip)")

    pending = [e for e in entities if e not in existing or not existing[e]]
    print(f"[plan] {len(pending)} entities need descriptions")

    if not pending:
        print("[done] all entities already have descriptions")
        return

    # Load LLM
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    print(f"[load] LLM: {vlm_path}")
    tok = AutoTokenizer.from_pretrained(vlm_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        vlm_path, torch_dtype=dtype_map[args.dtype], device_map="auto", trust_remote_code=True
    )
    model.eval()

    descriptions = dict(existing)
    for i in tqdm(range(0, len(pending), args.batch_size), desc="[describe]"):
        batch = pending[i:i + args.batch_size]
        prompt = make_user_prompt(batch)
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
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=0.1,
                )
            gen = tok.decode(
                out_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
        except Exception as e:
            print(f"[warn] generation failed: {e}")
            gen = "{}"

        parsed = parse_json_response(gen)
        if parsed:
            for k, v in parsed.items():
                if isinstance(v, str) and v.strip() and k in batch:
                    descriptions[k] = v.strip()
        # Write incrementally
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

    print(f"[save] wrote {args.out} ({len(descriptions)} descriptions)")


if __name__ == "__main__":
    main()

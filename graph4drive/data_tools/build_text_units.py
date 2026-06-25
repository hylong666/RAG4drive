"""Build Microsoft GraphRAG-style text units with entity_ids + relationship_ids.

Each text unit is a chunk with explicit back-references to entities and
relationships extracted (here: derived from PMI-weighted entity KG). This
matches the Microsoft GraphRAG `text_units.jsonl` schema:

  {
    "id": "<chunk_id>",
    "text": "<chunk text>",
    "entity_ids": ["Vehicle@CAM_FRONT@r10", "Pedestrian@CAM_BACK@r11"],
    "relationship_ids": ["rel_Vehicle@CAM_FRONT@r10__Pedestrian@CAM_BACK@r11"],
    "source_doc_id": "<scene_token>",
    "n_tokens": 200   # approximate
  }

Inputs:
  --chunks-meta    chunks_metadata.pkl (from build_chunks_index.py)
  --entity-chunk-sets  entity_chunk_sets.pkl (from build_entity_kg.py)
  --entity-kg      entity_kg.gpickle (NetworkX Graph)

Output:
  --out            text_units.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re

import pandas as pd


def _rel_id(a: str, b: str) -> str:
    """Stable, sorted relationship id (alphabetical order)."""
    if a <= b:
        return f"rel_{a}__{b}"
    return f"rel_{b}__{a}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chunks-meta",
        default="./graphrag_community/chunks/chunks_metadata.pkl",
    )
    p.add_argument(
        "--entity-chunk-sets",
        default="./graphrag_community/entity_kg/entity_chunk_sets.pkl",
    )
    p.add_argument(
        "--entity-kg",
        default="./graphrag_community/entity_kg/entity_kg.gpickle",
    )
    p.add_argument(
        "--out",
        default="./graphrag_community/text_units/text_units.jsonl",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[load] chunks_meta from {args.chunks_meta}")
    chunks_meta = pd.read_pickle(args.chunks_meta)
    print(f"[load] {len(chunks_meta)} chunks")

    print(f"[load] entity_chunk_sets from {args.entity_chunk_sets}")
    with open(args.entity_chunk_sets, "rb") as f:
        ent_to_chunks = pickle.load(f)
    print(f"[load] {len(ent_to_chunks)} entities")

    # Build reverse index: chunk_id → [entity, ...]
    chunk_to_ents: dict[str, list[str]] = {}
    for ent, cids in ent_to_chunks.items():
        for cid in cids:
            chunk_to_ents.setdefault(cid, []).append(ent)
    print(f"[plan] {len(chunk_to_ents)} chunks have at least 1 entity")

    # Load entity KG to get edges → generate relationship_ids
    print(f"[load] entity_kg from {args.entity_kg}")
    import networkx as nx
    with open(args.entity_kg, "rb") as f:
        G = pickle.load(f)
    edge_set = set()
    for u, v in G.edges():
        edge_set.add(_rel_id(u, v))
    print(f"[plan] {len(edge_set)} unique edges in entity KG")

    n_total = 0
    n_with_ents = 0
    n_with_rels = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for _, row in chunks_meta.iterrows():
            cid = row["chunk_id"]
            ents = sorted(set(chunk_to_ents.get(cid, [])))
            # Generate relationship_ids from entity pairs that have an edge
            rels: set[str] = set()
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    rid = _rel_id(ents[i], ents[j])
                    if rid in edge_set:
                        rels.add(rid)

            tu = {
                "id": cid,
                "text": row["text"],
                "entity_ids": ents,
                "relationship_ids": sorted(rels),
                "source_doc_id": row.get("scene_token", ""),
                "n_tokens": len(row["text"].split()),
            }
            fout.write(json.dumps(tu, ensure_ascii=False) + "\n")
            n_total += 1
            if ents:
                n_with_ents += 1
            if rels:
                n_with_rels += 1

    print(f"[save] wrote {args.out}")
    print(f"[stats] total text units: {n_total}")
    print(f"[stats] with ≥1 entity: {n_with_ents} ({n_with_ents / n_total * 100:.1f}%)")
    print(f"[stats] with ≥1 relationship: {n_with_rels} ({n_with_rels / n_total * 100:.1f}%)")


if __name__ == "__main__":
    main()
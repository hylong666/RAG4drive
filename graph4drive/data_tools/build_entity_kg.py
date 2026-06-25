"""Build an entity-level knowledge graph from per-QA chunks.

Aligned with Microsoft GraphRAG:
  Chunks (per-QA)  →  Entities  →  Co-occurrence KG  →  Communities

  Entity = (Category, Camera, RegionBin)
       where RegionBin = discretization of (x, y) into a 3x3 grid (front/back/left/right)

  Edge (e1, e2) weight = number of chunks in which both entities co-occur.
  This is the standard "entity co-occurrence graph" that GraphRAG builds from
  text-extracted entities; here we use scene-graph entities directly.

Output:
  - <out>/entity_kg.gpickle        NetworkX Graph (nodes = entity types)
  - <out>/entity_index.parquet     entity → {category, camera, region, n_chunks}
  - <out>/entity_chunk_sets.pkl    entity → set[chunk_id] (for downstream use)
  - <out>/entity_kg_stats.json     n_nodes, n_edges, density, top entities
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


NODE_RE = re.compile(r"<([^,>]+),([^,>]+),([\d.\-]+),([\d.\-]+)>\(([^)]+)\)")


def parse_objects(s: str) -> list[tuple[str, str, float, float, str]]:
    if not s:
        return []
    out = []
    for m in NODE_RE.finditer(s):
        try:
            obj_id, cam, x, y, cat = m.groups()
            out.append((obj_id, cam, float(x), float(y), cat.strip()))
        except (ValueError, AttributeError):
            continue
    return out


def region_bin(x: float, y: float, img_w: float = 1600.0, img_h: float = 900.0) -> str:
    """Discretize (x, y) into a 3x3 grid label."""
    col = min(int(x / img_w * 3), 2)
    row = min(int(y / img_h * 3), 2)
    # layout: r00=top-left, r22=bottom-right (image y grows downward)
    return f"r{row}{col}"


def entity_key(cat: str, cam: str, region: str) -> str:
    return f"{cat}@{cam}@{region}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chunks",
        default="./graphrag_community/chunks/drivelm_train_chunks.jsonl",
    )
    p.add_argument(
        "--out-dir",
        default="./graphrag_community/entity_kg",
    )
    p.add_argument(
        "--edge-weight", choices=["count", "pmi"], default="pmi",
        help="Edge weighting: raw co-occurrence count, or PMI (pointwise mutual info).",
    )
    p.add_argument(
        "--min-cooccurrence", type=int, default=5,
        help="Drop edges with co-occurrence count < this (helps PMI quality).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Walk chunks, collect per-chunk entity sets
    print(f"[load] reading {args.chunks}")
    chunk_entities: dict[str, set[str]] = {}
    chunk_raw_objects: dict[str, list[tuple]] = {}
    n_chunks = 0

    for line in tqdm(open(args.chunks, encoding="utf-8"), desc="[chunks]"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        chunk_id = d.get("chunk_id") or f"unk.{n_chunks}"
        objs = parse_objects(d.get("graph_nodes", ""))
        if not objs:
            continue
        ents = set()
        for obj_id, cam, x, y, cat in objs:
            r = region_bin(x, y)
            ents.add(entity_key(cat, cam, r))
        chunk_entities[chunk_id] = ents
        chunk_raw_objects[chunk_id] = objs
        n_chunks += 1

    print(f"[load] {n_chunks} chunks with entities")

    # 2) Aggregate entity → set of chunk_ids (for downstream retrieval)
    entity_chunk_sets: dict[str, set[str]] = defaultdict(set)
    for chunk_id, ents in chunk_entities.items():
        for e in ents:
            entity_chunk_sets[e].add(chunk_id)

    print(f"[ents] {len(entity_chunk_sets)} unique entity types")

    # 3) Build co-occurrence graph: edge weight = |chunks(e1) ∩ chunks(e2)|
    #    To avoid O(N^2), iterate over chunk sets and bump co-occurrence pairs.
    pair_counts: Counter = Counter()
    for ents in tqdm(chunk_entities.values(), desc="[co-occur]"):
        ents_list = sorted(ents)
        if len(ents_list) < 2:
            continue
        for i in range(len(ents_list)):
            for j in range(i + 1, len(ents_list)):
                pair_counts[(ents_list[i], ents_list[j])] += 1

    print(f"[pairs] {len(pair_counts)} non-zero co-occurrence pairs")

    # Compute per-entity marginal counts
    n_chunks_total = max(n_chunks, 1)
    entity_marginal = {e: len(c) for e, c in entity_chunk_sets.items()}

    # Filter and reweight
    kept_pairs: Counter = Counter()
    dropped_low = 0
    for (e1, e2), w in pair_counts.items():
        if w < args.min_cooccurrence:
            dropped_low += 1
            continue
        if args.edge_weight == "pmi":
            # PMI = log( P(e1,e2) / (P(e1)*P(e2)) )
            import math
            p12 = w / n_chunks_total
            p1 = entity_marginal[e1] / n_chunks_total
            p2 = entity_marginal[e2] / n_chunks_total
            if p1 <= 0 or p2 <= 0:
                continue
            pmi = math.log(p12 / (p1 * p2) + 1e-12)
            if pmi <= 0:
                # No positive association → drop
                continue
            kept_pairs[(e1, e2)] = pmi
        else:
            kept_pairs[(e1, e2)] = int(w)
    print(
        f"[weight] dropped {dropped_low} low-count pairs; "
        f"kept {len(kept_pairs)} after {args.edge_weight} weighting"
    )

    G = nx.Graph()
    for ent, chunks in entity_chunk_sets.items():
        parts = ent.split("@")
        cat, cam, region = parts[0], parts[1], parts[2]
        G.add_node(
            ent,
            category=cat,
            camera=cam,
            region=region,
            n_chunks=len(chunks),
        )
    for (e1, e2), w in kept_pairs.items():
        G.add_edge(e1, e2, weight=float(w))

    # Drop nodes that never co-occur with anything (degree 0)
    isolates = list(nx.isolates(G))
    print(
        f"[graph] {G.number_of_nodes()} entities, {G.number_of_edges()} co-occurrence edges, "
        f"{len(isolates)} isolated"
    )

    out_gpickle = os.path.join(args.out_dir, "entity_kg.gpickle")
    with open(out_gpickle, "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"[save] wrote {out_gpickle}")

    # Entity index
    ent_rows = []
    for ent, data in G.nodes(data=True):
        ent_rows.append(
            {
                "entity": ent,
                "category": data.get("category", ""),
                "camera": data.get("camera", ""),
                "region": data.get("region", ""),
                "n_chunks": data.get("n_chunks", 0),
                "degree": int(G.degree(ent)) if ent in G else 0,
            }
        )
    ent_df = pd.DataFrame(ent_rows).sort_values("n_chunks", ascending=False)
    out_idx = os.path.join(args.out_dir, "entity_index.parquet")
    ent_df.to_parquet(out_idx, index=False)
    print(f"[save] wrote {out_idx}")

    # Entity → set of chunk_ids
    out_sets = os.path.join(args.out_dir, "entity_chunk_sets.pkl")
    with open(out_sets, "wb") as f:
        pickle.dump({k: sorted(v) for k, v in entity_chunk_sets.items()}, f)
    print(f"[save] wrote {out_sets}")

    # Stats
    degrees = [d for _, d in G.degree()]
    cat_counter = Counter(d.get("category", "") for _, d in G.nodes(data=True))
    cam_counter = Counter(d.get("camera", "") for _, d in G.nodes(data=True))
    region_counter = Counter(d.get("region", "") for _, d in G.nodes(data=True))
    stats = {
        "n_input_chunks": n_chunks,
        "n_entities": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_isolated": len(isolates),
        "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "degree_median": float(np.median(degrees)) if degrees else 0.0,
        "degree_max": int(np.max(degrees)) if degrees else 0,
        "top_categories": cat_counter.most_common(10),
        "top_cameras": cam_counter.most_common(10),
        "top_regions": region_counter.most_common(10),
    }
    out_stats = os.path.join(args.out_dir, "entity_kg_stats.json")
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[save] wrote {out_stats}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

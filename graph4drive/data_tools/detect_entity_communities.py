"""Run Leiden community detection on the entity-level knowledge graph.

Reads entity_kg.gpickle (NetworkX) and produces:
  - <out>/entity_communities.jsonl   one row per community with members + stats
  - <out>/entity_community_assignments.parquet   entity → community_id
  - <out>/entity_community_chunks.jsonl        community_id → list of chunk_ids
                                            (entities → their chunks)

Communities of entities (e.g., a cluster of "Vehicle@CAM_FRONT@r10 + Pedestrian
@CAM_FRONT@r10 + Traffic element@CAM_FRONT@r10") represent a recurring scene
type.  Community summaries then describe "what this kind of scene looks like".
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph",
        default="./graphrag_community/entity_kg/entity_kg.gpickle",
    )
    p.add_argument(
        "--entity-chunk-sets",
        default="./graphrag_community/entity_kg/entity_chunk_sets.pkl",
        help="Pickle mapping entity → list[chunk_id].",
    )
    p.add_argument(
        "--out-dir",
        default="./graphrag_community/entity_communities",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] reading {args.graph}")
    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    print(f"[load] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"[load] reading {args.entity_chunk_sets}")
    with open(args.entity_chunk_sets, "rb") as f:
        ent_to_chunks = pickle.load(f)

    nodes = list(G.nodes())
    node_to_idx = {e: i for i, e in enumerate(nodes)}
    edges = []
    weights = []
    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(float(data.get("weight", 1.0)))

    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.es["weight"] = weights

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[leiden] ModularityVertexPartition …")
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        seed=args.seed,
        n_iterations=10,
    )
    membership = partition.membership
    n_communities = len(set(membership))
    modularity = g.modularity(membership, weights="weight")
    print(f"[leiden] {n_communities} communities, modularity = {modularity:.4f}")

    # Aggregate per-community stats
    comm_to_ents: dict[int, list[str]] = defaultdict(list)
    for ent, comm in zip(nodes, membership):
        comm_to_ents[comm].append(ent)

    comm_to_chunks: dict[int, set[str]] = defaultdict(set)
    for comm, ents in comm_to_ents.items():
        for e in ents:
            comm_to_chunks[comm].update(ent_to_chunks.get(e, []))

    out_comm = os.path.join(args.out_dir, "entity_communities.jsonl")
    with open(out_comm, "w", encoding="utf-8") as f:
        for cid in sorted(comm_to_ents.keys()):
            ents = sorted(comm_to_ents[cid])
            chunks = sorted(comm_to_chunks[cid])
            cat_counter = Counter(e.split("@")[0] for e in ents)
            cam_counter = Counter(e.split("@")[1] for e in ents)
            reg_counter = Counter(e.split("@")[2] for e in ents)
            rec = {
                "community_id": cid,
                "members": ents,
                "stats": {
                    "n_members": len(ents),
                    "n_chunks": len(chunks),
                    "top_categories": cat_counter.most_common(5),
                    "top_cameras": cam_counter.most_common(5),
                    "top_regions": reg_counter.most_common(5),
                    "internal_weight": sum(
                        G[u][v].get("weight", 1) for u, v in G.edges() if membership[node_to_idx[u]] == cid and membership[node_to_idx[v]] == cid
                    ),
                },
                "chunks": chunks[:50],  # cap for size; full set is in chunk map
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] wrote {out_comm}")

    # Community → chunks full mapping (one file)
    out_map = os.path.join(args.out_dir, "entity_community_chunks.jsonl")
    with open(out_map, "w", encoding="utf-8") as f:
        for cid in sorted(comm_to_chunks.keys()):
            f.write(json.dumps(
                {"community_id": cid, "chunk_ids": sorted(comm_to_chunks[cid])},
                ensure_ascii=False,
            ) + "\n")
    print(f"[save] wrote {out_map}")

    # Per-entity assignment
    assign = pd.DataFrame(
        [{"entity": e, "community_id": membership[i]} for i, e in enumerate(nodes)]
    )
    out_assign = os.path.join(args.out_dir, "entity_community_assignments.parquet")
    assign.to_parquet(out_assign, index=False)
    print(f"[save] wrote {out_assign}")

    # Summary
    sizes = sorted([len(v) for v in comm_to_ents.values()], reverse=True)
    print(f"[stats] community size distribution (top 10): {sizes[:10]}")
    print(f"[stats] total communities: {n_communities}, modularity: {modularity:.4f}")

    with open(os.path.join(args.out_dir, "entity_leiden_stats.json"), "w") as f:
        json.dump({
            "n_communities": n_communities,
            "modularity": modularity,
            "size_distribution": sizes,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
        }, f, indent=2)


if __name__ == "__main__":
    main()

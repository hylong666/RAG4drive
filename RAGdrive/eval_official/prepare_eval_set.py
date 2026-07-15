"""Build a DriveLM official-format evaluation set from the full train JSON.

Pipeline (steps 1-2 of the eval flow; no GPU needed):

    v1_1_train_nus.json
        --(sample N scenes)-->      train_Nscene.json
        --(challenge/extract_data)--> my_test.json      (select graph/MCQ/etc)
        --(challenge/convert_data)--> my_test_eval.json (open->MCQ + fill tags)

This reuses the official ``extract_data`` / ``loop_test`` functions verbatim
(their source is exec'd), so the produced tags and question selection match
the official challenge exactly. ``convert_data``'s dead ``import numpy`` line
is stripped so it runs without numpy.

Usage
-----
::

    python prepare_eval_set.py \\
        --train ../data/json/v1_1_train_nus.json \\
        --n-scenes 5 \\
        --challenge-dir ../../../DriveLM/challenge \\
        --out-dir .
"""

from __future__ import annotations

import argparse
import json
import os
import random


def _load_func_source(path: str, drop_numpy: bool = False) -> str:
    src = open(path, "r", encoding="utf-8").read()
    # Silence debug prints in extract_data.
    src = src.replace("print(classes)", "pass").replace("print(locations)", "pass")
    if drop_numpy:
        src = src.replace("import numpy as np", "pass")
    # Keep only the function defs, drop the __main__ block.
    return src.split("if __name__")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train", required=True, help="Full v1_1_train_nus.json.")
    ap.add_argument("--n-scenes", type=int, default=5, help="Number of scenes to sample (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--challenge-dir", required=True,
                    help="Path to DriveLM/challenge (for extract_data.py & convert_data.py).")
    ap.add_argument("--out-dir", default=".", help="Where to write intermediates + my_test_eval.json.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # Step 0: sample scenes.
    with open(args.train, "r", encoding="utf-8") as f:
        full = json.load(f)
    keys = list(full.keys())
    if args.n_scenes and args.n_scenes < len(keys):
        keys = keys[: args.n_scenes]  # deterministic first-N (change to random.sample if desired)
    sampled = {k: full[k] for k in keys}
    sampled_path = os.path.join(args.out_dir, f"train_{len(keys)}scene.json")
    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False)
    n_qa = sum(len(fr["QA"][t]) for s in sampled.values()
               for fr in s["key_frames"].values() for t in fr["QA"])
    print(f"[sample] {len(keys)} scenes, {n_qa} raw QA -> {sampled_path}")

    # Step 1: extract_data (official selection).
    extract_src = _load_func_source(os.path.join(args.challenge_dir, "extract_data.py"))
    ns1: dict = {}
    exec(extract_src, ns1)
    my_test = os.path.join(args.out_dir, "my_test.json")
    ns1["extract_data"](sampled_path, my_test)
    print(f"[extract] -> {my_test}")

    # Step 2: convert_data (open -> MCQ, fill tags).
    convert_src = _load_func_source(os.path.join(args.challenge_dir, "convert_data.py"), drop_numpy=True)
    ns2: dict = {}
    exec(convert_src, ns2)
    my_test_eval = os.path.join(args.out_dir, "my_test_eval.json")
    ns2["loop_test"](my_test, my_test_eval)

    # Report.
    d = json.load(open(my_test_eval, "r", encoding="utf-8"))
    from collections import Counter
    c: Counter = Counter()
    for s in d.values():
        for fr in s["key_frames"].values():
            for t in fr["QA"]:
                c[t] += len(fr["QA"][t])
    print(f"[convert] -> {my_test_eval}")
    print(f"[done] eval set ready: {dict(c)} total={sum(c.values())}")


if __name__ == "__main__":
    main()

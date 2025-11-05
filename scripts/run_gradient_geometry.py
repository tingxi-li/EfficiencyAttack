#!/usr/bin/env python3
"""
Launch gradient-geometry ablation runs for randomly selected query combos.

Each selected configuration is executed via main.py with compute_grad_metrics enabled,
so the pipeline records cosine similarity, sign agreement, and gradient norms at every
iteration.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple


def _load_json(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _unique_query_pairs(pairs) -> List[Tuple[int, int]]:
    unique = []
    seen = set()
    for pair in pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        tup = (int(pair[0]), int(pair[1]))
        if tup not in seen:
            seen.add(tup)
            unique.append(tup)
    return unique


def _write_temp_config(config: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    try:
        json.dump(config, tmp, indent=4)
    finally:
        tmp.close()
    return Path(tmp.name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run gradient geometry ablation across sampled query combos.")
    parser.add_argument("--config_file", type=str, default="config/loss_sweep.json", help="Base configuration JSON.")
    parser.add_argument("--pipeline_id", type=int, default=3, help="Pipeline id to pass to main.py.")
    parser.add_argument("--num_combos", type=int, default=3, help="Number of query combos to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling query combos.")
    parser.add_argument("--ngpus", type=int, default=None, help="Override GPU count for parallel execution.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned runs without executing them.")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    base_cfg_path = project_root / args.config_file
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    base_cfg = _load_json(base_cfg_path)
    available_pairs = _unique_query_pairs(base_cfg.get("loss_sweep_queries")) or [
        (int(base_cfg.get("num_queries_1", 5)), int(base_cfg.get("num_queries_2", 5)))
    ]

    random.seed(args.seed)
    sample_count = min(int(args.num_combos), len(available_pairs))
    selected = random.sample(available_pairs, sample_count)

    print(f"[ablation] sampling {sample_count} / {len(available_pairs)} query combos from {base_cfg_path}")
    if not selected:
        print("No query combos available to run.")
        return 0

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    for idx, (q1, q2) in enumerate(selected, start=1):
        cfg = json.loads(json.dumps(base_cfg))  # deep copy
        cfg["num_queries_1"] = int(q1)
        cfg["num_queries_2"] = int(q2)
        cfg["compute_grad_metrics"] = True
        cfg.setdefault("output_log", True)
        cfg.setdefault("parallel", base_cfg.get("parallel", False))
        cfg.setdefault("log_dir", base_cfg.get("log_dir", "./log"))
        if args.ngpus is not None:
            cfg["ngpus"] = int(args.ngpus)
        cfg.pop("loss_sweep_queries", None)
        cfg.pop("loss_sweep_weights", None)
        cfg["_grad_geometry"] = {
            "index": idx,
            "timestamp": timestamp,
            "queries": [int(q1), int(q2)],
            "seed": args.seed,
            "source_list_size": len(available_pairs),
        }

        tmp_cfg_path = _write_temp_config(cfg)
        cmd = [
            sys.executable,
            "main.py",
            "--config_file",
            str(tmp_cfg_path),
            "--pipeline_id",
            str(args.pipeline_id),
        ]
        print(f"[run ] ({idx}/{sample_count}) queries=({q1},{q2}) cfg={tmp_cfg_path}")
        if args.dry_run:
            tmp_cfg_path.unlink(missing_ok=True)
            continue

        try:
            subprocess.run(cmd, cwd=project_root, check=True)
        finally:
            tmp_cfg_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Ensure project root (parent of this scripts directory) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_zoo.zoo import Pipeline_dict


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def list_log_files(log_dir: Path):
    return {p for p in log_dir.glob("*.json") if not p.name.endswith("_configs.json")}


def metric_from_log(log_path: Path) -> float:
    """Return a scalar to minimize. Default: average final obj_count_2 across datapoints."""
    data = load_json(log_path)
    if not isinstance(data, dict) or not data:
        return float("inf")
    finals = []
    for _, series in data.items():
        if not series:
            continue
        last = series[-1]
        val = last.get("obj_count_2")
        if val is None:
            val = last.get("total_loss")
        if val is None:
            continue
        try:
            finals.append(float(val))
        except Exception:
            continue
    if not finals:
        return float("inf")
    return float(np.mean(finals))


def run_pipeline_once(cfg: dict, pipeline_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_cls = Pipeline_dict[pipeline_id]
    pipeline = pipeline_cls(cfg, device)
    dataset = pipeline.load_dataset()
    pipeline.load_model()
    pipeline.run_attack(dataset)


def grid_values(precision: float) -> List[float]:
    if precision <= 0 or precision > 1:
        raise ValueError("precision must be in (0,1]. e.g., 0.2 -> 0,0.2,...,1.0")
    m = int(round(1.0 / precision))
    # Use integer grid to avoid FP drift
    return [i / m for i in range(m + 1)]


def enum_weight_tuples(num_weights: int, precision: float) -> List[Tuple[float, ...]]:
    if num_weights < 2:
        raise ValueError("num_weights must be >= 2")
    vals = grid_values(precision)
    m = int(round(1.0 / precision))

    if num_weights == 2:
        combos = []
        for i in range(m + 1):
            a = i / m
            b = 1.0 - a
            combos.append((a, b))
            combos.append((b, a))
        # deduplicate while preserving order
        seen = set()
        uniq = []
        for t in combos:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    # General case: ordered K-tuples of multiples of 1/m that sum to 1
    # Enumerate integer compositions of m into K parts.
    results: List[Tuple[float, ...]] = []

    def rec(prefix: List[int], remaining: int, k_left: int):
        if k_left == 1:
            prefix.append(remaining)
            results.append(tuple([x / m for x in prefix]))
            prefix.pop()
            return
        for x in range(remaining + 1):
            prefix.append(x)
            rec(prefix, remaining - x, k_left - 1)
            prefix.pop()

    rec([], m, num_weights)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Weight search over cls_loss_weight_i for selected pipelines.")
    p.add_argument("pipelines", type=str, help="Comma-separated pipeline IDs, e.g. '3,6'")
    p.add_argument("--precision", type=float, default=0.1, help="Grid precision in (0,1], e.g. 0.2 -> 0,0.2,...,1.0")
    p.add_argument("--num_weights", type=int, default=2, help="Number of weights (>=2). For 2, runs all ordered pairs (a,1-a) and (1-a,a)")
    p.add_argument("--base_config", type=str, default="config/test-1k.json", help="Base config JSON")
    p.add_argument("--num_iterations", type=int, default=None, help="Override iterations for faster sweep")
    p.add_argument("--test_size", type=int, default=None, help="Override dataset size for faster sweep")
    p.add_argument("--log_dir", type=str, default="logs/weight_search", help="Logs directory to monitor and parse")
    p.add_argument("--sleep_after", type=float, default=2.0, help="Seconds to wait after each run for logs to flush")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        pipeline_ids = [int(x) for x in args.pipelines.split(",") if x.strip() != ""]
    except Exception as exc:
        print(f"Failed to parse pipelines list: {exc}", file=sys.stderr)
        sys.exit(1)
    if not pipeline_ids:
        print("No pipeline IDs provided.", file=sys.stderr)
        sys.exit(1)

    # Current pipelines support two classification loss weights.
    if args.num_weights != 2:
        print("Only num_weights=2 is supported by current pipelines.", file=sys.stderr)
        sys.exit(1)

    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.exists():
        print(f"Base config not found: {base_cfg_path}", file=sys.stderr)
        sys.exit(1)

    base_cfg = load_json(base_cfg_path)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    combos = enum_weight_tuples(args.num_weights, args.precision)

    for pid in pipeline_ids:
        print(f"=== Pipeline {pid}: {len(combos)} combos at precision {args.precision} ===")
        for w_tuple in combos:
            w1, w2 = float(w_tuple[0]), float(w_tuple[1])
            cfg = dict(base_cfg)
            cfg["cls_loss_weight_1"] = w1
            cfg["cls_loss_weight_2"] = w2
            cfg["output_log"] = True
            cfg["parallel"] = False
            if args.num_iterations is not None:
                cfg["num_iterations"] = int(args.num_iterations)
            if args.test_size is not None:
                cfg["test_size"] = int(args.test_size)

            before = list_log_files(log_dir)
            print(f"[sweep] pid={pid} w1={w1:.6f} w2={w2:.6f}")
            try:
                run_pipeline_once(cfg, pid)
            except Exception as exc:
                print(f"Run failed for pid={pid} w1={w1:.6f} w2={w2:.6f}: {exc}", file=sys.stderr)
                continue

            time.sleep(max(0.0, float(args.sleep_after)))
            after = list_log_files(log_dir)
            new_logs = sorted(after - before, key=lambda p: p.stat().st_mtime)
            if not new_logs:
                print(f"No new logs detected for pid={pid} w1={w1:.6f} w2={w2:.6f}", file=sys.stderr)
                continue
            latest_log = new_logs[-1]
            metric = metric_from_log(latest_log)
            print(f"[metric] pid={pid} w1={w1:.6f} w2={w2:.6f} -> {metric:.6f} ({latest_log.name})")


if __name__ == "__main__":
    main()

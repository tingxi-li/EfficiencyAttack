#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def now_seconds() -> float:
    return time.time()


def list_log_files(log_dir: Path):
    return {p for p in log_dir.glob("*.json") if not p.name.endswith("_configs.json")}


def parse_args():
    p = argparse.ArgumentParser(description="Grid search for loss term weights (cls_loss_weight_1/2)")
    p.add_argument("--base_config", type=str, default="config/test-20k.json", help="Base config JSON")
    p.add_argument("--pipeline_id", type=int, default=3, help="Pipeline ID (see pipeline_zoo/zoo.py)")
    p.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0", help="Comma list of alpha where w1=alpha, w2=1-alpha")
    p.add_argument("--num_iterations", type=int, default=None, help="Override iterations for faster sweep")
    p.add_argument("--test_size", type=int, default=None, help="Override dataset size for faster sweep")
    p.add_argument("--log_dir", type=str, default="logs", help="Logs directory to monitor and parse")
    p.add_argument("--out_csv", type=str, default="output/weight_sweep/results.csv", help="CSV to write sweep results")
    p.add_argument("--sleep_after", type=float, default=2.0, help="Seconds to wait after each run for logs to flush")
    return p.parse_args()


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
        # fallbacks if key missing
        val = last.get("obj_count_2", None)
        if val is None:
            # try total_loss if obj_count_2 absent
            val = last.get("total_loss", None)
        if val is None:
            continue
        try:
            finals.append(float(val))
        except Exception:
            continue
    if not finals:
        return float("inf")
    return float(np.mean(finals))


def main():
    args = parse_args()
    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.exists():
        print(f"Base config not found: {base_cfg_path}", file=sys.stderr)
        sys.exit(1)

    base_cfg = load_json(base_cfg_path)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip() != ""]
    results = []

    # Run each alpha sequentially; for parallel use, spawn in your own scheduler/cluster.
    for alpha in alphas:
        w1 = float(alpha)
        w2 = float(1.0 - alpha)
        cfg = dict(base_cfg)
        cfg["cls_loss_weight_1"] = w1
        cfg["cls_loss_weight_2"] = w2
        # ensure logs are written for parsing
        cfg["output_log"] = True
        # avoid parallel runs here to keep attribution simple
        cfg["parallel"] = False
        if args.num_iterations is not None:
            cfg["num_iterations"] = int(args.num_iterations)
        if args.test_size is not None:
            cfg["test_size"] = int(args.test_size)

        tmp_cfg_path = Path(f"config/_sweep_tmp_alpha_{alpha:.3f}.json")
        save_json(tmp_cfg_path, cfg)

        before = list_log_files(log_dir)
        print(f"[sweep] Running alpha={alpha:.3f} -> w1={w1:.3f}, w2={w2:.3f}")
        cmd = [sys.executable, "main.py", "--config_file", str(tmp_cfg_path), "--pipeline_id", str(args.pipeline_id)]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Run failed for alpha={alpha:.3f} (exit {ret.returncode}); skipping.", file=sys.stderr)
            results.append((alpha, float("inf"), "<run_failed>"))
            continue

        time.sleep(max(0.0, float(args.sleep_after)))
        after = list_log_files(log_dir)
        new_logs = sorted(after - before, key=lambda p: p.stat().st_mtime)
        if not new_logs:
            print(f"No new logs detected for alpha={alpha:.3f}; skipping.", file=sys.stderr)
            results.append((alpha, float("inf"), "<no_log>"))
            continue

        latest_log = new_logs[-1]
        metric = metric_from_log(latest_log)
        print(f"[sweep] alpha={alpha:.3f} metric={metric:.4f} log={latest_log.name}")
        results.append((alpha, metric, latest_log.name))

    # write CSV summary
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("alpha,w1,w2,metric,logfile\n")
        for alpha, metric, logname in results:
            f.write(f"{alpha:.6f},{alpha:.6f},{1.0-alpha:.6f},{metric:.6f},{logname}\n")
    print(f"Results written to {out_csv}")

    # print best
    finite = [(a, m, ln) for a, m, ln in results if np.isfinite(m)]
    if finite:
        best_alpha, best_metric, best_log = min(finite, key=lambda x: x[1])
        print(f"Best alpha={best_alpha:.3f} (w1={best_alpha:.3f}, w2={1.0-best_alpha:.3f}) -> metric={best_metric:.4f}")
        print(f"Best log: {best_log}")


if __name__ == "__main__":
    main()


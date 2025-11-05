#!/usr/bin/env python3
"""
Iterate over loss sweep configurations and launch `main.py` for each combo so
the existing multiprocessing logic can handle data parallel execution.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pipeline_zoo.dual_object_detection_sweep import (
    _expand_sweep_configs,
    _build_combo_tag,
)


def _load_config(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _write_temp_config(cfg: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    try:
        json.dump(cfg, tmp, indent=4)
    finally:
        tmp.close()
    return Path(tmp.name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Launch loss sweep via main.py for data parallel execution.")
    parser.add_argument("--config_file", type=str, default="config/loss_sweep.json", help="Base config JSON.")
    parser.add_argument("--pipeline_id", type=int, default=3, help="Pipeline id to pass to main.py.")
    parser.add_argument("--ngpus", type=int, default=None, help="Override number of GPUs for parallel runs.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned runs without executing them.")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    base_cfg_path = project_root / args.config_file
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    base_cfg = _load_config(base_cfg_path)
    entries = _expand_sweep_configs(base_cfg)
    total = len(entries)
    if total == 0:
        print("No sweep entries found.")
        return 0

    print(f"[sweep] {total} combinations loaded from {base_cfg_path}")

    for entry in entries:
        cfg = entry["config"]
        cfg.setdefault("parallel", base_cfg.get("parallel", False))
        cfg.setdefault("output_log", True)

        log_dir = base_cfg.get("log_dir")
        if log_dir is not None:
            cfg["log_dir"] = log_dir

        if args.ngpus is not None:
            cfg["ngpus"] = int(args.ngpus)
        elif "ngpus" in cfg and not cfg.get("parallel", False):
            cfg.pop("ngpus", None)

        tag = _build_combo_tag(entry["index"], entry["queries"], entry["weights"])
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        combo_meta = {
            "index": entry["index"],
            "total": total,
            "queries": entry["queries"],
            "weights": entry["weights"],
            "tag": tag,
        }
        cfg["_sweep_combo"] = combo_meta

        tmp_cfg_path = _write_temp_config(cfg)
        cmd = [sys.executable, "main.py", "--config_file", str(tmp_cfg_path), "--pipeline_id", str(args.pipeline_id)]
        print(f"[run ] ({entry['index']}/{total}) tag={tag} cfg={tmp_cfg_path}")
        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, cwd=project_root, check=True)
        finally:
            tmp_cfg_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Aggregate and visualize gradient geometry metrics recorded in dual-object detection logs.

Example:
    python scripts/visualize_grad_metrics.py \
        --logs log/log_dual_object_detection_20251104-235214.json \
               log/log_dual_object_detection_20251104-233954.json
"""
import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = ["cosine", "sign_agreement", "norm_1", "norm_2", "norm_ratio"]


@dataclass
class Series:
    iterations: List[int]
    mean: List[float]
    std: List[float]
    count: List[int]


def _load_json(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def extract_grad_metrics(log_path: Path) -> Dict[str, Series]:
    raw_data = _load_json(log_path)
    per_metric: Dict[str, Dict[int, List[float]]] = {key: {} for key in METRIC_KEYS}

    for _, entries in raw_data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            grad_metrics = entry.get("grad_metrics")
            if not grad_metrics:
                continue
            iteration = int(entry.get("iteration", 0))
            for metric in METRIC_KEYS:
                value = grad_metrics.get(metric)
                if value is None or (isinstance(value, float) and math.isinf(value)):
                    continue
                per_metric[metric].setdefault(iteration, []).append(float(value))

    series_data: Dict[str, Series] = {}
    for metric, values in per_metric.items():
        if not values:
            continue
        iterations = sorted(values.keys())
        means = []
        stds = []
        counts = []
        for itr in iterations:
            data = values[itr]
            means.append(float(sum(data) / len(data)))
            stds.append(float(statistics.pstdev(data)) if len(data) > 1 else 0.0)
            counts.append(len(data))
        series_data[metric] = Series(iterations=iterations, mean=means, std=stds, count=counts)
    return series_data


def make_plot(series_per_log: List[Tuple[str, Dict[str, Series]]], output_path: Path) -> None:
    if not series_per_log:
        print("No gradient metric data available to plot.")
        return

    n_metrics = len(METRIC_KEYS)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4 * rows + 1), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for metric_idx, metric in enumerate(METRIC_KEYS):
        ax = axes[metric_idx // cols][metric_idx % cols]
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Iteration")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axis("on")

        y_label = {
            "cosine": "cos(g1, g2)",
            "sign_agreement": "fraction",
            "norm_1": "||g1||",
            "norm_2": "||g2||",
            "norm_ratio": "||g1|| / ||g2||",
        }.get(metric, metric)
        ax.set_ylabel(y_label)

        for label, series_dict in series_per_log:
            series = series_dict.get(metric)
            if not series:
                continue
            iterations = series.iterations
            mean = series.mean
            std = series.std
            counts = series.count
            ax.plot(iterations, mean, marker="o", label=f"{label} (n≈{np.mean(counts):.1f})")
            std = np.asarray(std)
            mean_arr = np.asarray(mean)
            if std.any():
                ax.fill_between(iterations, mean_arr - std, mean_arr + std, alpha=0.2)

        ax.legend(fontsize=8)

    for metric_idx in range(len(METRIC_KEYS), rows * cols):
        axes[metric_idx // cols][metric_idx % cols].axis("off")

    fig.suptitle("Gradient Geometry Metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[viz ] saved to {output_path}")


def summarize(series_per_log: List[Tuple[str, Dict[str, Series]]]) -> None:
    if not series_per_log:
        print("No gradient metric data collected.")
        return
    for label, series_dict in series_per_log:
        print(f"\n=== {label} ===")
        for metric, series in series_dict.items():
            if not series.mean:
                continue
            final_mean = series.mean[-1]
            final_std = series.std[-1] if series.std else 0.0
            print(
                f"{metric:>15}: final={final_mean:>8.4f} (std={final_std:>6.4f}) "
                f"len={len(series.iterations)} samples≈{np.mean(series.count):.1f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize gradient metrics from pipeline logs.")
    parser.add_argument(
        "--logs",
        type=str,
        nargs="+",
        required=True,
        help="List of log JSON files containing grad_metrics entries.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the combined figure (default: vis/grad_metrics_<timestamp>.png).",
    )
    return parser.parse_args()


def main(argv=None) -> int:
    args = parse_args()
    timestamp = None
    if args.output is None:
        timestamp = Path(args.logs[0]).stem.split("_")[-1]
        output_path = Path("vis") / f"grad_metrics_{timestamp}.png"
    else:
        output_path = Path(args.output)

    series_per_log: List[Tuple[str, Dict[str, Series]]] = []
    for log in args.logs:
        path = Path(log)
        if not path.exists():
            print(f"[warn] log not found: {path}")
            continue
        label = path.stem
        series_per_log.append((label, extract_grad_metrics(path)))

    summarize(series_per_log)
    make_plot(series_per_log, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

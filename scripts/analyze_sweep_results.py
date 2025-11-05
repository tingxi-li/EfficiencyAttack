#!/usr/bin/env python3
"""
Aggregate loss-sweep results and produce a simple visualization.

Usage:
    python scripts/analyze_sweep_results.py \
        --log_dir log \
        --metric obj_count_2 \
        --output vis/loss_sweep_summary.png
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SweepResult:
    run_id: str
    log_path: Path
    config_path: Path
    queries: Tuple[int, int]
    weights: Tuple[float, float]
    metric_value: float
    metric_key: str
    extra: Dict[str, float]
    final_obj1: Optional[float]
    final_obj2: Optional[float]


def _load_json(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def _pair_logs_and_configs(log_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for cfg_path in sorted(log_dir.glob("*_configs.json")):
        log_path = cfg_path.with_name(cfg_path.name.replace("_configs", ""))
        if log_path.exists():
            pairs.append((log_path, cfg_path))
    return pairs


def _extract_combo(meta: dict, config: dict) -> Tuple[Tuple[int, int], Tuple[float, float], str]:
    if meta:
        queries = tuple(int(x) for x in meta.get("queries", []))
        weights = tuple(float(x) for x in meta.get("weights", []))
        tag = str(meta.get("tag", ""))
        if queries and weights:
            return queries, weights, tag
    queries = (int(config.get("num_queries_1", -1)), int(config.get("num_queries_2", -1)))
    weights = (
        float(config.get("cls_loss_weight_1", math.nan)),
        float(config.get("cls_loss_weight_2", math.nan)),
    )
    tag = f"q{queries[0]}-{queries[1]}_w{weights[0]}-{weights[1]}"
    return queries, weights, tag


def _mean_final_metric(log_data: dict, key: str) -> Optional[float]:
    finals = []
    for _, series in log_data.items():
        if not series:
            continue
        last = series[-1]
        if key not in last or last[key] is None:
            continue
        try:
            finals.append(float(last[key]))
        except (TypeError, ValueError):
            continue
    if not finals:
        return None
    return float(np.mean(finals))


def _final_metric_map(log_data: dict, keys: List[str]) -> Dict[str, Optional[float]]:
    return {key: _mean_final_metric(log_data, key) for key in keys}


def _collect_results(
    log_dir: Path,
    metric_key: str,
    fallback_key: Optional[str],
) -> List[SweepResult]:
    results: List[SweepResult] = []
    for log_path, cfg_path in _pair_logs_and_configs(log_dir):
        config = _load_json(cfg_path)
        log_data = _load_json(log_path)

        combo_meta = config.get("_sweep_combo", {})
        queries, weights, combo_tag = _extract_combo(combo_meta, config)

        final_map = _final_metric_map(log_data, [metric_key, fallback_key] if fallback_key else [metric_key])
        value = final_map.get(metric_key)
        fallback_used = False
        if value is None and fallback_key:
            value = final_map.get(fallback_key)
            fallback_used = value is not None

        if value is None:
            print(f"[warn] metric '{metric_key}' missing in {log_path.name}", file=sys.stderr)
            continue

        result = SweepResult(
            run_id=combo_tag or log_path.stem,
            log_path=log_path,
            config_path=cfg_path,
            queries=queries,
            weights=weights,
            metric_value=value,
            metric_key=metric_key if not fallback_used else f"{metric_key} (fallback {fallback_key})",
            extra={},
            final_obj1=_mean_final_metric(log_data, "obj_count_1"),
            final_obj2=_mean_final_metric(log_data, "obj_count_2"),
        )
        results.append(result)
    return results


def _make_scatter(results: List[SweepResult], metric_label: str, output_path: Path) -> None:
    if not results:
        print("No results gathered; skipping visualization.")
        return

    queries_groups: Dict[Tuple[int, int], List[SweepResult]] = defaultdict(list)
    for result in results:
        queries_groups[result.queries].append(result)

    num_groups = len(queries_groups)
    cols = min(3, num_groups)
    rows = (num_groups + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 1, 4 * rows + 1), squeeze=False)
    vmin = min(r.metric_value for r in results)
    vmax = max(r.metric_value for r in results)
    cmap = plt.cm.get_cmap("viridis")

    for ax in axes.flat:
        ax.axis("off")

    for idx, (queries, group_results) in enumerate(sorted(queries_groups.items())):
        ax = axes[idx // cols][idx % cols]
        ax.set_title(f"queries: {queries[0]}/{queries[1]}")
        ax.set_xlabel("cls_loss_weight_1")
        ax.set_ylabel("cls_loss_weight_2")
        ax.grid(True, linestyle="--", alpha=0.3)

        xs = [r.weights[0] for r in group_results]
        ys = [r.weights[1] for r in group_results]
        cs = [r.metric_value for r in group_results]

        sc = ax.scatter(xs, ys, c=cs, cmap=cmap, vmin=vmin, vmax=vmax, s=120, edgecolor="k")
        for r in group_results:
            ax.annotate(f"{r.metric_value:.2f}", xy=(r.weights[0], r.weights[1]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlim(min(xs) - 0.05, max(xs) + 0.05)
        ax.set_ylim(min(ys) - 0.05, max(ys) + 0.05)
        ax.axis("on")

    fig.suptitle(f"Loss Sweep Results ({metric_label})", fontsize=14)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.85)
    cbar.set_label(metric_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[viz ] saved to {output_path}")


def _safe_divide(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if abs(denom) < 1e-12:
        return math.nan
    return float(numerator) / denom


def _plot_by_query(results: List[SweepResult], output_path: Path) -> None:
    groups: Dict[Tuple[int, int], List[SweepResult]] = defaultdict(list)
    for result in results:
        groups[result.queries].append(result)

    if not groups:
        return

    num_groups = len(groups)
    cols = min(3, num_groups)
    rows = (num_groups + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4 * rows + 1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    plotted = False
    for idx, (queries, group) in enumerate(sorted(groups.items())):
        ax = axes[idx // cols][idx % cols]
        points = []
        for result in group:
            ratio = _safe_divide(result.weights[0], result.weights[1])
            if math.isnan(ratio) or math.isinf(ratio):
                continue
            obj1 = result.final_obj1
            obj2 = result.final_obj2
            if obj1 is None and obj2 is None:
                continue
            points.append((ratio, obj1 if obj1 is not None else math.nan, obj2 if obj2 is not None else math.nan))

        if not points:
            ax.set_title(f"Queries {queries[0]}/{queries[1]}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        points.sort(key=lambda x: x[0])
        ratios = [p[0] for p in points]
        obj1_values = [p[1] for p in points]
        obj2_values = [p[2] for p in points]

        ax.plot(ratios, obj1_values, marker="o", label="obj_count_1")
        ax.plot(ratios, obj2_values, marker="s", label="obj_count_2")
        ax.set_title(f"Queries {queries[0]}/{queries[1]}")
        ax.set_xlabel("cls_loss_weight ratio (w1 / w2)")
        ax.set_ylabel("Average object count")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        ax.axis("on")
        plotted = True

    if not plotted:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Per-query loss weight ratios vs. object counts", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[viz ] per-query ratio grid saved to {output_path}")


# def _plot_by_weights(results: List[SweepResult], output_path: Path) -> None:
#     groups: Dict[Tuple[float, float], List[SweepResult]] = defaultdict(list)
#     for result in results:
#         groups[result.weights].append(result)

#     if not groups:
#         return

#     num_groups = len(groups)
#     cols = min(3, num_groups)
#     rows = (num_groups + cols - 1) // cols

#     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4 * rows + 1), squeeze=False)
#     for ax in axes.flat:
#         ax.axis("off")

#     plotted = False
#     for idx, (weights, group) in enumerate(sorted(groups.items())):
#         ax = axes[idx // cols][idx % cols]
#         points = []
#         for result in group:
#             obj1 = result.final_obj1
#             obj2 = result.final_obj2
#             if obj1 is None and obj2 is None:
#                 continue
#             label = f"({result.queries[0]},{result.queries[1]})"
#             points.append(((result.queries[0], result.queries[1]), label, obj1 if obj1 is not None else math.nan, obj2 if obj2 is not None else math.nan))

#         if not points:
#             ax.set_title(f"Weights {weights[0]:g}/{weights[1]:g}")
#             ax.text(0.5, 0.5, "No data", ha="center", va="center")
#             continue

#         points.sort(key=lambda x: (x[0][0], x[0][1]))
#         labels = [p[1] for p in points]
#         obj1_values = [p[2] for p in points]
#         obj2_values = [p[3] for p in points]
#         x_positions = list(range(len(points)))

#         ax.plot(x_positions, obj1_values, marker="o", label="obj_count_1")
#         ax.plot(x_positions, obj2_values, marker="s", label="obj_count_2")
#         ax.set_title(f"Weights {weights[0]:g}/{weights[1]:g}")
#         ax.set_xlabel("Queries (q1, q2)")
#         ax.set_ylabel("Average object count")
#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(labels, rotation=25)
#         ax.grid(True, linestyle="--", alpha=0.3)
#         ax.legend()
#         ax.axis("on")
#         plotted = True

#     if not plotted:
#         return

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.suptitle("Per-weight query combos vs. object counts", fontsize=14)
#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     fig.savefig(output_path, dpi=200)
#     plt.close(fig)
#     print(f"[viz ] per-weight grid saved to {output_path}")


def _plot_by_weights(results: List[SweepResult], output_path: Path) -> None:
    groups: Dict[Tuple[float, float], List[SweepResult]] = defaultdict(list)
    for result in results:
        groups[result.weights].append(result)

    if not groups:
        return

    num_groups = len(groups)
    cols = min(3, num_groups)
    rows = (num_groups + cols - 1) // cols

    # Increased base width per subplot for better spacing
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols + 2, 5 * rows + 1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    plotted = False
    for idx, (weights, group) in enumerate(sorted(groups.items())):
        ax = axes[idx // cols][idx % cols]
        points = []
        for result in group:
            obj1 = result.final_obj1
            obj2 = result.final_obj2
            if obj1 is None and obj2 is None:
                continue
            label = f"({result.queries[0]},{result.queries[1]})"
            points.append(((result.queries[0], result.queries[1]), label, obj1 if obj1 is not None else math.nan, obj2 if obj2 is not None else math.nan))

        if not points:
            ax.set_title(f"Weights {weights[0]:g}/{weights[1]:g}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        points.sort(key=lambda x: (x[0][0], x[0][1]))
        labels = [p[1] for p in points]
        obj1_values = [p[2] for p in points]
        obj2_values = [p[3] for p in points]

        # --- better spaced x positions
        spacing_factor = 2.5
        x_positions = np.arange(0, len(points) * spacing_factor, spacing_factor)

        ax.plot(x_positions, obj1_values, marker="o", label="obj_count_1")
        ax.plot(x_positions, obj2_values, marker="s", label="obj_count_2")

        ax.set_title(f"Weights {weights[0]:g}/{weights[1]:g}")
        ax.set_xlabel("Queries (q1, q2)")
        ax.set_ylabel("Average object count")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)

        ax.set_xlim(-spacing_factor, len(points) * spacing_factor)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        ax.axis("on")
        plotted = True

    if not plotted:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Per-weight query combos vs. object counts", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[viz ] per-weight grid saved to {output_path}")
    
def _print_summary(results: List[SweepResult], topk: Optional[int]) -> None:
    sorted_results = sorted(results, key=lambda r: r.metric_value)
    lines = []
    width = 5
    lines.append("idx | metric  | weights         | queries | run_id")
    lines.append("-" * 60)
    for idx, result in enumerate(sorted_results):
        if topk is not None and idx >= topk:
            break
        lines.append(
            f"{idx:>3} | {result.metric_value:>7.3f} | "
            f"({result.weights[0]:>5.2f},{result.weights[1]:>5.2f}) | "
            f"{result.queries[0]:>3}/{result.queries[1]:<3} | {result.run_id}"
        )
    print("\n".join(lines))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze sweep results and generate a visualization.")
    parser.add_argument("--log_dir", type=str, default="log", help="Directory containing sweep logs/configs.")
    parser.add_argument("--metric", type=str, default="obj_count_2", help="Metric key to average from final iteration.")
    parser.add_argument("--fallback_metric", type=str, default="total_loss", help="Fallback metric if primary is missing.")
    parser.add_argument("--output", type=str, default=None, help="Output image path for visualization.")
    parser.add_argument("--viz_dir", type=str, default="vis", help="Directory for additional ratio plots.")
    parser.add_argument("--topk", type=int, default=10, help="How many rows to print in the summary table. -1 for all.")
    parser.add_argument("--no_plot", action="store_true", help="Disable visualization generation.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    fallback = args.fallback_metric if args.fallback_metric else None
    results = _collect_results(log_dir, args.metric, fallback)
    if not results:
        print("No sweep results were collected.")
        return 0

    topk = None if args.topk is None or args.topk < 0 else args.topk
    _print_summary(results, topk)

    if not args.no_plot:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = Path(args.output) if args.output else Path(args.viz_dir) / f"loss_sweep_summary_{timestamp}.png"
        _make_scatter(results, args.metric, output_path)

        viz_root = Path(args.viz_dir)
        _plot_by_query(results, viz_root / f"per_query_{timestamp}.png")
        _plot_by_weights(results, viz_root / f"per_loss_{timestamp}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

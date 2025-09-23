import os
import re
import pdb
import sys
import math
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import islice

def parse_args():
    parser = argparse.ArgumentParser(description="avg detection over iterations")
    parser.add_argument("--base_dir", type=str, default="./logs/", help="path to log directory")
    parser.add_argument("--output_dir", type=str, default="./vis", help="Output directory for the plot")
    parser.add_argument("--log_file_index", type=int, default=None, help="Optional: only process this file index (sorted)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect candidate log files
    filenames = [f for f in os.listdir(base_dir) if f.endswith(".json") and not f.endswith("_configs.json")]
    filenames = sorted(filenames)

    # Optionally restrict to a single file by index (for convenience)
    if args.log_file_index is not None:
        if args.log_file_index < 0 or args.log_file_index >= len(filenames):
            print(f"Provided log_file_index {args.log_file_index} out of range (0..{len(filenames)-1}). Exiting.")
            sys.exit(1)
        filenames = [filenames[args.log_file_index]]

    if not filenames:
        print(f"No log files found in {base_dir}")
        sys.exit(0)

    for fname in filenames:
        # Derive pipeline name and timestamp from filename if possible
        parts = re.split(r"(\d{8}-\d{6})", fname)
        if len(parts) >= 3:
            pipeline_name = parts[0]
            timestamp = parts[1]
        else:
            pipeline_name = os.path.splitext(fname)[0]
            timestamp = ""

        out_basename = f"{pipeline_name}_{timestamp}_avg_detection_over_iterations" if timestamp else f"{pipeline_name}_avg_detection_over_iterations"
        outfile = os.path.join(args.output_dir, f"{out_basename}.png")

        # Skip if the visualization already exists
        if os.path.exists(outfile):
            print(f"Skip existing: {outfile}")
            continue

        log_path = os.path.join(base_dir, fname)
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {log_path}: {e}")
            continue

        try:
            # Expect data as { datapoint_idx: [ {metric_key: value, ...}, ... ], ... }
            first_datapoint_index = next(iter(data))
            first_datapoint_logs = data[first_datapoint_index]
            if not first_datapoint_logs:
                print(f"Empty logs in {fname}; skipping.")
                continue
            first_iteration = first_datapoint_logs[0]

            matrix = np.zeros((len(data), len(first_datapoint_logs), len(first_iteration)))
            # build a consistent ordering of log keys (to keep columns aligned)
            log_keys = list(first_iteration.keys())

            for i, (datapoint_idx, logs) in enumerate(data.items()):
                for j, iteration_dict in enumerate(logs):
                    for k, key in enumerate(log_keys):
                        matrix[i, j, k] = iteration_dict[key]

            avg_all_datapoints = np.nanmean(matrix, axis=0)

            iterations = avg_all_datapoints[:, 0]   # first key = iteration count
            metrics = avg_all_datapoints[:, 1:]     # the rest are actual metrics
            metric_names = log_keys[1:]             # skip the first key (iteration)

            # Define markers (cycled if more metrics than markers)
            markers = ['o', 's', '^', 'D', 'v', 'x', '*']

            plt.figure(figsize=(10, 6))
            for i, name in enumerate(metric_names):
                plt.plot(
                    iterations,
                    metrics[:, i],
                    marker=markers[i % len(markers)],
                    label=name
                )

            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.title("Metrics Over Iterations (Average Across Datapoints)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(outfile)
            print(f"Plot saved to {outfile}")
            plt.close()
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    # pdb.set_trace()

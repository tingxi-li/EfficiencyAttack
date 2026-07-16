"""Aggregate SUMMARY.json files into the three required tables.

Tables:
  A) Clean-side-effects (4 rows; no seed averaging).
  B) Mix study: 5 conditions x 3 ratios, mean +/- std over 5 seeds.

Metrics per row:
  Wall Time | Throughput | Avg E2E | P50 | P95 | P99 | LPR Workload | #Drops | Total FLOPs
"""
import os
import json
import math
import glob
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(p):
    with open(p) as f:
        return json.load(f)


def _row(summary):
    pl = summary["pipeline"]
    pil = summary["per_image_latency"]
    fl = summary["flops"]
    wl = summary.get("workload", {})
    lpr = wl.get("lprStream", {}).get("count", 0)
    drops = summary.get("drops_total", 0)
    return {
        "wall": float(pl["total_wall_time"]),
        "tput": float(pl["throughput_img_per_sec"]),
        "avg":  float(pil["avg"]),
        "p50":  float(pil.get("p50", 0)),
        "p95":  float(pil.get("p95", 0)),
        "p99":  float(pil.get("p99", 0)),
        "lpr":  float(lpr),
        "drops": float(drops),
        "flops": float(fl["total_flops"]),
    }


def _stat(values):
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    return mean, math.sqrt(var)


def _fmt_flops(x):
    if x >= 1e12:
        return f"{x/1e12:.1f}T"
    if x >= 1e9:
        return f"{x/1e9:.1f}G"
    return f"{x:.0f}"


def _fmt_pair(mean, std, fmt="{:.2f}"):
    if math.isnan(mean):
        return "n/a"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


HEADER = ["Wall(s)", "Tput(img/s)", "AvgE2E", "P50", "P95", "P99", "LPR#", "Drops", "TotalFLOPs"]


def table_clean_side_effects(base="./profile_exp/clean100"):
    print("\n=== TABLE A: Clean-image side effects (n=100) ===")
    print(f"{'Defense':<14} | " + " | ".join(f"{h:>14}" for h in HEADER))
    print("-" * (14 + 3 + 14 * len(HEADER) + 3 * (len(HEADER) - 1)))
    for label in ["none", "gauss", "smooth", "svm"]:
        path = os.path.join(base, label, "SUMMARY.json")
        if not os.path.exists(path):
            print(f"{label:<14} | (missing)")
            continue
        r = _row(_load(path))
        cells = [
            f"{r['wall']:.1f}", f"{r['tput']:.3f}", f"{r['avg']:.2f}", f"{r['p50']:.2f}",
            f"{r['p95']:.2f}", f"{r['p99']:.2f}", f"{r['lpr']:.0f}",
            f"{r['drops']:.0f}", _fmt_flops(r['flops'])
        ]
        print(f"{label:<14} | " + " | ".join(f"{c:>14}" for c in cells))


def table_mix_study(base="./profile_exp/mix"):
    print("\n=== TABLE B: Mixed clean+attacked inputs, n=100, 5-seed mean +/- std ===")
    cond_order = ["no_defense", "def_gaussian", "def_smoothing", "def_svm", "sys_batch_conf"]
    pretty = {
        "no_defense": "none (i)",
        "def_gaussian": "gauss (ii)",
        "def_smoothing": "smooth (ii)",
        "def_svm": "svm (ii)",
        "sys_batch_conf": "b16+conf.5 (iii)",
    }
    ratios = [0.10, 0.05, 0.01]
    ratio_label = {0.10: "9:1", 0.05: "9.5:0.5", 0.01: "9.9:0.1"}
    print(f"{'Ratio':<10}{'Cond':<20} | " + " | ".join(f"{h:>20}" for h in HEADER))
    print("-" * (10 + 20 + 3 + 20 * len(HEADER) + 3 * (len(HEADER) - 1)))
    for ratio in ratios:
        for cname in cond_order:
            seeds = sorted(glob.glob(os.path.join(base, f"r{ratio:.2f}", cname, "seed*", "SUMMARY.json")))
            if not seeds:
                print(f"{ratio_label[ratio]:<10}{pretty[cname]:<20} | (missing)")
                continue
            rows = [_row(_load(p)) for p in seeds]
            stats = {k: _stat([r[k] for r in rows]) for k in rows[0]}
            cells = [
                _fmt_pair(*stats["wall"], "{:.1f}"),
                _fmt_pair(*stats["tput"], "{:.3f}"),
                _fmt_pair(*stats["avg"], "{:.2f}"),
                _fmt_pair(*stats["p50"], "{:.2f}"),
                _fmt_pair(*stats["p95"], "{:.2f}"),
                _fmt_pair(*stats["p99"], "{:.2f}"),
                _fmt_pair(*stats["lpr"], "{:.0f}"),
                _fmt_pair(*stats["drops"], "{:.1f}"),
                _fmt_flops(stats["flops"][0]),
            ]
            print(f"{ratio_label[ratio]:<10}{pretty[cname]:<20} | " + " | ".join(f"{c:>20}" for c in cells))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", default="./profile_exp/clean100")
    ap.add_argument("--mix_dir", default="./profile_exp/mix")
    args = ap.parse_args()
    table_clean_side_effects(args.clean_dir)
    table_mix_study(args.mix_dir)


if __name__ == "__main__":
    main()

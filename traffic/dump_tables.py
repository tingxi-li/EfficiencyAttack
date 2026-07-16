"""Dump every table we have to ./results/ as both Markdown and CSV.

Tables produced:
  TABLE_A_clean_side_effects   (4 rows, single run each)
  TABLE_B_mix_study            (5 conds x 3 ratios, mean +/- std over 5 seeds)
  TABLE_B_svm_per_seed         (3 ratios x 5 seeds, raw values)
  TABLE_C_attacked10           (3 defenses, single run each)
  ALL_TABLES.md                concatenation of all four for easy reading
"""
import os
import json
import math
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

H = ["Wall(s)", "Tput(img/s)", "AvgE2E", "P50", "P95", "P99", "LPR#", "Drops", "TotalFLOPs"]
RATIO_LABEL = {0.10: "9:1", 0.05: "9.5:0.5", 0.01: "9.9:0.1"}
PRETTY = {
    "no_defense": "none (i)",
    "def_gaussian": "gauss (ii)",
    "def_smoothing": "smooth (ii)",
    "def_svm": "svm (ii)",
    "sys_batch_conf": "b16+conf.5 (iii)",
}


def _load(p):
    with open(p) as f:
        return json.load(f)


def _row(s):
    pl = s["pipeline"]; pil = s["per_image_latency"]; fl = s["flops"]; wl = s.get("workload", {})
    return dict(
        wall=float(pl["total_wall_time"]),
        tput=float(pl["throughput_img_per_sec"]),
        avg=float(pil["avg"]), p50=float(pil.get("p50", 0)),
        p95=float(pil.get("p95", 0)), p99=float(pil.get("p99", 0)),
        lpr=float(wl.get("lprStream", {}).get("count", 0)),
        drops=float(s.get("drops_total", 0)),
        flops=float(fl["total_flops"]),
    )


def _stat(values):
    if not values:
        return float("nan"), float("nan")
    n = len(values); m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / max(n - 1, 1)
    return m, math.sqrt(var)


def _f_flops(x):
    if math.isnan(x):
        return "n/a"
    if x >= 1e12:
        return f"{x/1e12:.1f}T"
    if x >= 1e9:
        return f"{x/1e9:.1f}G"
    return f"{x:.0f}"


def _pair(mean, std, fmt="{:.2f}"):
    if math.isnan(mean):
        return "n/a"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def _md_table(headers, rows):
    pad = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    lines = ["| " + " | ".join(h.ljust(p) for h, p in zip(headers, pad)) + " |"]
    align = ["---:" if i > 0 else ":---" for i in range(len(headers))]
    lines.append("| " + " | ".join(align) + " |")
    for r in rows:
        lines.append("| " + " | ".join(c.ljust(p) for c, p in zip(r, pad)) + " |")
    return "\n".join(lines)


def _csv_write(path, headers, rows):
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(c).replace(",", ";") for c in r) + "\n")


def table_a():
    headers = ["Defense"] + H
    rows = []
    for label in ["none", "gauss", "smooth", "svm"]:
        p = os.path.join(ROOT, "profile_exp", "clean100", label, "SUMMARY.json")
        if not os.path.exists(p):
            rows.append([label, "(missing)"] + ["-"] * (len(H) - 1))
            continue
        r = _row(_load(p))
        rows.append([
            label,
            f"{r['wall']:.1f}", f"{r['tput']:.3f}", f"{r['avg']:.2f}",
            f"{r['p50']:.2f}", f"{r['p95']:.2f}", f"{r['p99']:.2f}",
            f"{r['lpr']:.0f}", f"{r['drops']:.0f}", _f_flops(r['flops']),
        ])
    title = "TABLE A — Clean-image side effects (n=100, single run each)"
    config = ("Defenses: gauss=N(0,5/255), smooth=median k=3 strength=0.8, "
              "svm=LinearSVC trained on 900 clean + 90 attacked, FNR-target tau≈1.12.")
    md = f"# {title}\n\n{config}\n\n" + _md_table(headers, rows) + "\n"
    with open(os.path.join(OUT, "TABLE_A_clean_side_effects.md"), "w") as f:
        f.write(md)
    _csv_write(os.path.join(OUT, "TABLE_A_clean_side_effects.csv"), headers, rows)
    return md


def table_b():
    cond_order = ["no_defense", "def_gaussian", "def_smoothing", "def_svm", "sys_batch_conf"]
    headers = ["Ratio", "Cond"] + H
    rows = []
    for ratio in [0.10, 0.05, 0.01]:
        for cname in cond_order:
            seeds = sorted(glob.glob(os.path.join(
                ROOT, "profile_exp", "mix", f"r{ratio:.2f}", cname, "seed*", "SUMMARY.json")))
            if not seeds:
                rows.append([RATIO_LABEL[ratio], PRETTY[cname]] + ["(missing)"] * len(H))
                continue
            data = [_row(_load(p)) for p in seeds]
            stats = {k: _stat([d[k] for d in data]) for k in data[0]}
            rows.append([
                RATIO_LABEL[ratio], PRETTY[cname],
                _pair(*stats["wall"], "{:.1f}"),
                _pair(*stats["tput"], "{:.3f}"),
                _pair(*stats["avg"], "{:.2f}"),
                _pair(*stats["p50"], "{:.2f}"),
                _pair(*stats["p95"], "{:.2f}"),
                _pair(*stats["p99"], "{:.2f}"),
                _pair(*stats["lpr"], "{:.0f}"),
                _pair(*stats["drops"], "{:.1f}"),
                _f_flops(stats["flops"][0]),
            ])
    title = "TABLE B — Mixed clean+attacked inputs, n=100, mean ± std over 5 seeds"
    notes = ("Conditions: (i) no defense; (ii) input defense alone (gauss/smooth/svm); "
             "(iii) system defense (batch=16, conf=0.5).\n"
             "Smoothing/SVM rows used the held-out attack pool (indices 90–99). "
             "Other rows used indices 50–99.")
    md = f"# {title}\n\n{notes}\n\n" + _md_table(headers, rows) + "\n"
    with open(os.path.join(OUT, "TABLE_B_mix_study.md"), "w") as f:
        f.write(md)
    _csv_write(os.path.join(OUT, "TABLE_B_mix_study.csv"), headers, rows)
    return md


def table_b_svm():
    headers = ["Ratio", "Seed"] + H
    rows = []
    for ratio in [0.01, 0.05, 0.10]:
        for seed in [0, 1, 2, 3, 4]:
            p = os.path.join(ROOT, "profile_exp", "mix", f"r{ratio:.2f}", "def_svm",
                             f"seed{seed}", "SUMMARY.json")
            if not os.path.exists(p):
                rows.append([RATIO_LABEL[ratio], str(seed)] + ["(missing)"] * len(H))
                continue
            r = _row(_load(p))
            rows.append([
                RATIO_LABEL[ratio], str(seed),
                f"{r['wall']:.1f}", f"{r['tput']:.3f}", f"{r['avg']:.2f}",
                f"{r['p50']:.2f}", f"{r['p95']:.2f}", f"{r['p99']:.2f}",
                f"{r['lpr']:.0f}", f"{r['drops']:.0f}", _f_flops(r['flops']),
            ])
    title = "TABLE B-svm — per-seed def_svm results (n=100)"
    notes = ("Reveals bimodal SVM behavior at low attack ratios: each seed picks a "
             "different attack subset; SVM either catches it or misses it cleanly.")
    md = f"# {title}\n\n{notes}\n\n" + _md_table(headers, rows) + "\n"
    with open(os.path.join(OUT, "TABLE_B_svm_per_seed.md"), "w") as f:
        f.write(md)
    _csv_write(os.path.join(OUT, "TABLE_B_svm_per_seed.csv"), headers, rows)
    return md


def table_c():
    headers = ["Defense"] + H
    rows = []
    for label in ["gauss", "smooth", "svm"]:
        p = os.path.join(ROOT, "profile_exp", "attacked10", label, "SUMMARY.json")
        if not os.path.exists(p):
            rows.append([label, "(missing)"] + ["-"] * (len(H) - 1))
            continue
        r = _row(_load(p))
        rows.append([
            label,
            f"{r['wall']:.1f}", f"{r['tput']:.3f}", f"{r['avg']:.2f}",
            f"{r['p50']:.2f}", f"{r['p95']:.2f}", f"{r['p99']:.2f}",
            f"{r['lpr']:.0f}", f"{r['drops']:.0f}", _f_flops(r['flops']),
        ])
    title = "TABLE C — Input defenses on 10 attacked images (held-out indices 90–99)"
    notes = ("Single run per defense, eval_size=10, mix_ratio=1.0. "
             "All 10 images are attacked and unseen by the SVM.")
    md = f"# {title}\n\n{notes}\n\n" + _md_table(headers, rows) + "\n"
    with open(os.path.join(OUT, "TABLE_C_attacked10.md"), "w") as f:
        f.write(md)
    _csv_write(os.path.join(OUT, "TABLE_C_attacked10.csv"), headers, rows)
    return md


def main():
    parts = [table_a(), table_b(), table_b_svm(), table_c()]
    combined = "\n\n---\n\n".join(parts)
    with open(os.path.join(OUT, "ALL_TABLES.md"), "w") as f:
        f.write(combined)
    print(f"wrote tables to {OUT}/")
    for fn in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, fn))
        print(f"  {fn:<40} {size} bytes")


if __name__ == "__main__":
    main()

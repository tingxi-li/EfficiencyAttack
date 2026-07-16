"""Driver that chains all experiments across 2 GPUs.

Layout under ./profile_exp/:
  clean100/{none,gauss,smooth,svm}/   <- Part F: clean-image side effects (n=100)
  mix/r{ratio}/{cond}/seed{N}/        <- Part G: mixed inputs

Conditions for Part G (per ratio, per seed):
  no_defense       : no defense, batch=1
  def_gaussian     : input_defense=gaussian, batch=1
  def_smoothing    : input_defense=smoothing, batch=1
  def_svm          : input_defense=svm, batch=1
  sys_batch_conf   : batch=16, conf_filter=0.5

Total: 4 (clean side-effects) + 3 ratios * 5 conditions * 5 seeds = 79 runs.
Runs 2 in parallel — one per GPU — by alternating CUDA_VISIBLE_DEVICES.
"""
import os
import sys
import time
import json
import threading
import subprocess
from queue import Queue, Empty

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(ROOT, "pipe.py")
PYTHON = "/home/lxt230026/.conda/envs/dyndl/bin/python"

EVAL_SIZE = 100
SEEDS = [0, 1, 2, 3, 4]
RATIOS = [0.10, 0.05, 0.01]

CONDITIONS = {
    "no_defense":     {"input_defense": "none",      "conf": None, "batch": 1},
    "def_gaussian":   {"input_defense": "gaussian",  "conf": None, "batch": 1},
    "def_smoothing":  {"input_defense": "smoothing", "conf": None, "batch": 1},
    "def_svm":        {"input_defense": "svm",       "conf": None, "batch": 1},
    "sys_batch_conf": {"input_defense": "none",      "conf": 0.5,  "batch": 16},
}

CLEAN_SIDE_EFFECTS = {
    "none":     {"input_defense": "none"},
    "gauss":    {"input_defense": "gaussian"},
    "smooth":   {"input_defense": "smoothing"},
    "svm":      {"input_defense": "svm"},
}

LOG_LOCK = threading.Lock()
def log(msg):
    with LOG_LOCK:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def build_mix_cmd(ratio, cond_name, cond, seed):
    out = os.path.join("./profile_exp", "mix", f"r{ratio:.2f}", cond_name, f"seed{seed}")
    args = [PYTHON, PIPE,
            "--mix_mode",
            "--mix_ratio", f"{ratio}",
            "--shuffle_seed", str(seed),
            "--eval_size", str(EVAL_SIZE),
            "--input_defense", cond["input_defense"],
            "--ps_path", out,
            "--batch_size", str(cond["batch"]),
            "--clean_path", "../saved/clean_pool",
            "--attack_path", "../saved/model_0/teaspoon_tgt_2"]
    # SVM-affected runs must skip the SVM training subset (attack[:90])
    if cond.get("eval_index_start") is not None:
        args += ["--eval_index_start", str(cond["eval_index_start"])]
    if cond["conf"] is not None:
        args += ["--conf_filter", str(cond["conf"])]
    return args, out


def build_clean_side_cmd(label, cond):
    """Clean-only run (n=100) using mix_mode with ratio=0.0 so source labels are recorded."""
    out = os.path.join("./profile_exp", "clean100", label)
    args = [PYTHON, PIPE,
            "--mix_mode",
            "--mix_ratio", "0.0",
            "--shuffle_seed", "0",
            "--eval_size", str(EVAL_SIZE),
            "--input_defense", cond["input_defense"],
            "--ps_path", out,
            "--batch_size", "1",
            "--clean_path", "../saved/clean_pool",
            "--attack_path", "../saved/model_0/teaspoon_tgt_2"]
    return args, out


def task_id(out):
    return out.replace("./profile_exp/", "").replace("/", "::")


def run_one(args, out, gpu):
    summary_path = os.path.join(out, "SUMMARY.json")
    if os.path.exists(summary_path):
        log(f"SKIP gpu{gpu} {task_id(out)} (already done)")
        return out, 0, 0.0
    os.makedirs(out, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = os.path.join(out, "RUN.log")
    log(f"START gpu{gpu} {task_id(out)}")
    t0 = time.time()
    with open(log_path, "w") as fh:
        try:
            r = subprocess.run(args, env=env, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=10800)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            fh.write("\n[DRIVER] timed out after 3h\n")
    dt = time.time() - t0
    log(f"DONE  gpu{gpu} {task_id(out)} rc={rc} {dt:.0f}s")
    return out, rc, dt


def worker(gpu, q):
    while True:
        try:
            args, out = q.get_nowait()
        except Empty:
            return
        try:
            run_one(args, out, gpu)
        except Exception as e:
            log(f"ERR  gpu{gpu} {task_id(out)} -> {e}")


def main():
    tasks = []

    # Part F: clean side-effects (4 runs, fastest first)
    for label, cond in CLEAN_SIDE_EFFECTS.items():
        tasks.append(build_clean_side_cmd(label, cond))

    # Part G: mixed-ratio experiments (75 runs)
    # Order by expected wall-time ascending so 2-GPU parallel finishes faster
    cond_order = ["sys_batch_conf", "def_svm", "def_smoothing", "def_gaussian", "no_defense"]
    for ratio in [0.01, 0.05, 0.10]:  # easy ratios first
        for cname in cond_order:
            cond = CONDITIONS[cname]
            for seed in SEEDS:
                tasks.append(build_mix_cmd(ratio, cname, cond, seed))

    log(f"queued {len(tasks)} tasks")

    q = Queue()
    for t in tasks:
        q.put(t)

    threads = [
        threading.Thread(target=worker, args=(0, q), daemon=False),
        threading.Thread(target=worker, args=(1, q), daemon=False),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    log("ALL DONE")


if __name__ == "__main__":
    main()

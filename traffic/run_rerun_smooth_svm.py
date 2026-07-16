"""Re-run only def_smoothing and def_svm Table B cells with the weakened defenses.

Uses attack[90:] (10 held-out attacks) for both, so the SVM's training subset
(attack[:90]) is excluded.

Old SUMMARY.json files in those cell dirs are removed before re-running.
"""
import os
import sys
import time
import threading
import subprocess
from queue import Queue, Empty

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(ROOT, "pipe.py")
PYTHON = "/home/lxt230026/.conda/envs/dyndl/bin/python"

EVAL_SIZE = 100
SEEDS = [0, 1, 2, 3, 4]
RATIOS = [0.10, 0.05, 0.01]
CLEAN_EVAL_IDX_START = 50   # clean_pool has 150; eval = 100
ATTACK_EVAL_IDX_START = 90  # attack pool has 100; eval = 10 held-out (SVM never saw)

CONDITIONS = {
    "def_smoothing":  {"input_defense": "smoothing", "conf": None, "batch": 1},
    "def_svm":        {"input_defense": "svm",       "conf": None, "batch": 1},
}

LOG_LOCK = threading.Lock()
def log(msg):
    with LOG_LOCK:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def build_cmd(ratio, cname, cond, seed):
    out = os.path.join("./profile_exp", "mix", f"r{ratio:.2f}", cname, f"seed{seed}")
    args = [PYTHON, PIPE,
            "--mix_mode",
            "--mix_ratio", f"{ratio}",
            "--shuffle_seed", str(seed),
            "--eval_size", str(EVAL_SIZE),
            "--input_defense", cond["input_defense"],
            "--ps_path", out,
            "--batch_size", str(cond["batch"]),
            "--clean_path", "../saved/clean_pool",
            "--attack_path", "../saved/model_0/teaspoon_tgt_2",
            "--clean_eval_index_start", str(CLEAN_EVAL_IDX_START),
            "--attack_eval_index_start", str(ATTACK_EVAL_IDX_START)]
    if cond["conf"] is not None:
        args += ["--conf_filter", str(cond["conf"])]
    return args, out


def run_one(args, out, gpu):
    # remove stale dir entirely (forces fresh run)
    if os.path.isdir(out):
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
        os.rmdir(out)
    os.makedirs(out, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = os.path.join(out, "RUN.log")
    log(f"START gpu{gpu} {out.replace('./profile_exp/', '')}")
    t0 = time.time()
    with open(log_path, "w") as fh:
        try:
            r = subprocess.run(args, env=env, cwd=ROOT, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=10800)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            fh.write("\n[DRIVER] timed out after 3h\n")
    dt = time.time() - t0
    log(f"DONE  gpu{gpu} {out.replace('./profile_exp/', '')} rc={rc} {dt:.0f}s")
    return rc


def worker(gpu, q):
    while True:
        try:
            args, out = q.get_nowait()
        except Empty:
            return
        try:
            run_one(args, out, gpu)
        except Exception as e:
            log(f"ERR  gpu{gpu} {out}: {e}")


def main():
    tasks = []
    # Easy ratios first so 2-GPU dispatch finishes faster
    cond_order = ["def_svm", "def_smoothing"]
    for ratio in [0.01, 0.05, 0.10]:
        for cname in cond_order:
            cond = CONDITIONS[cname]
            for seed in SEEDS:
                tasks.append(build_cmd(ratio, cname, cond, seed))
    log(f"queued {len(tasks)} tasks")

    q = Queue()
    for t in tasks: q.put(t)

    threads = [
        threading.Thread(target=worker, args=(0, q), daemon=False),
        threading.Thread(target=worker, args=(1, q), daemon=False),
    ]
    for th in threads: th.start()
    for th in threads: th.join()

    log("ALL DONE")


if __name__ == "__main__":
    main()

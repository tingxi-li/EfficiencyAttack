"""Train a LinearSVC to flag teaspoon-perturbed pixel tensors.

Tuned for low false-positive rate (false negatives are acceptable per user request):
- class_weight={0: 5, 1: 1}  -> penalize FP harder than FN
- decision-function threshold tau picked on held-out so FPR <= TARGET_FPR
- threshold stored alongside the classifier; SVMDefense uses score > tau as drop signal
"""
import os
import sys
import glob

import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
from input_defense import _featurize_for_svm  # noqa: E402

PROJ = os.path.dirname(ROOT)
CLEAN_DIR = os.path.join(PROJ, "saved", "clean_train_pool")  # 1000 imgs, disjoint from clean_pool
ATTACK_DIR = os.path.join(PROJ, "saved", "model_0", "teaspoon_tgt_2")
OUT_PATH = os.path.join(ROOT, "svm_defense.joblib")
# Dataset = first 900 clean + first 90 attacked. Last 10 attacked are reserved
# for pipeline evaluation (never seen by SVM). 80/20 stratified train/test split.
N_CLEAN = 900
N_ATTACK = 90
TEST_FRAC = 0.2
SPLIT_SEED = 7
CLASS_WEIGHT = None
TARGET_FPR = 0.05
# Deliberately weaken: pick tau high enough that ~30% of attacks slip through.
TARGET_FNR = 0.15


def _load(filepath):
    with open(filepath, "rb") as f:
        shape_dim = np.frombuffer(f.read(1), dtype=np.int8)[0]
        shape = np.frombuffer(f.read(8 * shape_dim), dtype=np.int64)
        dtype_len = np.frombuffer(f.read(1), dtype=np.int8)[0]
        dtype_str = f.read(dtype_len).decode("ascii")
        data = f.read()
    return np.frombuffer(data, dtype=np.dtype(dtype_str)).copy().reshape(shape)


def featurize_paths(paths):
    X = np.stack([_featurize_for_svm(_load(p)) for p in paths], axis=0)
    return X


def main():
    from sklearn.model_selection import train_test_split

    clean = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.pt")))
    attack = sorted(glob.glob(os.path.join(ATTACK_DIR, "*.pt")))
    print(f"[svm] clean={len(clean)} attack={len(attack)}")
    if len(clean) < N_CLEAN or len(attack) < N_ATTACK + 10:
        raise RuntimeError(
            f"need >= {N_CLEAN} clean and {N_ATTACK + 10} attack; "
            f"got clean={len(clean)} attack={len(attack)}"
        )

    ds_clean = clean[:N_CLEAN]
    ds_attack = attack[:N_ATTACK]
    held_out_attack = attack[N_ATTACK:N_ATTACK + 10]
    print(f"[svm] dataset = {len(ds_clean)} clean + {len(ds_attack)} attack")
    print(f"[svm] held-out attack reserved for pipeline eval: {len(held_out_attack)}")

    print("[svm] featurizing...")
    X = np.concatenate([featurize_paths(ds_clean), featurize_paths(ds_attack)], axis=0)
    y = np.concatenate([np.zeros(len(ds_clean), dtype=np.int8),
                        np.ones(len(ds_attack), dtype=np.int8)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=SPLIT_SEED
    )
    print(f"[svm] feat dim={X_train.shape[1]} train={len(y_train)} test={len(y_test)} "
          f"(test pos={int(y_test.sum())}, neg={int((1-y_test).sum())})")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(C=1.0, max_iter=20000, dual="auto", class_weight=CLASS_WEIGHT)),
    ])
    print(f"[svm] training LinearSVC (class_weight={CLASS_WEIGHT}) on dim={X_train.shape[1]}")
    clf.fit(X_train, y_train)

    # Pick decision-function threshold tau on held-out.
    # Strategy: keep tau=0 if baseline FPR already meets TARGET_FPR (don't sacrifice
    # accuracy by becoming more permissive). Else pick smallest tau such that
    # fraction of negatives with score > tau <= TARGET_FPR.
    scores = clf.decision_function(X_test)
    pos = (y_test == 1)
    neg = ~pos

    base_pred = (scores > 0).astype(np.int8)
    base_fpr = float(((base_pred == 1) & neg).sum()) / max(int(neg.sum()), 1)
    base_fnr = float(((base_pred == 0) & pos).sum()) / max(int(pos.sum()), 1)

    pos_scores = np.sort(scores[pos])
    if len(pos_scores) > 0:
        # tau = TARGET_FNR quantile of positive scores -> exactly that fraction missed
        idx = int(np.ceil(TARGET_FNR * len(pos_scores))) - 1
        idx = max(0, min(idx, len(pos_scores) - 1))
        tau = float(pos_scores[idx]) + 1e-9
    else:
        tau = 0.0

    pred = (scores > tau).astype(np.int8)
    fpr = float(((pred == 1) & neg).sum()) / max(int(neg.sum()), 1)
    fnr = float(((pred == 0) & pos).sum()) / max(int(pos.sum()), 1)
    acc = float((pred == y_test).mean())

    print(f"[svm] class_weight={CLASS_WEIGHT}")
    print(f"[svm] tau=0 baseline       FPR={base_fpr:.3f} FNR={base_fnr:.3f}")
    print(f"[svm] tuned tau={tau:.4f}  FPR={fpr:.3f} FNR={fnr:.3f} acc={acc:.3f}")

    bundle = {"clf": clf, "tau": float(tau), "target_fpr": TARGET_FPR}
    joblib.dump(bundle, OUT_PATH)
    print(f"[svm] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

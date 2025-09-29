#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Tuple, List
import hashlib
import random
import shutil
import pdb
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Ensure project root is on sys.path so we can import pipeline_zoo when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_zoo import utilities as U
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="Annotate boxes from model1 and model2 on saved output images")
    p.add_argument("--log", type=str, default=None, help="Path to a log file under logs/ (either log_...json or log_..._configs.json). Will infer the matching run_dir under output/ and the config.")
    p.add_argument("--run_dir", type=str, default=None, help="[Fallback] Specific run directory under output/ to process, e.g. output/dual_object_detection_20250928-134905")
    p.add_argument("--output_root", type=str, default="output", help="Root folder containing run directories")
    p.add_argument("--logs_root", type=str, default="logs", help="Logs root, used to locate corresponding *_configs.json")
    p.add_argument("--save_root", type=str, default="vis/annotated", help="Where to save annotated images")
    p.add_argument("--sample_count", type=int, default=10, help="Number of images to randomly sample per run (default 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--override_thresh1", type=float, default=None, help="Optional: override conf_threshold_1")
    p.add_argument("--override_thresh2", type=float, default=None, help="Optional: override conf_threshold_2")
    return p.parse_args()


def find_config_for_run(run_dir: Path, logs_root: Path) -> Path:
    """Given run_dir like output/dual_object_detection_YYYYMMDD-HHMMSS, find logs/log_dual_object_detection_YYYYMMDD-HHMMSS_configs.json"""
    name = run_dir.name
    if "_" not in name:
        return None
    stem, ts = name.rsplit("_", 1)
    cfg_name = f"log_{stem}_{ts}_configs.json"
    cfg_path = logs_root / cfg_name
    return cfg_path if cfg_path.exists() else None


def parse_stem_ts_from_log(log_path: Path):
    """Parse (stemname, timestamp) from log filename.
    Accepts both log_<stem>_<ts>.json and log_<stem>_<ts>_configs.json.
    """
    name = log_path.name
    # Expected patterns: log_dual_object_detection_20250928-134905.json or ..._configs.json
    # Strip prefix 'log_'
    if not name.startswith("log_"):
        return None, None
    base = name[len("log_"):]
    # Remove optional suffix
    if base.endswith("_configs.json"):
        base = base[:-len("_configs.json")]
    elif base.endswith(".json"):
        base = base[:-len(".json")]
    else:
        return None, None
    # Split by last underscore to separate timestamp
    if "_" not in base:
        return None, None
    stem, ts = base.rsplit("_", 1)
    return stem, ts


def run_dir_from_log(log_path: Path, output_root: Path) -> Path | None:
    stem, ts = parse_stem_ts_from_log(log_path)
    if stem is None or ts is None:
        return None
    return output_root / f"{stem}_{ts}"


def load_models_from_config(cfg: dict, device: torch.device):
    m1_name = cfg.get("hf_model_name_1")
    m2_name = cfg.get("hf_model_name_2")
    if not m1_name or not m2_name:
        raise RuntimeError("Config missing hf_model_name_1/2")
    model1 = RTDetrForObjectDetection.from_pretrained(m1_name).to(device)
    model2 = RTDetrForObjectDetection.from_pretrained(m2_name).to(device)
    proc1 = RTDetrImageProcessor.from_pretrained(m1_name)
    proc2 = RTDetrImageProcessor.from_pretrained(m2_name)
    nq1 = int(cfg.get("num_queries_1", model1.config.num_queries))
    nq2 = int(cfg.get("num_queries_2", model2.config.num_queries))
    model1.config.num_queries = nq1
    model2.config.num_queries = nq2
    model1.eval()
    model2.eval()
    return model1, model2, proc1, proc2


def to_tensor(proc: RTDetrImageProcessor, pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    return proc(images=pil_img, return_tensors="pt")["pixel_values"].to(device)


def boxes_from_outputs(outputs, logits_sigmoid: bool = True):
    """Return pred_boxes [Q,4 in cx,cy,w,h 0..1] and conf [Q] as max prob across classes."""
    pred_boxes = outputs.pred_boxes.squeeze(0).detach().cpu().numpy()  # [Q,4]
    logits = outputs.logits
    if logits_sigmoid:
        probs = torch.sigmoid(logits)
    else:
        probs = logits
    conf = probs.squeeze(0).detach().cpu().numpy().max(axis=-1)  # [Q]
    return pred_boxes, conf


def restricted_conf(probs: torch.Tensor, target_indices: List[int] | None) -> torch.Tensor:
    """Compute per-query confidence restricted to target class indices.
    - probs: [1,Q,C]
    - target_indices: list of class ids; if empty/None -> use all classes.
    Returns: [1,Q] tensor with max prob over the target set (or all if empty).
    """
    if (not target_indices) or len(target_indices) == 0:
        return probs.max(dim=2)[0]
    return probs[:, :, target_indices].max(dim=2)[0]


def cxcywh_to_xyxy(norm_box: np.ndarray, W: int, H: int) -> Tuple[int, int, int, int]:
    cx, cy, w, h = norm_box.tolist()
    x1 = int(np.floor((cx - w / 2.0) * W))
    y1 = int(np.floor((cy - h / 2.0) * H))
    x2 = int(np.ceil ((cx + w / 2.0) * W))
    y2 = int(np.ceil ((cy + h / 2.0) * H))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W,     x2))
    y2 = max(0, min(H,     y2))
    return x1, y1, x2, y2


def draw_boxes_cv(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    conf: np.ndarray,
    thresh: float,
    color: Tuple[int, int, int],
    label: str,
    cls_names: List[str] | None = None,
):
    H, W = img_bgr.shape[:2]
    for i, (b, s) in enumerate(zip(boxes, conf)):
        if float(s) < float(thresh):
            continue
        x1, y1, x2, y2 = cxcywh_to_xyxy(b, W, H)
        if x2 <= x1 or y2 <= y1:
            continue
        name = cls_names[i] if (cls_names is not None and i < len(cls_names)) else ""
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label}:{name} {s:.2f}" if name else f"{label}:{s:.2f}"
        cv2.putText(img_bgr, text, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def restricted_conf_with_idx(
    probs: torch.Tensor, target_indices: List[int] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (conf, cls_idx) where conf=max prob over target set; cls_idx in original class ids.
    probs: [1,Q,C]
    target_indices: list[int] or None/empty -> use all classes
    Returns: conf [1,Q], idx [1,Q]
    """
    if (not target_indices) or len(target_indices) == 0:
        vals, idx = probs.max(dim=2)
        return vals, idx
    sub = probs[:, :, target_indices]  # [1,Q,T]
    vals, idx_sub = sub.max(dim=2)
    tgt = torch.tensor(target_indices, device=probs.device, dtype=idx_sub.dtype)
    idx = tgt[idx_sub]
    return vals, idx


def _config_fingerprint(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def _clean_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def annotate_run(run_dir: Path, logs_root: Path, save_root: Path, sample_count: int = 10, seed: int = 42, override_t1: float = None, override_t2: float = None, cfg_path: Path | None = None):
    if cfg_path is None:
        cfg_path = find_config_for_run(run_dir, logs_root)
    if cfg_path is None:
        print(f"[warn] Config not found for {run_dir.name}; skipping.")
        return
    cfg = json.loads(Path(cfg_path).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1, model2, proc1, proc2 = load_models_from_config(cfg, device)
    # build id->name maps for pretty labels
    def _id2name_map(m):
        mapp = getattr(m.config, "id2label", None)
        if isinstance(mapp, dict):
            try:
                return {int(k): v for k, v in mapp.items()}
            except Exception:
                return {k: v for k, v in mapp.items()}
        return {}
    id2name1 = _id2name_map(model1)
    id2name2 = _id2name_map(model2)
    t1 = float(cfg.get("conf_threshold_1", 0.25) if override_t1 is None else override_t1)
    t2 = float(cfg.get("conf_threshold_2", 0.25) if override_t2 is None else override_t2)
    # target labels per model; draw only these
    tgt_labels = cfg.get("target_labels", [[], []])
    tgt1 = tgt_labels[0] if isinstance(tgt_labels, (list, tuple)) and len(tgt_labels) > 0 else []
    tgt2 = tgt_labels[1] if isinstance(tgt_labels, (list, tuple)) and len(tgt_labels) > 1 else []

    # Save directory naming: <pipeline>_<timestamp>
    # derive from run_dir name: <pipeline>_<timestamp>
    if "_" in run_dir.name:
        pipeline_name, ts = run_dir.name.rsplit("_", 1)
    else:
        pipeline_name, ts = run_dir.name, ""
    out_dir = save_root / (f"{pipeline_name}_{ts}" if ts else pipeline_name)
    # Always overwrite previous outputs for the same config
    _clean_dir(out_dir)
    subdir_m1 = out_dir / "m1"
    subdir_m2 = out_dir / "m2"
    subdir_m1.mkdir(parents=True, exist_ok=True)
    subdir_m2.mkdir(parents=True, exist_ok=True)

    # gather images
    images = sorted([p for p in run_dir.glob("*.png")])
    # random sampling
    if sample_count and sample_count > 0 and len(images) > sample_count:
        rnd = random.Random(seed)
        images = rnd.sample(images, sample_count)

    for img_path in images:
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] Failed to open {img_path.name}: {e}")
            continue

        # Prepare base images in BGR for separate drawings
        base_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img_m1 = base_bgr.copy()
        img_m2 = base_bgr.copy()

        # model 1 (full image)
        with torch.no_grad():
            px1 = to_tensor(proc1, pil, device)
            out1 = model1(px1, output_hidden_states=False)
        # build combined tensor to reuse ROI logic
        probs1 = torch.sigmoid(out1.logits)  # [1,Q,C]
        # restrict to target labels for model 1 when computing confidence and ROI
        conf1, cls1 = restricted_conf_with_idx(probs1, tgt1)  # [1,Q]
        combined1 = torch.cat([out1.pred_boxes, conf1.unsqueeze(-1), out1.logits], dim=-1)
        b1 = out1.pred_boxes.squeeze(0).detach().cpu().numpy()
        c1 = conf1.squeeze(0).detach().cpu().numpy()
        n1 = [id2name1.get(int(i), str(int(i))) for i in cls1.squeeze(0).detach().cpu().numpy().tolist()]
        draw_boxes_cv(img_m1, b1, c1, t1, color=(0, 0, 255), label="M1", cls_names=n1)  # red in BGR

        # model 2 (NOT parallel): run per-ROI derived from model1 detections
        # Build ROI masks from model1 predictions above t1
        roi_masks_list = U.masks_from_boxes(px1, combined1, t1)
        flat_masks = []
        for masks in roi_masks_list:
            flat_masks.extend(masks)

        # prepare accumulators for drawing
        b2_all = []  # list of (boxes, conf, names)
        if len(flat_masks) > 0:
            with torch.no_grad():
                Bf, C, H, W = px1.shape
                # Keep num_queries safe (optional):
                min_tokens = max(1, (H // 32) * (W // 32))
                safe_queries = max(1, min(int(model2.config.num_queries), min_tokens))
                if model2.config.num_queries != safe_queries:
                    model2.config.num_queries = safe_queries
                for m in flat_masks:
                    _m = m.to(px1.dtype).unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
                    masked = px1 * _m
                    out2 = model2(masked, output_hidden_states=False)
                    probs2 = torch.sigmoid(out2.logits)
                    # restrict to target labels for model 2 when drawing
                    conf2, cls2 = restricted_conf_with_idx(probs2, tgt2)
                    b2 = out2.pred_boxes.squeeze(0).detach().cpu().numpy()
                    c2 = conf2.squeeze(0).detach().cpu().numpy()
                    n2 = [id2name2.get(int(i), str(int(i))) for i in cls2.squeeze(0).detach().cpu().numpy().tolist()]
                    b2_all.append((b2, c2, n2))
        if b2_all:
            b2_cat = np.concatenate([b for (b, _, _) in b2_all], axis=0)
            c2_cat = np.concatenate([c for (_, c, _) in b2_all], axis=0)
            n2_cat = sum([n for (_, _, n) in b2_all], start=[])
            draw_boxes_cv(img_m2, b2_cat, c2_cat, t2, color=(255, 0, 0), label="M2", cls_names=n2_cat)  # blue in BGR

        # save separately
        out_path_m1 = subdir_m1 / img_path.name
        out_path_m2 = subdir_m2 / img_path.name
        cv2.imwrite(str(out_path_m1), img_m1)
        cv2.imwrite(str(out_path_m2), img_m2)
        print(f"Saved {out_path_m1} and {out_path_m2}")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    logs_root = Path(args.logs_root)
    save_root = Path(args.save_root)

    # Preferred path: user provides a log file and we infer the run_dir and config.
    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            return
        # Derive run_dir and cfg_path from the provided log
        rd = run_dir_from_log(log_path, output_root)
        if rd is None or not rd.exists():
            print(f"Failed to infer run_dir from log or directory does not exist: {log_path}")
            return
        # Determine config path: if provided log is configs.json, use it; else map to the sibling *_configs.json
        if log_path.name.endswith("_configs.json"):
            cfg_path = log_path
        else:
            stem, ts = parse_stem_ts_from_log(log_path)
            cfg_name = f"log_{stem}_{ts}_configs.json"
            cfg_path = log_path.parent / cfg_name
        annotate_run(
            rd,
            logs_root,
            save_root,
            sample_count=int(args.sample_count) if args.sample_count else 10,
            seed=int(args.seed) if args.seed is not None else 42,
            override_t1=args.override_thresh1,
            override_t2=args.override_thresh2,
            cfg_path=cfg_path,
        )
        return

    # Fallbacks kept for backward compatibility
    if args.run_dir:
        rd = Path(args.run_dir)
        annotate_run(
            rd,
            logs_root,
            save_root,
            sample_count=int(args.sample_count) if args.sample_count else 10,
            seed=int(args.seed) if args.seed is not None else 42,
            override_t1=args.override_thresh1,
            override_t2=args.override_thresh2,
        )
        return

    # If nothing provided, scan all run dirs (legacy behavior)
    runs: List[Path] = [p for p in output_root.glob("*") if p.is_dir()]
    runs = sorted(runs)
    if not runs:
        print(f"No run directories found under {output_root}")
        return
    for rd in runs:
        annotate_run(
            rd,
            logs_root,
            save_root,
            sample_count=int(args.sample_count) if args.sample_count else 10,
            seed=int(args.seed) if args.seed is not None else 42,
            override_t1=args.override_thresh1,
            override_t2=args.override_thresh2,
        )


if __name__ == "__main__":
    main()

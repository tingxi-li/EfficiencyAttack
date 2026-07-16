"""Build a 150-image clean pool from COCO val for input-defense experiments.

Output format matches `ultra_fast_save` in base_attack.py so that
`ultra_fast_load` in c1_img.py can read it identically to attacked images.
"""
import os
import sys
import glob
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import RTDetrImageProcessor

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
sys.path.append(PROJ)
from utils import set_all_seeds, parse_example  # noqa: E402

import argparse

ATTACK_DIR = os.path.join(PROJ, "saved", "model_0", "teaspoon_tgt_2")
DEFAULT_OUT_DIR = os.path.join(PROJ, "saved", "clean_pool")
DEFAULT_N = 150
SEED = 1


def ultra_fast_save(tensor, filepath):
    t = tensor.detach().cpu().contiguous()
    shape = t.shape
    shape_dims = np.array([len(shape)], dtype=np.int8).tobytes()
    shape_header = np.array(shape, dtype=np.int64).tobytes()
    dtype_str = np.dtype(t.numpy().dtype).str.encode("ascii")
    dtype_length = np.array([len(dtype_str)], dtype=np.int8).tobytes()
    data_bytes = t.numpy().tobytes()
    with open(filepath, "wb") as f:
        fd = f.fileno()
        os.write(fd, shape_dims)
        os.write(fd, shape_header)
        os.write(fd, dtype_length)
        os.write(fd, dtype_str)
        os.write(fd, data_bytes)
        os.fsync(fd)


def already_attacked_ids():
    excl = set()
    for f in glob.glob(os.path.join(ATTACK_DIR, "img_id_*.pt")):
        b = os.path.basename(f).replace(".pt", "")
        parts = b.split("_")
        if len(parts) >= 3:
            excl.add(parts[2])
    return excl


def existing_ids(out_dir):
    """image_ids already saved in out_dir (so we never overwrite)."""
    have = set()
    for f in glob.glob(os.path.join(out_dir, "img_id_*.pt")):
        b = os.path.basename(f).replace(".pt", "")
        parts = b.split("_")
        if len(parts) >= 3:
            have.add(parts[2])
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--exclude_dirs", nargs="*", default=[],
                    help="extra clean dirs whose image_ids must not be reused")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    excl = already_attacked_ids()
    have = existing_ids(args.out_dir)
    extra_excl = set()
    for d in args.exclude_dirs:
        extra_excl |= existing_ids(d)
    excl = excl | extra_excl
    print(f"[build_clean_pool] excluding {len(excl)} ids "
          f"({len(extra_excl)} from extra dirs); already have {len(have)} in {args.out_dir}")

    set_all_seeds(SEED)
    coco = load_dataset("detection-datasets/coco", split="val")
    indices = random.sample(range(len(coco)), len(coco))

    processor = None
    saved = len(have)
    for idx in indices:
        if saved >= args.n:
            break
        ex = coco[idx]
        image_id, image, *_ = parse_example(ex)
        if str(image_id) in excl or str(image_id) in have:
            continue
        if processor is None:
            processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        out = os.path.join(args.out_dir, f"img_id_{image_id}_clean.pt")
        ultra_fast_save(pixel_values, out)
        have.add(str(image_id))
        saved += 1
        if saved % 50 == 0:
            print(f"[build_clean_pool] saved {saved}/{args.n}")

    print(f"[build_clean_pool] DONE. {saved} images at {args.out_dir}")


if __name__ == "__main__":
    main()

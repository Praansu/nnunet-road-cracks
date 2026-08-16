#!/usr/bin/env python3
"""
Step 2: Prepare the raw dataset for nnU-Net.

- Finds matching image/mask pairs in two folders (any common format: jpg, png, bmp, tiff...)
- Converts everything to .png
- Uses the classes from classes.json (0 = background; 255 in masks is converted
  to the first non-background class automatically)
- Resizes very large images down (auto: no resize on GPUs with 10+ GB VRAM,
  otherwise longest side max 1536 px - thin cracks need all the resolution they can get)
- Splits data into train (90%) and test (10%)
- Builds the nnUNet_raw/Dataset001_RoadCracks folder that nnU-Net expects

All paths and settings come from config.py (and classes.json) - edit those
files instead of this script.

Usage:
  python prepare_dataset.py --images C:\\path\\to\\images --masks C:\\path\\to\\masks
  (or run without arguments and follow the prompts)
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image

import config

# Common name decorations found on mask files (e.g. "photo_mask.png" vs "photo.jpg")
AFFIXES = [
    "_mask", "-mask", "mask_", "mask-",
    "_label", "-label", "label_", "label-",
    "_gt", "-gt", "_seg", "-seg", "seg_", "seg-",
    "_anno", "-anno", "_annotation", "-annotation",
    "_truth", "-truth", "_result", "-result",
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def normalized_name(stem: str) -> str:
    """Lowercase stem with common mask prefixes/suffixes removed, so that
    'crack123_mask' and 'crack123' still match each other."""
    s = stem.lower()
    for a in AFFIXES:
        if s.endswith(a):
            s = s[: -len(a)]
            break
        if s.startswith(a):
            s = s[len(a):]
            break
    return s


def collect_files(folder: str, what: str):
    if not folder or not os.path.isdir(folder):
        raise SystemExit(f"ERROR: {what} folder not found: {folder!r}")
    files = {}
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTS):
            files[name.lower()] = os.path.join(folder, name)
    if not files:
        raise SystemExit(
            f"ERROR: no images found in {what} folder {folder!r} "
            f"(supported: png, jpg, jpeg, bmp, tif, tiff, webp)"
        )
    return files


def find_pairs(images_dir: str, masks_dir: str):
    images = collect_files(images_dir, "image")
    masks = collect_files(masks_dir, "mask")

    exact_masks = {os.path.splitext(name)[0]: path for name, path in masks.items()}
    norm_masks = {}
    for name, path in masks.items():
        norm_masks.setdefault(normalized_name(os.path.splitext(name)[0]), []).append(path)

    pairs = []
    warnings = []
    for iname, ipath in images.items():
        stem = os.path.splitext(iname)[0]
        candidates = []
        if stem in exact_masks:
            candidates = [exact_masks[stem]]
        else:
            n = normalized_name(stem)
            candidates = norm_masks.get(n, [])
        if len(candidates) == 1:
            pairs.append((ipath, candidates[0]))
        elif len(candidates) > 1:
            warnings.append(f"AMBIGUOUS: image {iname!r} matches several masks -> skipped")
        else:
            warnings.append(f"NO MATCH: image {iname!r} has no mask -> skipped")

    for mname in sorted(masks):
        mstem = os.path.splitext(mname)[0]
        found = any(os.path.basename(mpath).lower() == mname for _, mpath in pairs)
        if not found:
            # mask used already?
            warnings.append(f"UNUSED MASK: {mname!r} (no matching image) -> skipped")

    if not pairs:
        raise SystemExit(
            "ERROR: no matching image/mask pairs found. Make sure every image has a mask "
            "with the same file name (e.g. photo.jpg + photo.png or photo.jpg + photo_mask.png)."
        )
    return pairs, warnings


def normalize_mask(mask: np.ndarray, label_ids) -> np.ndarray:
    """Map raw mask values to the class ids from classes.json. 0 = background.
    - {0, 255} masks: 255 -> the first non-background class id
    - anything else is kept as-is; unknown ids are reported by the caller
      (so a pothole id 2 never silently becomes crack id 1)"""
    vals = np.unique(mask)
    if vals.size == 1 and vals[0] == 0:
        return mask.astype(np.uint8)  # all-background mask, keep as is
    if set(vals.tolist()) <= {0, 255}:
        target = next((v for v in sorted(label_ids) if v > 0), 1)
        return (mask > 0).astype(np.uint8) * target
    return mask.astype(np.uint8)  # already class ids - keep them


def sanitize_stem(stem: str) -> str:
    stem = re.sub(r"[^\w\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "image"


def auto_max_size():
    """Pick the resize cap based on the GPU: big VRAM -> keep native resolution
    (best for thin cracks), small VRAM -> 1536 px cap so training fits.
    Thresholds come from config.py."""
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return config.MAX_SIZE_BIG_GPU if vram >= config.BIG_GPU_VRAM_GB else config.MAX_SIZE_SMALL_GPU
    except Exception:
        pass
    return config.MAX_SIZE_SMALL_GPU


def load_and_resize(path: str, max_size: int, is_mask: bool):
    img = Image.open(path)
    if is_mask:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    w, h = img.size
    if max_size and max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
        img = img.resize((nw, nh), resample)
    return img


def main():
    ap = argparse.ArgumentParser(description="Prepare road-crack dataset for nnU-Net")
    ap.add_argument("--images", help="Folder containing the photos")
    ap.add_argument("--masks", help="Folder containing the masks")
    ap.add_argument("--out", default=config.NNUNET_RAW,
                    help="Where to put the prepared dataset (default: ./nnUNet_raw)")
    ap.add_argument("--max-size", type=int, default=config.MAX_SIZE,
                    help="Resize images longer than this many pixels (0 = never resize; "
                         "default: auto - no resize on GPUs with 10+ GB VRAM, else 1536)")
    ap.add_argument("--test-size", type=float, default=config.TEST_SIZE,
                    help="Fraction of images held out as final test set (default 0.10)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    if args.max_size is None:
        args.max_size = auto_max_size()

    print("=" * 60)
    print("  STEP 2/4 - DATASET PREPARATION")
    print("=" * 60)

    images_dir = args.images
    masks_dir = args.masks
    while not images_dir:
        images_dir = input("Full path to the folder with the photos: ").strip().strip('"')
    while not masks_dir:
        masks_dir = input("Full path to the folder with the masks: ").strip().strip('"')

    images_dir = os.path.abspath(images_dir)
    masks_dir = os.path.abspath(masks_dir)

    print("\n[1/5] Matching images with masks ...")
    pairs, warnings = find_pairs(images_dir, masks_dir)
    for w in warnings:
        print("  !", w)
    print(f"  -> {len(pairs)} matching image/mask pairs found")

    if len(pairs) < 10:
        print("  WARNING: fewer than 10 pairs - nnU-Net works, but results will be limited")

    # --- split into train/test -------------------------------------------
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    n_test = max(1, int(round(len(pairs) * args.test_size)))
    test_idx = set(idx[:n_test].tolist())

    # --- build nnU-Net folder structure -----------------------------------
    dataset_dir = os.path.join(args.out, config.DATASET_NAME)
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")
    for d in (images_tr, labels_tr, images_ts):
        os.makedirs(d, exist_ok=True)

    print(f"\n[2/5] Converting to PNG (max size {args.max_size or 'unlimited'} px) ...")
    label_ids = sorted(config.LABELS.values())
    all_classes = set()
    n_train = n_test_actual = 0
    for i, (ipath, mpath) in enumerate(pairs, 1):
        try:
            stem = sanitize_stem(os.path.splitext(os.path.basename(ipath))[0])
            image = load_and_resize(ipath, args.max_size, is_mask=False)
            mask_img = load_and_resize(mpath, args.max_size, is_mask=True)
            mask = np.array(mask_img)
            mask = normalize_mask(mask, label_ids)
            mask_img = Image.fromarray(mask, mode="L")
            if mask_img.size != image.size:
                mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
            all_classes.update(np.unique(mask).tolist())

            is_test = i - 1 in test_idx
            if is_test:
                image.save(os.path.join(images_ts, f"{stem}_0000.png"))
                mask_img.save(os.path.join(images_ts, f"{stem}.png"))
                n_test_actual += 1
            else:
                image.save(os.path.join(images_tr, f"{stem}_0000.png"))
                mask_img.save(os.path.join(labels_tr, f"{stem}.png"))
                n_train += 1
        except Exception as e:
            print(f"  ! FAILED on {os.path.basename(ipath)}: {e}")
        if i % 500 == 0:
            print(f"  ... {i}/{len(pairs)} done")

    if n_train == 0:
        raise SystemExit("ERROR: nothing was prepared - check the warnings above")

    classes = sorted(all_classes)
    label_names = dict(config.LABELS)  # from classes.json
    missing = set(classes) - set(label_names.values())
    for v in sorted(missing):
        label_names[f"class_{v}"] = v
        print(f"  ! masks contain class id {v} which is not in classes.json - "
              f"added as 'class_{v}' (edit classes.json to name it properly)")
    unseen = set(label_names.values()) - set(classes)
    for v in sorted(unseen):
        name = next((n for n, i in label_names.items() if i == v), str(v))
        print(f"  NOTE: class '{name}' (id {v}) from classes.json never appears in the masks")
    ids = sorted(label_names.values())
    if ids != list(range(len(ids))):
        print("  WARNING: class ids in classes.json are not consecutive (0,1,2,...) - "
              "nnU-Net works best with consecutive ids")

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": label_names,
        "numTraining": n_train,
        "file_ending": ".png",
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    # --- summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Train images : {n_train}")
    print(f"  Test images  : {n_test_actual}")
    print(f"  Classes found: {label_names}")
    print(f"  Dataset ready at: {dataset_dir}")
    print("\n  Next: run  python train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

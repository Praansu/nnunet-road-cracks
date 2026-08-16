#!/usr/bin/env python3
"""
Step 4: Predict on the held-out test images and export:
  outputs/predictions/  -> clean prediction masks (PNG, 0=background, 1=crack, ...)
  outputs/overlays/     -> the photos with predicted cracks drawn in red (easy to eyeball)

Usage:
  python predict_and_export.py            (uses fold 0)
  python predict_and_export.py --best     (5-fold ensemble + postprocessing + test-time
                                           augmentation - the most accurate predictions)

All paths and settings come from config.py (and classes.json) - edit those
files instead of this script.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

import config

config.setup_environment()

DICE_RE = re.compile(r"val[_\-\s]*dice[:\s]*([0-9]*\.?[0-9]+)", re.I)


def find_cli(name):
    for cand in (name, name + ".exe"):
        p = shutil.which(cand)
        if p:
            return p
    return None


def last_val_dice_from_log(fold_dir):
    """Best-effort: read the last validation Dice printed during training."""
    logs = sorted(glob.glob(os.path.join(fold_dir, "training_log*.txt")))
    best = None
    for log in logs:
        try:
            with open(log, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = DICE_RE.search(line)
                    if m:
                        best = float(m.group(1))
        except OSError:
            continue
    return best


def main():
    ap = argparse.ArgumentParser(description="Predict + export road-crack results")
    ap.add_argument("--best", action="store_true",
                    help="Use the 5-fold ensemble (train with --best first)")
    args = ap.parse_args()

    print("=" * 60)
    print("  STEP 4/4 - PREDICTION + EXPORT")
    print("=" * 60)

    cli_predict = find_cli("nnUNetv2_predict")
    if not cli_predict:
        raise SystemExit("nnU-Net commands not found - run: pip install nnunetv2")

    dataset_dir = os.path.join(os.environ["nnUNet_raw"], config.DATASET_NAME)
    images_ts = os.path.join(dataset_dir, "imagesTs")
    if not os.path.isdir(images_ts) or not glob.glob(os.path.join(images_ts, "*.png")):
        raise SystemExit("No test images found - run prepare_dataset.py first.")

    results_root = os.path.join(os.environ["nnUNet_results"], config.DATASET_NAME, "2d")
    folds = []
    for f in range(5):
        if os.path.isfile(os.path.join(results_root, f"fold_{f}", "checkpoint_final.pth")):
            folds.append(f)
    if not folds:
        raise SystemExit("No trained model found - run train_model.py first.")
    folds = folds if args.best else [folds[0]]

    # --- 1. run prediction ---------------------------------------------------
    pred_raw = os.path.join(config.OUTPUTS_DIR, "predictions_raw")
    shutil.rmtree(pred_raw, ignore_errors=True)
    os.makedirs(pred_raw, exist_ok=True)

    cmd = [cli_predict, "-i", images_ts, "-o", pred_raw, "-d", config.DATASET_ID, "-c", "2d",
           "-f"] + [str(f) for f in folds]
    if args.best:
        cmd.append("--use_tta")  # test-time augmentation: free accuracy boost
        print("  Test-time augmentation ON (slower but more accurate)")
    print(">> " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"ERROR: prediction failed (exit code {r.returncode})")

    # --- 2. apply postprocessing if the best-config search produced it ------
    final_pred = pred_raw
    pp_pkl = None
    for cand in glob.glob(os.path.join(os.environ["nnUNet_results"], config.DATASET_NAME, "**", "postprocessing.pkl"),
                          recursive=True):
        pp_pkl = cand
        break
    if pp_pkl:
        pred_pp = os.path.join(config.OUTPUTS_DIR, "predictions_pp")
        shutil.rmtree(pred_pp, ignore_errors=True)
        os.makedirs(pred_pp, exist_ok=True)
        cli_pp = find_cli("nnUNetv2_apply_postprocessing")
        if cli_pp:
            print("\n  Applying postprocessing ...")
            cmd = [cli_pp, "-i", pred_raw, "-o", pred_pp, "-pp_pkl_file", pp_pkl]
            if subprocess.run(cmd).returncode == 0:
                final_pred = pred_pp
        else:
            print("  (postprocessing command missing - skipping it, raw predictions are fine)")

    # --- 3. export masks + overlays ------------------------------------------
    pred_dir = os.path.join(config.OUTPUTS_DIR, "predictions")
    ovl_dir = os.path.join(config.OUTPUTS_DIR, "overlays")
    for d in (pred_dir, ovl_dir):
        os.makedirs(d, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(final_pred, "*.png")))
    if not pred_files:
        raise SystemExit("Prediction produced no files - something went wrong.")

    print(f"\n  Exporting {len(pred_files)} results ...")
    exported = 0
    for pf in pred_files:
        stem = os.path.splitext(os.path.basename(pf))[0]
        if stem.endswith("_0000"):
            stem = stem[: -5]
        mask = np.array(Image.open(pf).convert("L"))
        Image.fromarray(mask, mode="L").save(os.path.join(pred_dir, f"{stem}.png"))

        src = os.path.join(images_ts, f"{stem}_0000.png")
        if os.path.isfile(src):
            img = np.array(Image.open(src).convert("RGB")).astype(np.float32)
            blend = img.copy()
            blend[mask > 0] = (blend[mask > 0] * (1.0 - config.OVERLAY_STRENGTH)
                               + np.array(config.OVERLAY_COLOR) * config.OVERLAY_STRENGTH)
            Image.fromarray(blend.astype(np.uint8)).save(os.path.join(ovl_dir, f"{stem}.png"))
        exported += 1

    # --- 4. report -------------------------------------------------------------
    dice = last_val_dice_from_log(os.path.join(results_root, f"fold_{folds[0]}"))
    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    print(f"  Test images processed : {exported}")
    print(f"  Model used            : folds {folds}" + (" (ensemble)" if len(folds) > 1 else ""))
    if dice is not None:
        print(f"  Expected Dice on new images : ~{dice:.3f}  (0.6-0.85 is normal for thin cracks)")
    print("\n  Where to look:")
    print(f"    {pred_dir}   <- clean prediction masks (open in any image viewer)")
    print(f"    {ovl_dir}    <- photos with cracks drawn in red (visual check)")
    print("\n  Reminder: for thin cracks, ALWAYS judge by eye too - the Dice number")
    print("  is strict. If the red lines follow the real cracks, the model works.")
    print("=" * 60)


if __name__ == "__main__":
    main()

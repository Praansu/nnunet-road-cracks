#!/usr/bin/env python3
"""
Step 3: Plan, preprocess and train the nnU-Net model (2D).

- Auto-detects your GPU and plans the best patch size for it
- Trains fold 0 first (fastest way to a good model)
- Watches validation performance and stops early when it plateaus
- Rerun the same command anytime to RESUME training

For the absolute best results (5x training time, ~+1-3% Dice):
  python train_model.py --best         (all 5 folds, 1000-epoch cap + early stop,
                                        then ensemble search)

Usage:
  python train_model.py                (train fold 0, ~250 epochs with early stop)
  python train_model.py --epochs 500   (more epochs = better but slower)
  python train_model.py --best         (best results: 5 folds, 1000 epochs, ensemble)
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("nnUNet_raw", os.path.join(BASE, "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(BASE, "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", os.path.join(BASE, "nnUNet_results"))

# Windows-friendly worker counts (spawn-based multiprocessing is safer with few workers)
if os.name == "nt":
    os.environ.setdefault("nnUNet_n_proc_DA", "2")
    os.environ.setdefault("nnUNet_def_n_proc", "2")

DATASET = "Dataset001_RoadCracks"
TRAINERS = {250: "nnUNetTrainer_250epochs",
            500: "nnUNetTrainer_500epochs",
            1000: "nnUNetTrainer_1000epochs"}

EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.I)
DICE_RES = [
    re.compile(r"val[_\-\s]*dice[:\s]*([0-9]*\.?[0-9]+)", re.I),
    re.compile(r"pseudo[_\-\s]*dice[:\s]*([0-9]*\.?[0-9]+)", re.I),
    re.compile(r"val[_\-\s]*dice[_\-\s]*avg[:\s]*([0-9]*\.?[0-9]+)", re.I),
]


def find_cli(name):
    for cand in (name, name + ".exe"):
        p = shutil.which(cand)
        if p:
            return p
    return None


def report_gpu():
    try:
        import torch
    except ImportError:
        print("  NOTE: torch not importable yet (dependencies may still be installing).")
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU found : {name} ({vram:.1f} GB VRAM)")
        if vram < 4:
            print("  WARNING: less than 4 GB VRAM - training may fail with out-of-memory.")
            print("           If it does, close other programs and retry.")
    else:
        print("  WARNING: NO NVIDIA GPU detected!")
        print("           Training will be extremely slow on CPU only (days, not hours).")
        print("           It will still work, but a GPU is strongly recommended.")


class EarlyStopper:
    """Watches the nnU-Net training log and stops training once the validation
    Dice stops improving for a while."""

    def __init__(self, fold_dir, patience=40, min_epochs=80):
        self.fold_dir = fold_dir
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_dice = -1.0
        self.best_epoch = 0
        self.last_epoch = 0
        self.stop = False
        self._seen_bytes = {}
        self._stop_lock = threading.Lock()

    def _log_files(self):
        return sorted(glob.glob(os.path.join(self.fold_dir, "training_log*.txt")))

    def request_stop(self):
        with self._stop_lock:
            self.stop = True

    def should_stop(self):
        with self._stop_lock:
            return self.stop

    def parse_new_lines(self):
        for log in self._log_files():
            try:
                with open(log, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(self._seen_bytes.get(log, 0))
                    data = f.read()
                    self._seen_bytes[log] = f.tell()
            except OSError:
                continue
            for line in data.splitlines():
                m = EPOCH_RE.search(line)
                if not m:
                    continue
                epoch = int(m.group(1))
                self.last_epoch = max(self.last_epoch, epoch)
                dice = None
                for r in DICE_RES:
                    dm = r.search(line)
                    if dm:
                        dice = float(dm.group(1))
                        break
                if dice is None:
                    continue
                if dice > self.best_dice:
                    self.best_dice = dice
                    self.best_epoch = epoch
                    print(f"    [early-stop] epoch {epoch}: best val Dice so far = {dice:.4f}")
                elif (epoch >= self.min_epochs
                      and epoch - self.best_epoch >= self.patience):
                    print(f"    [early-stop] no improvement for {self.patience} epochs "
                          f"(best {self.best_dice:.4f} at epoch {self.best_epoch}). Stopping.")
                    self.request_stop()
                    return

    def run(self, proc):
        while proc.poll() is None and not self.should_stop():
            try:
                self.parse_new_lines()
            except Exception:
                pass  # never let the watcher kill training
            time.sleep(15)
        if self.should_stop() and proc.poll() is None:
            print("  Finishing current epoch, then stopping cleanly ...")
            time.sleep(30)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()


def run_cmd(cmd, desc, allow_fail=False):
    print(f"\n>> {desc}")
    print(">> " + " ".join(cmd))
    try:
        r = subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    if r.returncode != 0 and not allow_fail:
        raise SystemExit(f"ERROR: {desc} failed (exit code {r.returncode}). See messages above.")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="Train nnU-Net 2D on road cracks")
    ap.add_argument("--epochs", type=int, choices=[250, 500, 1000], default=None,
                    help="Training length per fold (default: 250, or 1000 with --best)")
    ap.add_argument("--folds", default="0",
                    help="Comma-separated folds to train, e.g. '0' or '0,1,2' (default: 0)")
    ap.add_argument("--best", action="store_true",
                    help="Train all 5 folds and run ensemble search (best results, ~5x time)")
    args = ap.parse_args()
    if args.epochs is None:
        args.epochs = 1000 if args.best else 250

    print("=" * 60)
    print("  STEP 3/4 - TRAINING")
    print("=" * 60)

    dataset_dir = os.path.join(os.environ["nnUNet_raw"], DATASET)
    if not os.path.isfile(os.path.join(dataset_dir, "dataset.json")):
        raise SystemExit(
            "Dataset not found - run prepare_dataset.py first "
            "(or check the nnUNet_raw folder)."
        )
    with open(os.path.join(dataset_dir, "dataset.json"), encoding="utf-8") as f:
        import json
        n_train = json.load(f)["numTraining"]
    print(f"  Dataset: {n_train} training images")

    report_gpu()

    cli_plan = find_cli("nnUNetv2_plan_and_preprocess")
    cli_train = find_cli("nnUNetv2_train")
    cli_best = find_cli("nnUNetv2_find_best_configuration")
    if not cli_plan or not cli_train:
        raise SystemExit(
            "nnU-Net is not installed correctly (could not find its commands). "
            "Run: pip install nnunetv2"
        )

    # --- 1. plan + preprocess ---------------------------------------------
    run_cmd([cli_plan, "-d", "1", "-c", "2d"], "Planning + preprocessing (one-time, takes a while)")

    trainer = TRAINERS[args.epochs]
    folds = [0, 1, 2, 3, 4] if args.best else [int(f) for f in args.folds.split(",") if f.strip()]

    # --- 2. train folds ----------------------------------------------------
    for fold in folds:
        fold_dir = os.path.join(os.environ["nnUNet_results"], DATASET, "2d", f"fold_{fold}")
        final_ckpt = os.path.join(fold_dir, "checkpoint_final.pth")
        if os.path.isfile(final_ckpt):
            print(f"\n  fold {fold} already fully trained - skipping.")
            continue
        if os.path.isdir(fold_dir) and glob.glob(os.path.join(fold_dir, "checkpoint_latest.pth")):
            print(f"\n  fold {fold} has a saved checkpoint - training will RESUME automatically.")

        print(f"\n  Training fold {fold} ({trainer}) - this is the long step.")
        print("  Typical pace: 1-3 min/epoch. Watch the lines below.")
        cmd = [cli_train, "-d", "1", "-c", "2d", "-f", str(fold), "-tr", trainer]
        proc = subprocess.Popen(cmd)
        stopper = EarlyStopper(fold_dir)
        watcher = threading.Thread(target=stopper.run, args=(proc,), daemon=True)
        watcher.start()
        watcher.join()
        if proc.returncode != 0 and not stopper.should_stop():
            raise SystemExit(f"ERROR: training fold {fold} failed (exit code {proc.returncode})")
        if stopper.should_stop():
            print(f"\n  Fold {fold} stopped early (best val Dice {stopper.best_dice:.4f}).")
            print("  Rerun the same command anytime to continue training from where it stopped.")

    # --- 3. optional: find best configuration (ensemble) -------------------
    if args.best and cli_best:
        run_cmd([cli_best, "-d", "1", "-c", "2d"],
                "Searching the best combination of folds (ensembling)",
                allow_fail=True)

    print("\n" + "=" * 60)
    print("  TRAINING DONE")
    print("  Next: run  python predict_and_export.py" + (" --best" if args.best else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()

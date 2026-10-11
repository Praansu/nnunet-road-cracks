#!/usr/bin/env bash
# Road Crack Segmentation - nnU-Net: runs everything (install, prepare, train, predict).
# Linux/Mac equivalent of run_all.bat. First make it executable once:
#   chmod +x run_all.sh
set -u
cd "$(dirname "$0")"

PY=""
if command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "ERROR: Python was not found."
  echo "Install Python 3.10+ and rerun this script."
  exit 1
fi

fail() { echo ""; echo "$1"; exit 1; }

echo "============================================"
echo "  Road Crack Segmentation - nnU-Net"
echo "  Runs everything: install, prepare,"
echo "  train, predict."
echo "============================================"
echo ""

echo "[1/4] Installing required software (first run only - can take 10+ minutes)..."
$PY -m pip install --upgrade pip >/dev/null 2>&1
$PY -m pip install -r requirements.txt || fail "Install failed. Check your internet connection and try again."
echo ""
echo "[2/4] Preparing the dataset..."
echo "      You will be asked for the folders with the photos and the masks."
$PY prepare_dataset.py || fail "Dataset preparation failed. Read the message above."
echo ""
echo "[3/4] Training the model (the long step - plan for hours)..."
$PY train_model.py || fail "Training failed. Read the message above."
echo ""
echo "[4/4] Predicting on test images and creating overlays..."
$PY predict_and_export.py || fail "Prediction failed. Read the message above."
echo ""
echo "============================================"
echo "  ALL DONE!"
echo "  Open the outputs folder:"
echo "    outputs/predictions  - the crack masks"
echo "    outputs/overlays     - photos with cracks"
echo "                           drawn in red"
echo "============================================"

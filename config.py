"""
config.py - ALL settings for the road-crack nnU-Net pipeline in one place.

Edit THIS file (and classes.json) to change paths, classes, or training
settings. Every script in this package reads its settings from here, so you
never have to touch the .py scripts themselves.

Quick start:
  1. Set IMAGES_DIR and MASKS_DIR below (or leave None - you will be asked).
  2. Edit classes.json if your masks have more classes than background + crack.
  3. Run run_all.bat (or the scripts manually).
"""

import json
import os

# ---------------------------------------------------------------------------
# 1. INPUT FOLDERS - where the raw photos and masks are
#    Leave as None to be asked at startup. Use raw strings for Windows paths,
#    e.g.  IMAGES_DIR = r"C:\Users\me\Desktop\crack_photos"
# ---------------------------------------------------------------------------
IMAGES_DIR = None   # folder with the photos
MASKS_DIR = None    # folder with the matching masks

# ---------------------------------------------------------------------------
# 2. OUTPUT FOLDERS - where prepared data, models and results go
#    Defaults: subfolders next to this file. Change only if you know why.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NNUNET_RAW = os.path.join(BASE_DIR, "nnUNet_raw")                 # prepared dataset
NNUNET_PREPROCESSED = os.path.join(BASE_DIR, "nnUNet_preprocessed")  # nnU-Net cache
NNUNET_RESULTS = os.path.join(BASE_DIR, "nnUNet_results")         # trained models
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")                   # final results

# ---------------------------------------------------------------------------
# 3. DATASET IDENTITY - the folder name nnU-Net uses for this dataset.
#    Change DATASET_ID if you ever run a second dataset on the same machine.
# ---------------------------------------------------------------------------
DATASET_ID = "1"
DATASET_SUFFIX = "RoadCracks"
DATASET_NAME = f"Dataset{int(DATASET_ID):03d}_{DATASET_SUFFIX}"

# ---------------------------------------------------------------------------
# 4. CLASSES & LABELS - loaded from classes.json (edit THAT file, not here!)
#    Example: {"background": 0, "crack": 1, "pothole": 2}
# ---------------------------------------------------------------------------
CLASSES_JSON = os.path.join(BASE_DIR, "classes.json")


def load_classes():
    """Read classes.json -> {"background": 0, "crack": 1, ...}
    Keys starting with '_' are treated as comments and ignored.
    Falls back to background + crack if the file is missing or broken."""
    default = {"background": 0, "crack": 1}
    try:
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            labels = {str(k): int(v) for k, v in data.items() if not str(k).startswith("_")}
            if labels:
                return labels
    except Exception:
        pass
    return default


LABELS = load_classes()  # dict: label name -> class id (0 = background)

# ---------------------------------------------------------------------------
# 5. TRAINING SETTINGS
# ---------------------------------------------------------------------------
EPOCHS_DEFAULT = 250        # epochs per fold for a normal run
EPOCHS_BEST = 1000          # epochs per fold in --best mode (best results)
FOLDS = [0, 1, 2, 3, 4]     # the 5 cross-validation folds (--best trains all)
EARLY_STOP_PATIENCE = 40    # stop training if no improvement for this many epochs
EARLY_STOP_MIN_EPOCHS = 80  # never stop before this many epochs

# ---------------------------------------------------------------------------
# 6. DATA PREPARATION SETTINGS
# ---------------------------------------------------------------------------
TEST_SIZE = 0.10            # fraction of images held out as the final test set
SEED = 42                   # random split seed (keep fixed for reproducible splits)
MAX_SIZE = None             # None = auto (see below); 0 = never resize; e.g. 1536
MAX_SIZE_BIG_GPU = 0        # auto value on GPUs with lots of VRAM (0 = keep native resolution)
MAX_SIZE_SMALL_GPU = 1536   # auto value on smaller GPUs (px, longest side)
BIG_GPU_VRAM_GB = 10.0      # VRAM threshold for the auto choice above

# ---------------------------------------------------------------------------
# 7. VISUALIZATION - how cracks are drawn on the photos in outputs/overlays
# ---------------------------------------------------------------------------
OVERLAY_COLOR = (255, 40, 40)   # RGB color of the drawn cracks (red)
OVERLAY_STRENGTH = 0.55         # 0 = invisible, 1 = solid color

# ---------------------------------------------------------------------------
# 8. ENVIRONMENT - nnU-Net needs these variables; normally you don't touch them
# ---------------------------------------------------------------------------
def setup_environment():
    """Set the nnU-Net environment variables (safe to call from any script)."""
    os.environ.setdefault("nnUNet_raw", NNUNET_RAW)
    os.environ.setdefault("nnUNet_preprocessed", NNUNET_PREPROCESSED)
    os.environ.setdefault("nnUNet_results", NNUNET_RESULTS)
    if os.name == "nt":
        # Windows-friendly worker counts (spawn-based multiprocessing)
        os.environ.setdefault("nnUNet_n_proc_DA", "2")
        os.environ.setdefault("nnUNet_def_n_proc", "2")
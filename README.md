# Road Crack Segmentation with nnU-Net

A complete, automatic pipeline that trains a state-of-the-art crack-segmentation
model on your images and gives you clean prediction masks.

## What you need

- A Windows PC (Linux/Mac work too - see the "Manual commands" section)
- Internet connection (first run only, downloads ~3 GB of software)
- An NVIDIA GPU is **strongly recommended** (any size works - the script adapts).
  Without a GPU it still works, just very slowly.
- Python 3.10+ (install from https://www.python.org/downloads/ - **tick
  "Add Python to PATH"** during installation)

## How to use (the short version)

1. Put this whole folder anywhere on the PC (e.g. the Desktop).
2. Create two folders:
   - `images` - all the photos
   - `masks` - the matching masks
   - Every photo must have a mask with the **same name**
     (photo.jpg + photo.png, or photo.jpg + photo_mask.png - both work)
   - Photos can be jpg/png/bmp/tiff. Masks should be black/white
     (0 = no crack, 255 = crack) or class numbers (0, 1, 2, ...).
3. Double-click **run_all.bat**
4. When asked, type the full paths of the two folders and press Enter.
5. Wait. Training takes hours - the window shows live progress.
   You can close the window and rerun run_all.bat anytime later:
   training **resumes** from where it stopped.
6. Results land in `outputs\`:
   - `outputs\predictions` - the crack masks the model made
   - `outputs\overlays` - the photos with cracks drawn in **red** (best for checking)

## Getting the best possible results

The default run trains one model (fastest). For noticeably better accuracy:

```
python train_model.py --best        (trains 5 models, up to 1000 epochs each - ~5x longer)
python predict_and_export.py --best (combines them + postprocessing + test-time augmentation)
```

`--best` mode stacks four accuracy boosts: 5-fold ensemble, longer training
(1000-epoch cap, early stopping still protects against overfitting),
postprocessing, and test-time augmentation. Expect roughly +2-5% Dice over
the default run.

Big GPU note: on a machine with 10+ GB VRAM, preparation keeps your photos at
full resolution automatically (no resizing) - thin cracks benefit a lot from
that. On smaller GPUs it resizes to 1536 px so training still fits.

## What the numbers mean

nnU-Net reports a "Dice score" (0-1, 1 = perfect). For thin cracks:
- **0.60 - 0.85 is GOOD.** Thin structures always score lower, even when they
  look perfect (a 1-pixel shift on a 2-pixel crack drops the score a lot).
- Always judge with your eyes too: if the red lines follow the real cracks,
  the model works.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python was not found" | Install Python, tick "Add Python to PATH", rerun |
| No GPU message | It works, but expect days instead of hours - get a GPU if possible |
| Out of memory (CUDA OOM) | Close Chrome/games, then rerun (resumes automatically) |
| Training stopped (laptop sleep, crash) | Rerun the same command - it resumes from the last checkpoint |
| I want it to train longer | `python train_model.py --epochs 500` (or 1000) - `--best` already uses 1000 |

## Manual commands (Linux/Mac or advanced)

```bash
pip install -r requirements.txt
python prepare_dataset.py --images ./images --masks ./masks
python train_model.py          # or: python train_model.py --best
python predict_and_export.py   # or: python predict_and_export.py --best
```

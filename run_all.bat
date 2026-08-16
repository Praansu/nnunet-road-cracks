@echo off
chcp 65001 >nul
title Road Crack Segmentation - nnU-Net
cd /d "%~dp0"
set PYTHONUTF8=1

echo ============================================
echo   Road Crack Segmentation - nnU-Net
echo   Runs everything: install, prepare,
echo   train, predict.
echo ============================================
echo.

rem ---- find Python ----
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
    echo ERROR: Python was not found.
    echo Install it from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add Python to PATH" during installation.
    echo Then double-click this file again.
    pause
    exit /b 1
)

echo [1/4] Installing required software (first run only - can take 10+ minutes)...
%PY% -m pip install --upgrade pip >nul 2>&1
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Install failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [2/4] Preparing the dataset...
echo       You will be asked for the folders with the photos and the masks.
%PY% prepare_dataset.py
if errorlevel 1 (
    echo.
    echo Dataset preparation failed. Read the message above.
    pause
    exit /b 1
)

echo.
echo [3/4] Training the model (the long step - plan for hours)...
%PY% train_model.py
if errorlevel 1 (
    echo.
    echo Training failed. Read the message above.
    pause
    exit /b 1
)

echo.
echo [4/4] Predicting on test images and creating overlays...
%PY% predict_and_export.py
if errorlevel 1 (
    echo.
    echo Prediction failed. Read the message above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   ALL DONE!
echo   Open the "outputs" folder:
echo     outputs\predictions  - the crack masks
echo     outputs\overlays     - photos with cracks
echo                           drawn in red
echo ============================================
pause

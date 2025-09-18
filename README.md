
# KP+Hands Fusion Runtime (CPU, Webcam + Console)

This is a minimal runtime that loads your **fusion** model (`KPHandsFusion`) trained on keypoint+hand RGB crops and runs **webcam inference on CPU**. It prints the top-1 sign label to the console and exposes a `/health` endpoint.

## What it does

- Captures webcam at **1280×720** (or your device's closest match).
- Uses **MediaPipe Pose + Hands** (CPU) to extract:
  - Pose landmarks (33 × [x,y,z,v]) and
  - Left/Right hand landmarks (21 × [x,y,z] each).
- Builds a **258‑dim keypoint vector per frame**: `[pose(33*4) + left(21*3) + right(21*3)]`.
- Crops **left/right hand RGB patches** around the hand landmarks (96×96), masks missing hands.
- Maintains a sliding **window T=64** and feeds the model every frame.
- Prints: `[Sign] <label> (p=0.87)` to the console.
- **CPU only** (as requested). Target FPS ~30 (depends on CPU; reduce `PRINT_EVERY` or frame size if needed).

## Files

- `model_def.py` – model architecture to match your checkpoint keys
- `run_fusion.py` – the runtime (webcam + mediapipe + FastAPI health)
- `requirements.txt` – dependencies
- `README.md` – this file

## Setup

Create a virtual environment and install deps:

```bash
python -m venv .venv
/Windows PowerShell
. .venv/Scripts/Activate.ps1
/macOS/Linux
. source .venv/bin/activate

pip install -r requirements.txt
```

> **Torch CPU tip:** If default `pip install torch` fails or installs GPU builds you don't need, use:
> - Windows/Linux (CPU only):  
>   `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Put your files

- Model checkpoint: e.g. `fusion_best.pt` (your uploaded file)
- Labels CSV: e.g. `labels.csv` formatted as `id,label_name`

## Run (console + health API)

```bash
# In the same folder as run_fusion.py / model_def.py

set CAM_INDEX=0
set FRAME_W=1280
set FRAME_H=720
set TARGET_FPS=30
set T_WINDOW=64
set HAND_IMG_SIZE=96
set PRINT_EVERY=2
# (on macOS/Linux use "export" instead of "set")

# Start runtime + FastAPI
python run_fusion.py & uvicorn run_fusion:app --host 0.0.0.0 --port 8000
# Or, if you only want console (no API server), run:
# set NO_API=1 && python run_fusion.py
```

Open health check at: `http://localhost:8000/health`

## Notes


## Troubleshooting

- **Webcam won't open**: set a different `CAM_INDEX` (1, 2, ...). Close other apps using the camera.
- **Very low FPS**: lower `FRAME_W/FRAME_H` or set `PRINT_EVERY=5` to reduce console spam.
- **Import errors**: ensure `mediapipe==0.10.14`, `opencv-python`, and a CPU build of `torch` are installed.

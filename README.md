# tinyml-bearing-fault

Bearing fault classification from vibration signals, using a small neural network quantized to int8 (3.8 KB) for microcontrollers.

**Status:** runs in Python only. The int8 model has not been deployed to a board yet.

## Results

| Model | Test data | Accuracy | Size |
|---|---|---|---|
| MLP float32 (`baseline.keras`) | load 1 hp (held-out 20 %, 952 windows) | 98.8 % | 30.3 KB |
| MLP int8 (`baseline_int8.tflite`) | load 1 hp (same split) | 98.4 % | 3.8 KB |

Measured on 2026-10-30 with the saved models in `models/`. Retraining gives slightly different numbers.

## Pipeline

1. Load the drive-end signal of the CWRU bearing dataset (48 kHz): Normal, plus ball, inner ring and outer ring faults at 3 damage sizes (10 classes).
2. Cut each signal into windows of 2048 points, with a step of 1024.
3. Extract 13 features per window: 6 time features and 7 frequency features.
4. Split chronologically per class: 70 % train, 10 % validation, 20 % test.
5. Train an MLP (13 → 16 → 8 → 10).
6. Convert it to a full int8 `.tflite` model.

## How to run

The dataset is not in this repo. Download the CWRU `.mat` files (load 1 hp) into `data/raw/`. The expected filenames are in `src/load_data.py`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the scripts from the repo root, in this order. `src/model.py` retrains and overwrites the saved model.

```bash
.venv/bin/python src/load_data.py
.venv/bin/python src/windowing.py
.venv/bin/python src/features.py
.venv/bin/python src/split.py
.venv/bin/python src/model.py
.venv/bin/python src/quantize.py
.venv/bin/python src/testq.py
```

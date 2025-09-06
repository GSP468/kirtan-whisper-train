# Kirtan Whisper — Marker Package (with Padding)

This package lets markers run a minimal **training** cycle and **evaluate** on a provided test set, without needing the full dataset.

## Structure
```
Marker_Package/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ marker_small_test_script.py       # marker-friendly demo trainer (runnable)
│  ├─ evaluate_metrics.py               # evaluation script (WER/CER)
│  └─ full_training/
│     └─ train_whisper_fixed_manualpad_v2_continued.py   # original full script (view only)
├─ samples/                             # small demo dataset
├─ test_evaluate/                       # evaluation dataset
└─ checkpoints/                         # placeholder for checkpoints (empty here)
```

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train (demo, uses `samples/` by default)
```bash
python -m src.marker_small_test_script
```
This will pad/crop mel spectrograms to 3000 frames, tokenize text, and run a short training loop.  
The trained model is saved to `checkpoints/trained_demo`.

### 3. Evaluate
```bash
python -m src.evaluate_metrics
```
This evaluates on the provided `.wav` + `.txt` pairs in `test_evaluate/` and prints WER/CER.

---

## Important Notes for Markers

- The folder `src/full_training/` contains the **original training script** used during the project.  
  - It is included **for reference only**.  
  - It will **not run as-is**, because:
    - It expects Weights & Biases (`wandb`) logging to be set up.  
    - It resumes training from a **checkpoint folder (`epoch_10`)** which is not distributed here due to size.  
- The **only runnable trainer** is `src/marker_small_test_script.py`, designed for quick demonstration on the bundled `samples/`.

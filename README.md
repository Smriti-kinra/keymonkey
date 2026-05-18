#  KeyMonkey: Piano Note Prediction with ML
# Overview
KeyMonkey is a multi-model machine learning project that learns to predict the next frame of piano notes given a history of previous frames. Given a sequence of piano roll frames (which keys are pressed at each moment in time), the model predicts what notes will be active in the **next** time slice.

This is an **autoregressive, multi-label classification** problem  at each step the model must independently predict the state of all 88 piano keys simultaneously.

**Five models are trained and compared:**

| Model | Type | Description |
|---|---|---|
| Vanilla RNN | Recurrent Neural Network | Baseline sequential model using `tanh` nonlinearity |
| GRU | Recurrent Neural Network | Gated Recurrent Unit — efficient gating with ~66% of LSTM parameters |
| LSTM | Recurrent Neural Network | Long Short-Term Memory — best long-range memory |
| LightGBM | Gradient-Boosted Trees | Per-pitch binary classifiers on flattened sliding windows |
| SGD-SVM | Linear Support Vector Machine | Hinge-loss linear classifier on scaled sliding windows |

---

##  Repository Structure

```
keymonkey/
├── KeyMonkey_dataset_preprocessing_final.ipynb   # Step 1: MIDI to Piano Roll cache
├── KeyMonkey_model_final.ipynb                   # Step 2: Train RNN / GRU / LSTM
├── KeyMonkey_rf_svm_final.ipynb                  # Step 3: Train LightGBM + SGD-SVM
├── KeyMonkey_grand_comparison.ipynb              # Step 4: Unified evaluation & plots
└── README.md
```

> **Note:** All notebooks are designed for **Google Colab** with the MAESTRO dataset stored in Google Drive.



## Dataset — MAESTRO v3

The project uses the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) (**MA**estro **E**lectric **S**tdio **T**raining **R**ecordings and **O**nline) dataset:

- ~200 hours of virtuoso piano performances
- Classical music recorded on a Yamaha Disklavier — acoustic piano with MIDI capture
- ~1,200 MIDI files, each annotated with the original audio
- Pre-split into train / validation / test at the **composition level** (same piece never appears in two splits)

# Piano Roll Representation

Each MIDI file is converted to a binary piano roll matrix:

```
Shape: (T_frames, 88)
  T_frames = duration_in_seconds × 32   (32 Hz frame rate)
  88       = number of standard piano keys (A0 to C8, MIDI 21–108)

Cell value: 1.0 if that key is sounding at that frame, 0.0 otherwise
```

### Train / Validation / Test Split

The MAESTRO CSV split is used as-is — no re-splitting. This is critical: it ensures the same pianist's performances don't leak across splits.

---

## Pipeline

```
MAESTRO v3 MIDIs
      │
      ▼
[Preprocessing Notebook]
  pretty_midi → binary piano roll at 32 Hz → .npz cache files
  train_index.txt / validation_index.txt / test_index.txt
  pos_weight computed from training data only
      │
      ├──────────────────────────────┐
      ▼                              ▼
[Model Notebook]              [RF/SVM Notebook]
  Sequential windows            Sliding windows
  (B, 511, 88) tensors          W=16 frames, stride=4
  RNN / GRU / LSTM              LightGBM (per-pitch) + SGD-SVM
      │                              │
      └──────────────┬───────────────┘
                     ▼
            [Grand Comparison Notebook]
              Unified metrics + plots
```

---

## Model Details

### Recurrent Models (RNN, GRU, LSTM)

**Architecture (all three share the same spec):**

```
Input:  (batch, 511, 88)   — 511 frames of 88-key piano roll
RNN/GRU/LSTM:  hidden=256, layers=2, dropout=0.3
Linear FC:     256 → 88 (raw logits, one per key)
Output: (batch, 511, 88)   — next-frame predictions for every timestep
```

**Training details:**

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Max frames per clip | 512 |
| Learning rate | 3e-4 (AdamW) |
| Weight decay | 1e-2 |
| LR schedule | Linear warmup (5 epochs) → Cosine annealing |
| Max epochs | 50 |
| Early stopping | Patience = 8, min_delta = 1e-4 |
| Loss | `BCEWithLogitsLoss` with `pos_weight` (silence:note ratio) |
| TBPTT chunk | 64 frames (truncated backprop through time) |

**Key design choices:**
- **TBPTT (Truncated Backprop Through Time):** Hidden state is detached every 64 frames instead of backpropagating through all 511 steps — dramatically reduces memory and compute per batch.
- **pos_weight:** Computed from training data to correct severe class imbalance (most frames are silent). Prevents the model from collapsing to always predicting silence.
- **Automatic threshold grid-search:** The binarization threshold is searched on validation data each epoch (0.1–0.65 in 12 steps) by micro-F1, so no manual tuning is needed.

### LightGBM

- **88 independent binary classifiers**, one per piano key
- Input features: 16 consecutive frames flattened → `(16 × 88,) = 1408` features
- Sliding window with stride=4 (non-overlapping 125ms steps to prevent memorisation)
- `is_unbalance=True` to handle class imbalance (mirrors `pos_weight`)
- Pitches that are always silent in training are assigned a constant-class predictor

### SGD-SVM

- Linear SVM using hinge loss via `SGDClassifier`
- Features scaled with `StandardScaler` (mandatory for linear kernels)
- Trained on a 20,000-sample subsample (SVM is O(n²)–O(n³))
- `SafeSGD` wrapper handles constant-class pitches (very rare low/high register keys)

---

## Evaluation Metrics

All models are evaluated on the held-out **test set** using:

| Metric | Description |
|---|---|
| **Exact Match** | Fraction of frames where all 88 keys are predicted exactly right |
| **Hamming Accuracy** | Per-key accuracy averaged over all 88 keys and all frames |
| **F1 Macro** | F1 averaged equally over all 88 keys (sensitive to rare keys) |
| **F1 Micro** | F1 computed globally over all key-frame pairs |
| **F1 Sample** | F1 averaged per frame (measures per-chord quality) |
| **Skill Score** | Hamming accuracy minus the always-predict-silence baseline |

> The **skill score** is especially important: given that ~95% of all key-frame cells are silent, a trivial model that always predicts silence would achieve ~95% Hamming accuracy. Skill score measures the *gain above that floor*.


## How to Run

### Step-by-Step

**Step 1 - Preprocessing** (run once)

Open `KeyMonkey_dataset_preprocessing_final.ipynb` in Google Colab and run all cells. This will:
- Convert all ~1,200 MIDI files to `.npz` piano roll caches
- Write `train_index.txt`, `validation_index.txt`, `test_index.txt`
- Compute `pos_weight` for loss balancing

**Step 2 - Train RNN / GRU / LSTM**

Open `KeyMonkey_model_final.ipynb` and run all cells. Training resumes automatically from the latest checkpoint if interrupted.

Expected outputs in `maestro_checkpoints/`:
```
best_RNN.pt
best_GRU.pt
best_LSTM.pt
```

**Step 3 - Train LightGBM + SVM**

Open `KeyMonkey_rf_svm_final.ipynb` and run all cells.

Expected outputs:
```
lgbm_estimators.pkl
svm_model.pkl
```

**Step 4 - Grand Comparison**

Open `KeyMonkey_grand_comparison.ipynb`. This loads all five checkpoints and produces unified evaluation tables and visualisations.



## Dependencies

```
# Core
torch >= 2.0
numpy
pandas
matplotlib

# MIDI parsing
pretty_midi

# Classical ML
lightgbm
scikit-learn
joblib

# Utilities
tqdm
```

Install all at once:
```bash
pip install torch numpy pandas matplotlib pretty_midi lightgbm scikit-learn joblib tqdm
```

In Colab, `pretty_midi` and `lightgbm` are installed inline at the top of the relevant notebooks.



##  Key Implementation Details

### Why TBPTT?

Standard BPTT over 511 frames requires storing gradients for the entire sequence — expensive in both memory and compute. TBPTT detaches the hidden state every 64 frames, training as if the sequence consists of independent 64-frame chunks. The forward pass still sees the full context (hidden state is carried through); only gradient flow is truncated.

### Why LightGBM over Random Forest?

1. LightGBM handles all 88 keys natively as a multi-output problem — no need to train 88 separate forests.
2. LightGBM trains 5–20× faster via histogram-based gradient boosting.
3. `is_unbalance=True` replicates the pos_weight correction from the neural models.

### Why stride=4 in the sliding window?

Without striding, consecutive windows overlap by `W-1 = 15` frames, making ~94% of window pairs nearly identical. This inflates the training set without adding information and causes the model to memorise rather than generalise. Stride=4 gives non-overlapping 125 ms steps.

### Class Imbalance

Classical piano performances have long silent passages. Roughly **95%** of all key-frame cells are 0. Without correction, all models converge to always predicting silence. The fix differs by model type:

- Neural models: `pos_weight = (silent_cells / active_cells)` in `BCEWithLogitsLoss`
- LightGBM: `is_unbalance=True`
- SVM: `class_weight='balanced'` in `SGDClassifier`


## Author

Built as part of the MLPR (Machine Learning and Pattern Recognition) Semester 4 project.
By: Smriti Kinra, Khant Mota, Sameera John. 




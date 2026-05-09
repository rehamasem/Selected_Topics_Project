# Healthcare Vital Signs Risk Prediction System

A deep learning pipeline that classifies patient vital signs into risk categories using a residual neural network. Built as a senior capstone project for the Selected Topics course.

---

## Table of Contents

- [Team Members](#team-members)
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Outputs & Visualizations](#outputs--visualizations)
- [Technology Stack](#technology-stack)
- [AI Assistance Disclosure](#ai-assistance-disclosure)

---

## Team Members

| Name | Student ID |
|------|------------|
| Rahma Ali Bauomi | 192100170 |
| Reham Mohamed Asem | 192100089 |
| Roaa El Emam Mohamed | 192100083 |
| Fatemah Ahmed | 192100160 |
| Merna Ahmed | 192100144 |
| Salma Ahmed | 192200136 |

**Institution:** Egyptian Chinese University (ECU)
**Course:** Selected Topics — Senior Year, Semester 2

---

## Project Overview

This system monitors patient vital signs — **heart rate**, **SpO2 (blood oxygen)**, and **body temperature** — and classifies each reading into one of three risk levels:

| Label | Class | Meaning |
|-------|-------|---------|
| 0 | Normal | Vital signs within acceptable range |
| 1 | Abnormal | Readings outside normal bounds — requires attention |
| 2 | Critical | Dangerous readings — immediate intervention needed |

The model is trained on a dataset of ~170,000 patient readings and achieves **99.2% accuracy** on the held-out test set. It uses a deep residual MLP architecture with SMOTE augmentation and early stopping to handle class imbalance and prevent overfitting.

---

## System Architecture

```
Raw Patient Data (Excel)
(heart_rate, spo2, temperature, hour_of_day, is_night, risk_level)
         │
         ▼
  Preprocessing
  ├── Outlier removal (IQR, factor=2.0)  →  removes ~23.4% of training data
  ├── StandardScaler normalization
  └── SMOTE + Gaussian noise augmentation  →  227,488 final training samples
         │
         ▼
  Deep Residual MLP (PyTorch)
  ├── Stem:  Linear(5→128)
  ├── Block 1-3: 128 → 256 → 512  (expanding)
  ├── Block 4-6: 512 → 256 → 64   (contracting)
  └── Head:  Dropout → Linear(64→3)
         │
         ▼
  Output: class probabilities
  (Normal  /  Abnormal  /  Critical)
         │
         ▼
  Evaluation & Reporting
  ├── Accuracy, F1, ROC-AUC, Kappa, MCC
  ├── Per-class breakdown
  ├── Energy & CO₂ tracking
  └── 8 output visualizations
```

---

## Dataset

| Split | File | Samples |
|-------|------|---------|
| Train | `vital_signs_train.xlsx` | 119,154 |
| Validation | `vital_signs_val.xlsx` | 25,533 |
| Test | `vital_signs_test.xlsx` | 25,533 |
| Merged | `vital_signs_merged.xlsx` | ~170,220 |

**Features per row:**

| Feature | Description |
|---------|-------------|
| `heart_rate` | Beats per minute |
| `spo2` | Blood oxygen saturation (%) |
| `temperature` | Body temperature (°C) |
| `hour_of_day` | Hour extracted from timestamp (0–23) |
| `is_night` | Binary flag: 1 if hour is between 22:00–05:00 |
| `risk_level` | Target label: 0 (Normal), 1 (Abnormal), 2 (Critical) |

**Preprocessing pipeline:**
- Outlier removal via IQR method (factor=2.0) — removes ~23.4% of training samples
- StandardScaler normalization (mean: [78.4, 96.7, 36.8], std: [18.6, 3.0, 0.8])
- SMOTE oversampling + Gaussian noise augmentation → final training set of **227,488 samples**

---

## Model

**File:** [`selected topics/vital_signs_dl_pipeline.py`](selected%20topics/vital_signs_dl_pipeline.py)

A **Deep Residual MLP** architecture trained end-to-end in PyTorch.

**Architecture:**

```
Input (5 features)
    └── Stem: Linear(5→3) → BN → ReLU → Linear(3→128)
         └── Residual Block 1: 128 → 128
         └── Residual Block 2: 128 → 256
         └── Residual Block 3: 256 → 512
         └── Residual Block 4: 512 → 512
         └── Residual Block 5: 512 → 256
         └── Residual Block 6: 256 → 64
              └── Head: Dropout(0.2) → Linear(64→3)
                       └── Output: 3 class logits
```

Each residual block: `Linear → BatchNorm → ReLU → Dropout → Linear → BatchNorm → Skip Connection`

**Training configuration:**

| Setting | Value |
|---------|-------|
| Optimizer | Adam (LR=3e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Early stopping patience | 15 epochs |
| Max epochs | 100 |
| Total parameters | 1,622,275 |
| FLOPs per inference | 3.25 MFLOPs |

**Energy & Environmental Impact:**

| Metric | Value |
|--------|-------|
| Training time | ~32 minutes (CPU, 65W) |
| Training energy | 34.62 Wh |
| Training CO₂ | 8.07 grams |
| Inference energy per sample | 0.621 nanoWatt-hours |

---

## Results

Evaluated on the held-out test set (25,533 samples):

| Metric | Score |
|--------|-------|
| Accuracy | **99.20%** |
| F1-Score (Macro) | 0.9919 |
| Precision (Macro) | 0.9923 |
| Recall (Macro) | 0.9916 |
| ROC-AUC (Macro) | **0.9999** |
| Cohen's Kappa | 0.9879 |
| Matthews CC | 0.9880 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Normal | 0.9989 | 0.9761 | 0.9874 |
| Abnormal | 0.9779 | 0.9990 | 0.9883 |
| Critical | 1.0000 | 0.9998 | **0.9999** |

---

## Project Structure

```
project/
├── README.md
│
├── data/
│   ├── vital_signs_merged.xlsx           # Full combined dataset (~170K samples)
│   ├── vital_signs_train.xlsx            # Training split (119,154 samples)
│   ├── vital_signs_val.xlsx              # Validation split (25,533 samples)
│   └── vital_signs_test.xlsx             # Test split (25,533 samples)
│
└── selected topics/
    ├── vital_signs_dl_pipeline.py        # Main DL training pipeline (654 lines)
    ├── vital_signs_dl_pipeline.ipynb     # Notebook version of the pipeline
    ├── try_2.ipynb                       # Experimental variant notebook
    ├── run_log.txt                       # Training run log
    └── outputs/
        ├── vitalsigns_dl_model.pt        # Saved PyTorch model weights
        ├── results_log.txt               # Complete metrics log
        ├── 00_data_distribution.png      # Feature distributions before/after outlier removal
        ├── 01_learning_curves.png        # Train/val loss & accuracy over epochs
        ├── 02_confusion_matrix.png       # Normalized confusion matrix on test set
        ├── 03_roc_curves.png             # ROC curves per class + macro AUC
        ├── 04_prob_violin.png            # Predicted probability distributions
        ├── 05_energy_carbon.png          # Training energy & CO₂ emissions
        ├── 06_summary_dashboard.png      # 6-panel comprehensive results dashboard
        └── 07_live_predictions.png       # Live prediction interface visualization
```

---

## How to Run

### Prerequisites

```bash
pip install torch scikit-learn pandas numpy imbalanced-learn openpyxl matplotlib seaborn thop
```

### Train the Model

```bash
cd "selected topics"
python vital_signs_dl_pipeline.py
```

The script will:
1. Load and preprocess the Excel datasets from `../data/`
2. Remove outliers, apply SMOTE augmentation, and scale features
3. Train the residual MLP with early stopping
4. Save the trained model to `outputs/vitalsigns_dl_model.pt`
5. Generate all 8 visualizations in `outputs/`
6. Write full metrics to `outputs/results_log.txt`

### Notebook Version

Open `selected topics/vital_signs_dl_pipeline.ipynb` in Jupyter to run the pipeline interactively, cell by cell.

---

## Outputs & Visualizations

All plots are saved automatically to `selected topics/outputs/` during training:

| File | Description |
|------|-------------|
| `00_data_distribution.png` | Feature histograms before and after outlier removal |
| `01_learning_curves.png` | Training and validation loss/accuracy across epochs |
| `02_confusion_matrix.png` | Normalized confusion matrix on test set |
| `03_roc_curves.png` | Per-class ROC curves with AUC scores |
| `04_prob_violin.png` | Violin plot of predicted class probabilities |
| `05_energy_carbon.png` | Bar charts of training energy and CO₂ emissions |
| `06_summary_dashboard.png` | Combined 6-panel results summary |
| `07_live_predictions.png` | Visualization of live inference on sample patients |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch |
| Data Processing | pandas, numpy, openpyxl |
| Preprocessing & Metrics | scikit-learn |
| Class Rebalancing | imbalanced-learn (SMOTE) |
| Visualization | matplotlib, seaborn |
| Model Profiling | thop (FLOPs/MACs calculation) |

---

## AI Assistance Disclosure

AI tools (Claude and ChatGPT) were used in a limited, advisory capacity during the design phase of this project. Their contributions were restricted to:

- Suggesting the **number of residual blocks** and the expand-then-contract layer structure (128 → 512 → 64) based on the input size and task complexity.
- Advising on **feature scaling direction** — whether to scale up (wider layers) or scale down (narrower layers) at specific stages of the network.
All data collection, preprocessing decisions, model training, evaluation, and analysis were carried out entirely by the team. AI tools were not used to write any code or generate results.
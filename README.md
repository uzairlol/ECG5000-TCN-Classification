# ECG5000 Anomaly Detection & Classification with Temporal Convolutional Networks (TCN)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch%20Lightning-2.0%2B-orange.svg)](https://www.pytorchlightning.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modular, production-grade deep learning pipeline for anomaly detection and binary classification on the **ECG5000 dataset**. The project provides two state-of-the-art architectures using **Temporal Convolutional Networks (TCN)**:
1. **Supervised Classifier**: Direct classification model with TCN backbone trained on binary labels.
2. **Self-Supervised Reconstruction**: An anomaly detector trained solely on normal ECG signals, evaluating anomalies using reconstruction error thresholding.

---

## Key Features

- **TCN Backbone**: High-fidelity feature extraction using dilated causal convolutions.
- **Robust Preprocessing**: Standardized scaling using `RobustScaler` fit strictly on training splits.
- **Calibrated Thresholding**: Automatic detection threshold calibration on validation set probability distributions.
- **Dual Pipeline**: Switch seamlessly between fully-supervised classification and self-supervised reconstruction-based anomaly detection.
- **Comprehensive Visualization**: Generates confusion matrices, probability/score distributions, and reconstruction quality comparisons automatically.

---

## Directory Structure

```
├── data/                    # ECG5000 Train/Test datasets (ARFF, TS, TXT)
├── src/
│   └── ecg_project/         # Core python package
│       ├── cli.py           # CLI entry point command parser
│       ├── config.py        # Configuration and hyperparameters
│       ├── data.py          # PyTorch dataset builders & dataloaders
│       ├── evaluation.py    # Metric calculation & figure generators
│       ├── model.py         # Supervised TCN classifier model
│       ├── preprocessing.py # Scaler and data transformation utilities
│       ├── reconstruction.py# Self-supervised denoising reconstruction model
│       └── visualization.py # Figure plotting utilities
├── tests/                   # Suite of unit tests for data, model & preprocessing
├── artifacts/               # Directory for checkpoints, scalers, and plots (ignored by git)
├── pyproject.toml           # Package installation and dependency metadata
└── .gitignore               # Excludes build, cache, and artifact outputs
```

---

## Installation & Setup

Set up the project using your existing environment.

### 1. Install Package
Install the package in editable mode along with development dependencies:
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m pip install -e .[dev]
```

### 2. Verify Installation
Verify that the entry point `ecg5000-train` is successfully installed:
```bash
ecg5000-train --help
```

---

## CLI Usage Guide

The package provides a unified CLI that can be run directly via the Conda Python interpreter:

### Pipeline A: Supervised Classification

#### Step 1: Train the Classifier
Fits a supervised TCN model. Best checkpoints are saved to `artifacts/ecg_tcn.ckpt`.
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m ecg_project.cli train
```

#### Step 2: Evaluate the Classifier
Calibrates decision threshold on validation data and outputs metrics, predictions, and visualization plots on test data.
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m ecg_project.cli evaluate
```

---

### Pipeline B: Self-Supervised Anomaly Detection

#### Step 1: Train the Reconstruction Model
Trains a denoising TCN autoencoder strictly on normal ECG signals (Label `1`). Checkpoint is saved to `artifacts/ecg_reconstruction.ckpt`.
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m ecg_project.cli pretrain
```

#### Step 2: Detect Anomalies
Evaluates the reconstruction error on validation data to calibrate a detection threshold, then flags test samples and outputs plots.
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m ecg_project.cli detect
```

---

### Pipeline C: Self-Supervised Contrastive Learning (Research Mode)

#### Step 1: Pretrain via Contrastive Loss (SimCLR / NT-Xent)
Pretrains the TCN encoder backbone using raw unlabeled ECG signals with temporal and amplitude augmentations. Checkpoint is saved to `artifacts/ecg_ssl_encoder.ckpt`.
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m ecg_project.cli ssl-pretrain
```

#### Step 2: Evaluate representations via Linear Probing
Freezes the pretrained TCN encoder, extracts representations, and evaluates them by training a Logistic Regression model on varying proportions of labeled training data (1%, 10%, 100%).
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m ecg_project.cli ssl-probe
```

---

## Testing

Run unit tests to verify the integrity of dataset creation, model structures, and preprocessing pipelines:
```bash
& C:/ProgramData/miniconda3/envs/ml/python.exe -m pytest
```

---

## Output Artifacts & Visualizations

Upon evaluation/detection, the following outputs are generated inside the `artifacts/` folder:

- **Checkpoints & Scalers**:
  - `ecg_tcn.ckpt` & `ecg_reconstruction.ckpt`: Model weight files.
  - `scaler.joblib`: Fitted preprocessing configuration.
- **Metrics**:
  - `evaluation_summary.json` / `ecg_reconstruction_summary.json`: Precision, Recall, F1-Score, and ROC-AUC.
  - `confusion_matrix.csv` / `ecg_reconstruction_confusion_matrix.csv`: Tabular raw matrix.
- **Visuals**:
  - `classifier_confusion_matrix.png` / `reconstruction_confusion_matrix.png`
  - `classifier_score_distribution.png` / `reconstruction_score_distribution.png`
  - `reconstruction_examples.png` (Reconstruction only)

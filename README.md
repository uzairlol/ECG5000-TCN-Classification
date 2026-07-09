# ECG5000 TCN Classification

This project is being upgraded from a notebook prototype into a modular Python package. The new pipeline treats ECG5000 as a binary classification problem where label `1` is normal and every other label is anomalous.

## Layout

- `src/ecg_project/`: reusable library code
- `artifacts/`: saved checkpoints and preprocessing objects
- `src/ecg_project/cli.py`: command-line entry point
- `Attempts/`: legacy notebook experiments kept for reference

## Project Flow

- `train` loads the ECG5000 train/test files, creates a validation split from training data, fits the scaler on train only, trains the TCN classifier, and saves the best checkpoint plus the fitted scaler.
- `evaluate` reloads the saved checkpoint and scaler, calibrates the anomaly threshold on validation probabilities, and reports classification metrics on the test split.
- The notebook is no longer the primary workflow; it is kept only as historical reference.

## Install

Use an editable install so the package can be imported from scripts and tests.

```bash
pip install -e .
```

## Train

```bash
ecg5000-train train
```

## Evaluate

```bash
ecg5000-train evaluate
```

## Self-Supervised Upgrade

This repo also includes a reconstruction-based anomaly detector that trains only on normal ECGs and scores anomalies using reconstruction error.

Train it with:

```bash
ecg5000-train pretrain
```

Evaluate it with:

```bash
ecg5000-train detect
```

Its outputs are saved under `artifacts/` with the `ecg_reconstruction.*` filenames.

## Artifacts

After training, the project writes the following files to `artifacts/`:

- `ecg_tcn.ckpt`: the best validation checkpoint copied to a stable filename
- `scaler.joblib`: the fitted `RobustScaler` used for inference

### Saved Visuals

Classifier evaluation also saves:

- `classifier_confusion_matrix.png`
- `classifier_score_distribution.png`

Self-supervised reconstruction detection also saves:

- `reconstruction_confusion_matrix.png`
- `reconstruction_score_distribution.png`
- `reconstruction_examples.png`

You can embed the generated figures directly in the README:

![Classifier confusion matrix](artifacts/classifier_confusion_matrix.png)

![Classifier score distribution](artifacts/classifier_score_distribution.png)

![Reconstruction examples](artifacts/reconstruction_examples.png)

## What Changed

- The code is now organized as a package instead of a single notebook.
- The raw ECG rows are treated as individual samples rather than a fake time series.
- Validation monitoring now uses validation loss.
- The old z-score anomaly logic is replaced by a threshold calibrated from validation data.
- The scaler is saved separately so inference is reproducible.

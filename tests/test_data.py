from pathlib import Path

import numpy as np

from ecg_project.data import ECG_FEATURE_COUNT, load_ecg_file, split_train_validation, to_binary_labels


def test_load_ecg_file_reads_expected_columns():
    frame = load_ecg_file(Path("data/ECG5000/ECG5000_TRAIN.txt"))

    assert frame.shape[1] == ECG_FEATURE_COUNT + 1
    assert list(frame.columns)[0] == "label"
    assert frame["label"].dtype.kind in {"i", "u"}


def test_to_binary_labels_marks_non_normal_as_anomaly():
    labels = np.array([1, 1, 2, 3, 1])

    binary = to_binary_labels(labels)

    assert binary.tolist() == [0, 0, 1, 1, 0]


def test_split_train_validation_is_deterministic_and_stratified():
    frame = load_ecg_file(Path("data/ECG5000/ECG5000_TRAIN.txt")).head(200)

    train_a, val_a = split_train_validation(frame, validation_size=0.2, random_state=42)
    train_b, val_b = split_train_validation(frame, validation_size=0.2, random_state=42)

    assert train_a.equals(train_b)
    assert val_a.equals(val_b)
    assert set(train_a["label"].unique()) <= set(frame["label"].unique())
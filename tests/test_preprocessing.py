from pathlib import Path

import numpy as np

from ecg_project.data import load_ecg_file, split_train_validation
from ecg_project.preprocessing import load_scaler, prepare_arrays, save_scaler


def test_prepare_arrays_returns_expected_shapes(tmp_path):
    frame = load_ecg_file(Path("data/ECG5000/ECG5000_TRAIN.txt")).head(180)
    train_frame, validation_frame = split_train_validation(frame, validation_size=0.2, random_state=42)
    test_frame = load_ecg_file(Path("data/ECG5000/ECG5000_TEST.txt")).head(60)

    arrays, scaler = prepare_arrays(train_frame, validation_frame, test_frame)

    assert arrays.train_features.shape[1] == 140
    assert arrays.validation_features.shape[1] == 140
    assert arrays.test_features.shape[1] == 140
    assert arrays.train_features.dtype == np.float32
    assert set(np.unique(arrays.train_labels)).issubset({0, 1})

    scaler_path = tmp_path / "scaler.joblib"
    save_scaler(scaler, scaler_path)
    loaded_scaler = load_scaler(scaler_path)
    arrays_with_loaded_scaler, _ = prepare_arrays(train_frame, validation_frame, test_frame, scaler=loaded_scaler)

    np.testing.assert_allclose(arrays.train_features, arrays_with_loaded_scaler.train_features)
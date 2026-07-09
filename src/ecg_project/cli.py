from __future__ import annotations

import argparse
from shutil import copy2
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .data import (
    build_reconstruction_dataset,
    build_tensor_dataset,
    load_dataset,
    make_dataloaders,
    split_train_validation,
)
from .evaluation import (
    anomaly_predictions,
    best_threshold_from_validation,
    classification_metrics,
    save_confusion_matrix_figure,
    save_confusion_matrix,
    save_evaluation_summary,
    save_reconstruction_examples_figure,
    save_score_distribution_figure,
    save_predictions,
)
from .model import ECGTCNClassifier
from .reconstruction import ECGDenoisingReconstructionModel
from .preprocessing import load_scaler, prepare_arrays, save_scaler


def collect_probabilities(model: ECGTCNClassifier, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probabilities = model.predict_proba(inputs)[:, 1].detach().cpu().numpy()
            all_scores.append(probabilities)
            all_labels.append(labels.detach().cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_scores)


def run_training(config: ProjectConfig) -> None:
    seed_everything(config.data.random_state, workers=True)
    train_frame, test_frame = load_dataset(config.data)
    train_frame, validation_frame = split_train_validation(
        train_frame,
        validation_size=config.data.validation_size,
        random_state=config.data.random_state,
    )
    arrays, scaler = prepare_arrays(train_frame, validation_frame, test_frame)

    train_dataset = build_tensor_dataset(arrays.train_features, arrays.train_labels)
    validation_dataset = build_tensor_dataset(arrays.validation_features, arrays.validation_labels)
    test_dataset = build_tensor_dataset(arrays.test_features, arrays.test_labels)
    train_loader, validation_loader, test_loader = make_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=config.model.batch_size,
    )

    model = ECGTCNClassifier(
        input_length=config.model.input_length,
        channels=config.model.channels,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    checkpoint = ModelCheckpoint(
        dirpath=config.artifacts_dir,
        filename=config.checkpoint_name.replace(".ckpt", ""),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    trainer = Trainer(
        max_epochs=config.model.max_epochs,
        callbacks=[checkpoint, early_stopping],
        default_root_dir=config.artifacts_dir,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, validation_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("Training finished without a best checkpoint being recorded.")
    trainer.test(model=None, dataloaders=test_loader, ckpt_path=checkpoint.best_model_path)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, config.artifacts_dir / config.scaler_name)

    best_checkpoint_path = Path(checkpoint.best_model_path)
    canonical_checkpoint_path = config.artifacts_dir / config.checkpoint_name
    if best_checkpoint_path.resolve() != canonical_checkpoint_path.resolve():
        copy2(best_checkpoint_path, canonical_checkpoint_path)


def collect_reconstruction_errors(
    model: ECGDenoisingReconstructionModel, loader
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_originals: list[np.ndarray] = []
    all_reconstructions: list[np.ndarray] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            reconstructions = model(inputs)
            scores = ((reconstructions - inputs) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.detach().cpu().numpy())
            all_originals.append(inputs.detach().cpu().numpy())
            all_reconstructions.append(reconstructions.detach().cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_scores),
        np.concatenate(all_originals),
        np.concatenate(all_reconstructions),
    )


def run_reconstruction_training(config: ProjectConfig) -> None:
    seed_everything(config.data.random_state, workers=True)
    train_frame, test_frame = load_dataset(config.data)
    train_frame, validation_frame = split_train_validation(
        train_frame,
        validation_size=config.data.validation_size,
        random_state=config.data.random_state,
    )
    normal_train_frame = train_frame.loc[train_frame["label"] == 1].reset_index(drop=True)
    arrays, scaler = prepare_arrays(normal_train_frame, validation_frame, test_frame)

    train_dataset = build_reconstruction_dataset(arrays.train_features)
    validation_dataset = build_reconstruction_dataset(arrays.validation_features)
    test_dataset = build_reconstruction_dataset(arrays.test_features)
    train_loader, validation_loader, test_loader = make_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=config.model.batch_size,
    )

    model = ECGDenoisingReconstructionModel(
        input_length=config.model.input_length,
        channels=config.model.channels,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    checkpoint = ModelCheckpoint(
        dirpath=config.artifacts_dir,
        filename=config.reconstruction_checkpoint_name.replace(".ckpt", ""),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    trainer = Trainer(
        max_epochs=config.model.max_epochs,
        callbacks=[checkpoint, early_stopping],
        default_root_dir=config.artifacts_dir,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, validation_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("Training finished without a best checkpoint being recorded.")
    trainer.test(model=None, dataloaders=test_loader, ckpt_path=checkpoint.best_model_path)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, config.artifacts_dir / config.scaler_name)

    best_checkpoint_path = Path(checkpoint.best_model_path)
    canonical_checkpoint_path = config.artifacts_dir / config.reconstruction_checkpoint_name
    if best_checkpoint_path.resolve() != canonical_checkpoint_path.resolve():
        copy2(best_checkpoint_path, canonical_checkpoint_path)


def run_reconstruction_evaluation(config: ProjectConfig) -> None:
    train_frame, test_frame = load_dataset(config.data)
    train_frame, validation_frame = split_train_validation(
        train_frame,
        validation_size=config.data.validation_size,
        random_state=config.data.random_state,
    )
    normal_train_frame = train_frame.loc[train_frame["label"] == 1].reset_index(drop=True)
    scaler_path = config.artifacts_dir / config.scaler_name
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler at {scaler_path}. Run training first.")

    arrays, _ = prepare_arrays(normal_train_frame, validation_frame, test_frame, scaler=load_scaler(scaler_path))

    validation_dataset = build_tensor_dataset(arrays.validation_features, arrays.validation_labels)
    test_dataset = build_tensor_dataset(arrays.test_features, arrays.test_labels)
    validation_loader = DataLoader(validation_dataset, batch_size=config.model.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.model.batch_size)

    checkpoint_path = config.artifacts_dir / config.reconstruction_checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}. Run training first.")

    model = ECGDenoisingReconstructionModel.load_from_checkpoint(checkpoint_path)

    validation_labels, validation_scores, _, _ = collect_reconstruction_errors(model, validation_loader)
    test_labels, test_scores, test_originals, test_reconstructions = collect_reconstruction_errors(model, test_loader)

    threshold, _ = best_threshold_from_validation(validation_labels, validation_scores)
    test_pred = anomaly_predictions(test_scores, threshold)
    metrics = classification_metrics(test_labels, test_pred, test_scores)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(
        config.artifacts_dir / config.reconstruction_predictions_name,
        test_labels,
        test_scores,
        test_pred,
        threshold,
    )
    save_confusion_matrix(
        config.artifacts_dir / config.reconstruction_confusion_matrix_name,
        test_labels,
        test_pred,
    )
    save_confusion_matrix_figure(
        config.artifacts_dir / "reconstruction_confusion_matrix.png",
        test_labels,
        test_pred,
        title="Reconstruction detector confusion matrix",
    )
    save_score_distribution_figure(
        config.artifacts_dir / "reconstruction_score_distribution.png",
        test_labels,
        test_scores,
        title="Reconstruction score distribution",
        threshold=threshold,
    )
    save_reconstruction_examples_figure(
        config.artifacts_dir / "reconstruction_examples.png",
        test_originals.squeeze(1),
        test_reconstructions.squeeze(1),
        test_labels,
        test_scores,
        threshold,
    )
    save_evaluation_summary(
        config.artifacts_dir / config.reconstruction_summary_name,
        metrics,
        threshold,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def run_evaluation(config: ProjectConfig) -> None:
    train_frame, test_frame = load_dataset(config.data)
    train_frame, validation_frame = split_train_validation(
        train_frame,
        validation_size=config.data.validation_size,
        random_state=config.data.random_state,
    )
    scaler_path = config.artifacts_dir / config.scaler_name
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler at {scaler_path}. Run training first.")

    arrays, _ = prepare_arrays(train_frame, validation_frame, test_frame, scaler=load_scaler(scaler_path))

    validation_dataset = build_tensor_dataset(arrays.validation_features, arrays.validation_labels)
    test_dataset = build_tensor_dataset(arrays.test_features, arrays.test_labels)
    validation_loader = DataLoader(validation_dataset, batch_size=config.model.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.model.batch_size)

    checkpoint_path = config.artifacts_dir / config.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}. Run training first.")

    model = ECGTCNClassifier.load_from_checkpoint(checkpoint_path)

    validation_labels, validation_probs = collect_probabilities(model, validation_loader)
    test_labels, test_probs = collect_probabilities(model, test_loader)

    threshold, _ = best_threshold_from_validation(validation_labels, validation_probs)
    test_pred = anomaly_predictions(test_probs, threshold)
    metrics = classification_metrics(test_labels, test_pred, test_probs)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(
        config.artifacts_dir / config.predictions_name,
        test_labels,
        test_probs,
        test_pred,
        threshold,
    )
    save_confusion_matrix(
        config.artifacts_dir / config.confusion_matrix_name,
        test_labels,
        test_pred,
    )
    save_confusion_matrix_figure(
        config.artifacts_dir / "classifier_confusion_matrix.png",
        test_labels,
        test_pred,
        title="Classifier confusion matrix",
    )
    save_score_distribution_figure(
        config.artifacts_dir / "classifier_score_distribution.png",
        test_labels,
        test_probs,
        title="Classifier score distribution",
        threshold=threshold,
    )
    save_evaluation_summary(
        config.artifacts_dir / config.evaluation_summary_name,
        metrics,
        threshold,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECG5000 TCN project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the TCN classifier")
    subparsers.add_parser("evaluate", help="Evaluate saved outputs and metrics")
    subparsers.add_parser("pretrain", help="Train the self-supervised reconstruction model")
    subparsers.add_parser("detect", help="Evaluate the self-supervised anomaly detector")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ProjectConfig()

    if args.command == "train":
        run_training(config)
    elif args.command == "evaluate":
        run_evaluation(config)
    elif args.command == "pretrain":
        run_reconstruction_training(config)
    elif args.command == "detect":
        run_reconstruction_evaluation(config)


if __name__ == "__main__":
    main()

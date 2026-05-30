"""Train the multimodal Keras model on preprocessed tensors."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_multimodal_model


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def load_tensor_splits(data_dir: str = "data/tensors") -> dict:
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}_data.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing tensor file: {path}")
        splits[split] = torch.load(path, map_location="cpu")
    return splits


def train_model(
    data_dir: str = "data/tensors",
    model_output_path: str = "models/multimodal_vol_model.keras",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.2,
    text_units: int = 128,
    lstm_units: int = 96,
    fusion_units: int = 64,
    l2_reg: float = 1e-4,
    early_stop_patience: int = 10,
    reduce_lr_patience: int = 4,
    min_delta: float = 1e-4,
):
    splits = load_tensor_splits(data_dir)

    X_text_train = _to_numpy(splits["train"]["X_text"]).astype(np.float32)
    X_num_train = _to_numpy(splits["train"]["X_num"]).astype(np.float32)
    y_train = _to_numpy(splits["train"]["y"]).astype(np.float32)

    X_text_val = _to_numpy(splits["val"]["X_text"]).astype(np.float32)
    X_num_val = _to_numpy(splits["val"]["X_num"]).astype(np.float32)
    y_val = _to_numpy(splits["val"]["y"]).astype(np.float32)

    X_text_test = _to_numpy(splits["test"]["X_text"]).astype(np.float32)
    X_num_test = _to_numpy(splits["test"]["X_num"]).astype(np.float32)
    y_test = _to_numpy(splits["test"]["y"]).astype(np.float32)

    model = build_multimodal_model(
        text_embedding_dim=X_text_train.shape[1],
        sequence_length=X_num_train.shape[1],
        num_features=X_num_train.shape[2],
        dropout_rate=dropout_rate,
        text_units=text_units,
        lstm_units=lstm_units,
        fusion_units=fusion_units,
        l2_reg=l2_reg,
        learning_rate=learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            min_delta=min_delta,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=reduce_lr_patience,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        [X_text_train, X_num_train],
        y_train,
        validation_data=([X_text_val, X_num_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    # Calculate Naive Baseline (predicting tomorrow's volatility is the same as today's)
    # Since y_test is next_day_volatility, today's volatility is just y_test shifted by 1.
    naive_preds = y_test[:-1]
    actuals = y_test[1:]
    naive_mse = np.mean((actuals - naive_preds) ** 2)
    print(f"Naive Baseline MSE: {naive_mse:.6f}")

    test_results = model.evaluate([X_text_test, X_num_test], y_test, verbose=0, return_dict=True)
    test_mse = test_results.get("mse", test_results["loss"])
    print(f"Model Test MSE: {test_mse:.6f}")

    if model_output_path:
        output_dir = os.path.dirname(model_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        model.save(model_output_path)
        print(f"Saved model to {model_output_path}")

    return model, history, test_mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multimodal volatility model.")
    parser.add_argument("--data-dir", default="data/tensors", help="Directory with train/val/test .pt files")
    parser.add_argument(
        "--model-path",
        default="models/multimodal_vol_model.keras",
        help="Output path for the saved Keras model",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--text-units", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=96)
    parser.add_argument("--fusion-units", type=int, default=64)
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        model_output_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        text_units=args.text_units,
        lstm_units=args.lstm_units,
        fusion_units=args.fusion_units,
        l2_reg=args.l2_reg,
        early_stop_patience=args.early_stop_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()

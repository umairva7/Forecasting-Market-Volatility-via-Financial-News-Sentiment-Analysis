"""Train the multimodal Keras model on preprocessed tensors."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

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
    epochs: int = 20,
    batch_size: int = 32,
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
    )

    history = model.fit(
        [X_text_train, X_num_train],
        y_train,
        validation_data=([X_text_val, X_num_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
    )

    test_loss = model.evaluate([X_text_test, X_num_test], y_test, verbose=0)
    print(f"Test MSE: {test_loss:.6f}")

    if model_output_path:
        output_dir = os.path.dirname(model_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        model.save(model_output_path)
        print(f"Saved model to {model_output_path}")

    return model, history, test_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multimodal volatility model.")
    parser.add_argument("--data-dir", default="data/tensors", help="Directory with train/val/test .pt files")
    parser.add_argument(
        "--model-path",
        default="models/multimodal_vol_model.keras",
        help="Output path for the saved Keras model",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        model_output_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from transformers import TFAutoModel
import tensorflow as tf
import numpy as np
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_multimodal_model


def load_tensor_splits(data_dir="data/tensors"):
    splits = {}
    for split in ("train", "val", 'test'):
        path = os.path.join(data_dir, f"{split}_data.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing tensor file: {path}")
        splits[split] = np.load(path)
    return splits


def train_model(
    data_dir="data/tensors",
    model_output_path="models/multimodal_vol_model.keras",
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    dropout_rate=0.2,
    text_units=128,
    lstm_units=96,
    fusion_units=64,
    l2_reg=1e-4,
    early_stop_patience=10,
    reduce_lr_patience=4,
    min_delta=1e-4,
):
    splits = load_tensor_splits(data_dir)

    x_text_tr = splits["train"]["X_text"]
    in_ids_tr, mask_tr = x_text_tr[:, 0, :], x_text_tr[:, 1, :]
    x_num_tr = splits["train"]["X_num"]
    y_tr = splits["train"]["y"]

    x_text_val = splits["val"]["X_text"]
    in_ids_val, mask_val = x_text_val[:, 0, :], x_text_val[:, 1, :]
    x_num_val = splits["val"]["X_num"]
    y_val = splits["val"]["y"]

    x_text_ts = splits["test"]["X_text"]
    in_ids_ts, mask_ts = x_text_ts[:, 0, :], x_text_ts[:, 1, :]
    x_num_ts = splits["test"]["X_num"]
    y_ts = splits["test"]["y"]

    model = build_multimodal_model(
        max_text_length=x_text_tr.shape[2],
        sequence_length=x_num_tr.shape[1],
        num_features=x_num_tr.shape[2],
        dropout_rate=dropout_rate,
        text_units=text_units,
        lstm_units=lstm_units,
        fusion_units=fusion_units,
        l2_reg=l2_reg,
        learning_rate=learning_rate,
    )

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            min_delta=min_delta,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=reduce_lr_patience,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]


    hist = model.fit(
        [in_ids_tr, mask_tr, x_num_tr],
        y_tr,
        validation_data=([in_ids_val, mask_val, x_num_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
    )
    
    # baseline calc
    naive_preds = y_ts[:-1]
    actuals = y_ts[1:]
    naive_rmse = np.sqrt(np.mean((actuals - naive_preds) ** 2))
    print(f"Real-World Naive RMSE: {naive_rmse * 100:.4f}% daily volatility margin")

    test_res = model.evaluate([in_ids_ts, mask_ts, x_num_ts], y_ts, verbose=0, return_dict=True)
    test_mse = test_res.get("mse", test_res["loss"])
    test_rmse = np.sqrt(test_mse)
    print(f"Real-World Model RMSE: {test_rmse * 100:.4f}% daily volatility margin")

    if model_output_path:
        out_dir = os.path.dirname(model_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        model.save(model_output_path)
        print(f"Saved model to {model_output_path}")

    return model, hist, test_mse


def main():
    parser = argparse.ArgumentParser(description="Train the multimodal volatility model.")
    parser.add_argument("--data-dir", default='data/tensors', help="Directory with train/val/test .npz files")
    parser.add_argument(
        '--model-path',
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


if __name__ == '__main__':
    main()

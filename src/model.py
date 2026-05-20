"""Multi-modal volatility forecasting model (text + time series)."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_multimodal_model(
    text_embedding_dim: int = 768,
    sequence_length: int = 10,
    num_features: int = 1,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Build and compile the multi-modal Keras model.

    Args:
        text_embedding_dim: Dimension of the FinBERT embedding vector.
        sequence_length: Number of timesteps in the historical window.
        num_features: Number of numerical features per timestep.
        dropout_rate: Dropout rate for the text branch.

    Returns:
        A compiled tf.keras.Model.
    """

    # Branch A: text embeddings
    text_input = layers.Input(shape=(text_embedding_dim,), name="text_embedding")
    text_dense = layers.Dense(128, activation="relu", name="text_dense")(text_input)
    text_dropout = layers.Dropout(dropout_rate, name="text_dropout")(text_dense)

    # Branch B: time series window
    ts_input = layers.Input(shape=(sequence_length, num_features), name="timeseries")
    ts_lstm = layers.LSTM(64, return_sequences=False, name="ts_lstm")(ts_input)

    # Fusion + head
    fused = layers.Concatenate(name="fusion")([text_dropout, ts_lstm])
    dense_1 = layers.Dense(32, activation="relu", name="fusion_dense")(fused)
    output = layers.Dense(1, activation="linear", name="volatility_output")(dense_1)

    model = models.Model(inputs=[text_input, ts_input], outputs=output, name="multimodal_vol_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

    return model


if __name__ == "__main__":
    # Minimal smoke-test for model build.
    model = build_multimodal_model(text_embedding_dim=768, sequence_length=10, num_features=1)
    model.summary()

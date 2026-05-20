"""Multi-modal volatility forecasting model (text + time series)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_multimodal_model(
    text_embedding_dim: int = 768,
    sequence_length: int = 10,
    num_features: int = 1,
    dropout_rate: float = 0.2,
    text_units: int = 128,
    lstm_units: int = 96,
    fusion_units: int = 64,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile the multi-modal Keras model.

    Args:
        text_embedding_dim: Dimension of the FinBERT embedding vector.
        sequence_length: Number of timesteps in the historical window.
        num_features: Number of numerical features per timestep.
        dropout_rate: Dropout rate for the text branch.
        text_units: Hidden units for the text branch.
        lstm_units: Hidden units for the LSTM branch.
        fusion_units: Hidden units after modality fusion.
        l2_reg: L2 regularization strength for dense layers.
        learning_rate: Optimizer learning rate.

    Returns:
        A compiled tf.keras.Model.
    """

    # Branch A: text embeddings
    text_input = layers.Input(shape=(text_embedding_dim,), name="text_embedding")
    text_norm = layers.LayerNormalization(name="text_norm")(text_input)
    text_dense = layers.Dense(
        text_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="text_dense",
    )(text_norm)
    text_dropout = layers.Dropout(dropout_rate, name="text_dropout")(text_dense)

    # Branch B: time series window
    ts_input = layers.Input(shape=(sequence_length, num_features), name="timeseries")
    ts_norm = layers.LayerNormalization(name="ts_norm")(ts_input)
    ts_lstm = layers.LSTM(
        lstm_units,
        dropout=dropout_rate,
        return_sequences=False,
        name="ts_lstm",
    )(ts_norm)

    # Fusion + head
    fused = layers.Concatenate(name="fusion")([text_dropout, ts_lstm])
    fusion_dense = layers.Dense(
        fusion_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fusion_dense",
    )(fused)
    fusion_dropout = layers.Dropout(dropout_rate, name="fusion_dropout")(fusion_dense)
    output = layers.Dense(1, activation="linear", name="volatility_output")(fusion_dropout)

    model = models.Model(inputs=[text_input, ts_input], outputs=output, name="multimodal_vol_model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model


if __name__ == "__main__":
    # Minimal smoke-test for model build.
    model = build_multimodal_model(text_embedding_dim=768, sequence_length=10, num_features=1)
    model.summary()

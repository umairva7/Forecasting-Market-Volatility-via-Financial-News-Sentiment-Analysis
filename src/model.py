"""Multi-modal volatility forecasting model (text + time series)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from transformers import TFAutoModel


def build_multimodal_model(
    max_text_length: int = 64,
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
        max_text_length: Sequence length for the tokenized news headlines.
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

    # Branch A: text embeddings (Fine-tuning FinBERT)
    input_ids = layers.Input(shape=(max_text_length,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(max_text_length,), dtype=tf.int32, name="attention_mask")

    # Load pre-trained FinBERT (using from_pt=True if TF weights are missing locally)
    finbert = TFAutoModel.from_pretrained("ProsusAI/finbert", from_pt=True)

    # Freeze bottom 10 layers, leave top 2 unfrozen
    finbert.bert.embeddings.trainable = False
    for i in range(10):
        finbert.bert.encoder.layer[i].trainable = False

    # Get CLS token embedding from the last hidden state
    finbert_output = finbert(input_ids, attention_mask=attention_mask)
    cls_embedding = finbert_output.last_hidden_state[:, 0, :]

    text_norm = layers.LayerNormalization(name="text_norm")(cls_embedding)
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

    model = models.Model(inputs=[input_ids, attention_mask, ts_input], outputs=output, name="multimodal_vol_model")
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
    model = build_multimodal_model(max_text_length=64, sequence_length=10, num_features=1)
    model.summary()

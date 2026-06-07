import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from transformers import TFAutoModel
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_multimodal_model(
    max_text_length=64,
    sequence_length=10,
    num_features=1,
    dropout_rate=0.2,
    text_units=128,
    lstm_units=96,
    fusion_units=64,
    l2_reg=1e-4,
    learning_rate=1e-3,
):
    # build the model here

    # text embeddings stuff
    input_ids = layers.Input(shape=(max_text_length,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_text_length,), dtype=tf.int32, name="attention_mask")

    # load finbert
    finbert = TFAutoModel.from_pretrained('ProsusAI/finbert')

    # freeze bottom 10
    finbert.bert.embeddings.trainable = False
    for i in range(10):
        finbert.bert.encoder.layer[i].trainable = False

    # get cls token
    fb_out = finbert(input_ids, attention_mask=attention_mask)
    cls_emb = fb_out.last_hidden_state[:, 0, :]

    text_norm = layers.LayerNormalization(name='text_norm')(cls_emb)
    
    text_dense = layers.Dense(
        text_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="text_dense",
    )(text_norm)
    text_drop = layers.Dropout(dropout_rate, name='text_dropout')(text_dense)

    # time series window branch
    hist_data = layers.Input(shape=(sequence_length, num_features), name="timeseries")
    ts_norm = layers.LayerNormalization(name='ts_norm')(hist_data)
    ts_lstm = layers.LSTM(
        lstm_units,
        dropout=dropout_rate,
        return_sequences=False,
        name='ts_lstm',
    )(ts_norm)


    # fusion stuff
    fused = layers.Concatenate(name="fusion")([text_drop, ts_lstm])
    fusion_dense = layers.Dense(
        fusion_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name='fusion_dense',
    )(fused)
    
    fusion_drop = layers.Dropout(dropout_rate, name="fusion_dropout")(fusion_dense)
    pred_vol = layers.Dense(1, activation='linear', name="volatility_output")(fusion_drop)

    model = models.Model(inputs=[input_ids, attention_mask, hist_data], outputs=pred_vol, name='multimodal_vol_model')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model

if __name__ == "__main__":
    # test build
    model = build_multimodal_model(max_text_length=64, sequence_length=10, num_features=1)
    model.summary()

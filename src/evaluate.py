import argparse
import os

os.environ['TF_USE_LEGACY_KERAS'] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import TFAutoModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_multimodal_model

def evaluate_model(
    data_dir="data/tensors",
    model_path="models/multimodal_vol_model.keras",
    output_plot='output/volatility_forecast.png',
):
    print("Loading test data...")
    test_path = os.path.join(data_dir, 'test_data.npz')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Cannot find {test_path}")
        
    test_data = np.load(test_path)
    x_text = test_data["X_text"]
    in_ids = x_text[:, 0, :]
    mask = x_text[:, 1, :]
    x_num = test_data['X_num']
    y_ts = test_data["y"]

    print(f"Rebuilding model architecture...")
    model = build_multimodal_model(
        max_text_length=x_text.shape[2],
        sequence_length=x_num.shape[1],
        num_features=x_num.shape[2],
        dropout_rate=0.2,
        text_units=128,
        lstm_units=96,
        fusion_units=64,
    )
    
    print(f"Loading weights from {model_path}...")
    model.load_weights(model_path)

    print("Generating predictions...")
    preds = model.predict([in_ids, mask, x_num], verbose=0).flatten()


    # align arrays for baseline comparison
    actuals = y_ts[1:]
    naive_preds = y_ts[:-1]
    model_preds = preds[1:]

    # metrics
    mod_rmse = np.sqrt(mean_squared_error(actuals, model_preds))
    naiv_rmse = np.sqrt(mean_squared_error(actuals, naive_preds))
    mod_r2 = r2_score(actuals, model_preds)
    naiv_r2 = r2_score(actuals, naive_preds)

    print("\n--- Evaluation Metrics ---")
    print(f"Real-World Model RMSE: {mod_rmse * 100:.4f}% daily volatility margin")
    print(f"Real-World Naive RMSE: {naiv_rmse * 100:.4f}% daily volatility margin")
    print(f"Model R2:   {mod_r2:.4f}")
    print(f"Naive R2:   {naiv_r2:.4f}")

    # plot
    print("\nGenerating visualization...")
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    plt.plot(actuals * 100, label='True Next-Day Volatility', color="black", linewidth=1.5)
    plt.plot(model_preds * 100, label="Model Forecast", color='blue', alpha=0.8, linestyle="--")
    plt.plot(naive_preds * 100, label="Naive Baseline", color="red", alpha=0.5, linestyle=":")
    
    plt.title("Market Volatility Forecasting: Multimodal Network vs Baseline", fontsize=14, pad=15)
    plt.xlabel('Test Set Days (Chronological)', fontsize=12)
    plt.ylabel("Volatility (%)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_plot, dpi=300)
    print(f"Saved evaluation plot to: {output_plot}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the multimodal volatility model.")
    parser.add_argument("--data-dir", default='data/tensors')
    parser.add_argument("--model-path", default="models/multimodal_vol_model.keras")
    parser.add_argument("--output-plot", default="output/volatility_forecast.png")
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_path, args.output_plot)


if __name__ == '__main__':
    main()

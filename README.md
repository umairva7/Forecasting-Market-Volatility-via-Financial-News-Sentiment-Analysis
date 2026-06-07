# Multimodal Market Volatility Forecasting (News + Time Series)

This is my semester project. I built a pipeline that takes daily market data and financial news headlines, turns the text into FinBERT embeddings, creates time-series sequences, and trains a multi-input Keras model to predict what the market volatility will be tomorrow.

![Volatility Forecast](output/volatility_forecast.png)

I split the work into three phases:
1. Phase 1: Download market data and match it up with news headlines.
2. Phase 2: Create text embeddings, scale the numbers, build the sequences, and save the tensors.
3. Phase 3: Train the model, evaluate it, and save the final weights.

I also included an ethical analysis file (`ethical_analysis.md`) about the risks of using automated sentiment systems for trading.

## Table of contents

- Project goals
- Repository layout
- Data inputs
- End-to-end pipeline
- Outputs
- Environment setup
- Running the pipeline
- Detailed processing and modeling
- Reproducibility notes
- Limitations and missing pieces
- Troubleshooting
- Ethical analysis

## Project goals

- Merge financial news and market numbers into one dataset.
- Turn news text into numbers using FinBERT sentence embeddings.
- Predict next-day volatility using a dual-branch neural network.

## Repository layout

```text
.
├── README.md
├── ethical_analysis.md
├── requirement.txt
├── data/
│   ├── processed_market_news.csv
│   ├── DFN/
│   │   ├── analyst_ratings_processed.csv
│   │   ├── raw_analyst_ratings.csv
│   │   └── raw_partner_headlines.csv
│   ├── embeddings/
│   └── tensors/
│       ├── test_data.npz
│       ├── train_data.npz
│       └── val_data.npz
├── models/
│   └── multimodal_vol_model.keras
├── output/
│   └── volatility_forecast.png
└── src/
  ├── __init__.py
  ├── data_loader.py
  ├── model.py
  ├── preprocessing.py
  ├── evaluate.py
  └── train.py
```

Notes:
- The `data/DFN` folder should hold the Kaggle Daily Financial News dataset.
- Running the pipeline will overwrite the existing models and tensors.

## Data inputs

### Market data

- Source: `yfinance`
- Tickers: `^GSPC` (S&P 500) and `^VIX` (CBOE Volatility Index)
- Frequency: Daily
- Features: Close, High, Low, Open, Volume

The script normalizes the dates to trading days and adds the ticker name to the columns (like `^GSPC_Close`).

### News data

- Input file: `data/DFN/raw_partner_headlines.csv`
- What it does:
  - Drops rows with missing text.
  - Groups headlines by date and joins them with " | ".
  - Moves weekend/holiday news to the next open trading day so nothing is lost.

### Processed dataset schema

After phase 1, I save the data to `data/processed_market_news.csv`. It has columns like:
- `^GSPC_Close`, `^GSPC_High`, `^GSPC_Low`, `^GSPC_Open`, `^GSPC_Volume`
- `^VIX_Close`, `^VIX_High`, `^VIX_Low`, `^VIX_Open`, `^VIX_Volume`
- `headline`

## End-to-end pipeline

### Phase 1: Build the processed dataset

Script: `src/data_loader.py`

1. Download 5 years of market data.
2. Load the news CSV.
3. Align the headlines to trading days.
4. Merge them together.
5. Save to `data/processed_market_news.csv`.

### Phase 2: Preprocess and build tensors

Script: `src/preprocessing.py`

1. Load the merged dataset.
2. Calculate `targ_vol` as the 10-day rolling standard deviation of percentage returns. I used 10 days because the impact of financial news fades after two weeks.
3. Shift it by one day to get `next_day_vol`.
4. Tokenize the text using FinBERT.
5. Scale the numbers using `MinMaxScaler` (only fitting on the train set to prevent data leakage).
6. Build sequences of length 10.
7. Split into Train/Val/Test chronologically.
8. Save arrays and the scaler to `data/tensors/`.

### Phase 3: Train the model

Script: `src/train.py`

1. Load the train/val/test splits.
2. Build the Keras model.
3. Train it with early stopping and a learning-rate reducer.
4. Save it to `models/multimodal_vol_model.keras`.

### Phase 4: Evaluate the model

Script: `src/evaluate.py`

1. Load the saved weights and the test set.
2. Generate predictions.
3. Calculate real-world metrics like RMSE and R² against a Naive Baseline (which just predicts tomorrow is the same as today).
4. Plot the graph and save it to `output/volatility_forecast.png`.

## Outputs

- `data/processed_market_news.csv`: the merged data
- `data/tensors/*_data.npz`: the NumPy arrays
- `data/tensors/scaler.pkl`: the fitted scaler
- `models/multimodal_vol_model.keras`: the final model
- `output/volatility_forecast.png`: the results graph

## Environment setup

I used these libraries:
- `transformers`
- `tensorflow`
- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

Just run:
```bash
python -m pip install -r requirement.txt
```

## Running the pipeline

Run them in order from the root directory:
```bash
python src/data_loader.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py
```

## Reproducibility notes

- I didn't set random seeds everywhere, so the exact numbers might shift slightly if you retrain it, but the overall performance stays the same.

## Limitations

- You have to download the news dataset from Kaggle manually.
- I didn't do any crazy hyperparameter tuning because training the Transformer takes too long.

## Troubleshooting

- If it crashes on phase 1, make sure `data/DFN/raw_partner_headlines.csv` is actually there.
- If TensorFlow gives you a segmentation fault, make sure you keep `TF_USE_LEGACY_KERAS=1` and `TF_CPP_MIN_LOG_LEVEL=3` at the top of the scripts.

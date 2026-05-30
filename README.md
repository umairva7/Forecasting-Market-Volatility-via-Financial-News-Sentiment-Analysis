# Multimodal Market Volatility Forecasting (News + Time Series)

This project builds an end-to-end pipeline that merges daily market data with daily news headlines, extracts FinBERT text embeddings, builds time series sequences, and trains a multi-input Keras model to predict next-day volatility.

The pipeline is organized into three phases:

1. Phase 1: Download market data and align it with news headlines.
2. Phase 2: Create embeddings, scale numeric features, build sequences, and save tensors.
3. Phase 3: Train and evaluate the multimodal model and save it.

The project also includes an ethical analysis document that discusses risks of automated sentiment-driven systems.

## Table of contents

- Project goals
- Repository layout
- Data inputs
- End-to-end pipeline
- Outputs
- Environment setup
- Running the pipeline
- CLI reference
- Detailed processing and modeling
- Reproducibility notes
- Limitations and missing pieces
- Troubleshooting
- Ethical analysis

## Project goals

- Combine financial news and market data into a single dataset.
- Encode news using FinBERT sentence embeddings.
- Use a multi-input neural network to predict next-day volatility.
- Provide a simple CLI to run each pipeline phase or the full pipeline.

## Repository layout

```
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
│       ├── test_data.pt
│       ├── train_data.pt
│       └── val_data.pt
├── models/
│   └── multimodal_vol_model.keras
├── output/
└── src/
		├── __init__.py
		├── data_loader.py
		├── model.py
		├── preprocessing.py
		└── train.py
```

Notes:

- The repository already contains example outputs under data/ and models/. Running the pipeline will overwrite some of them.
- The dataset files under data/DFN are assumed to be the Kaggle Daily Financial News dataset (or a compatible schema).

## Data inputs

### Market data

- Provider: yfinance
- Tickers: ^GSPC (S&P 500) and ^VIX (CBOE Volatility Index)
- Frequency: daily
- Fields used: Close, High, Low, Open, Volume (if available)

The market data downloader normalizes the date index to trading days and prefixes columns with the ticker name (for example, ^GSPC_Close).

### News data

- Input file: data/DFN/raw_partner_headlines.csv
- Required columns: date and headline (or title)
- Behavior:
	- Rows with missing date/headline are dropped.
	- Headlines are grouped by date into a single string separated by " | ".
	- Weekend and holiday headlines are rolled forward to the next trading day.

If the news CSV is missing, phase 1 will raise a FileNotFoundError.

### Processed dataset schema

After phase 1, the merged dataset is saved to data/processed_market_news.csv. Typical columns include:

- ^GSPC_Close, ^GSPC_High, ^GSPC_Low, ^GSPC_Open, ^GSPC_Volume
- ^VIX_Close, ^VIX_High, ^VIX_Low, ^VIX_Open, ^VIX_Volume
- headline

## End-to-end pipeline

### Phase 1: Build the processed dataset

Script: src/data_loader.py

1. Download market data for ^GSPC and ^VIX over a specified period.
2. Load the news CSV and group headlines by date.
3. Align headlines to trading days (roll forward).
4. Merge market data and headlines.
5. Save to data/processed_market_news.csv.

Key defaults:

- Period: 5 years (period="5y")
- Output: data/processed_market_news.csv

### Phase 2: Preprocess and build tensors

Script: src/preprocessing.py

1. Load the processed dataset.
2. Compute the 10-day rolling standard deviation of ^GSPC_Close as target_volatility. (Note: A 10-day rolling window corresponds to two working weeks in the stock market. This window was chosen because the market impact of the financial news cycle typically decays after two weeks).
3. Shift target_volatility by one day to create next_day_volatility.
4. Drop rows with missing targets.
5. Tokenize each day's headline text for FinBERT.
6. Scale numeric features with MinMaxScaler (fitted only on the training set to prevent leakage).
7. Build fixed-length sequences (default length 10).
8. Split into train/val/test chronologically.
9. Save NumPy arrays and scaler to data/tensors/ and data/tensors/scaler.pkl.

### Phase 3: Train the model

Script: src/train.py

1. Load train/val/test tensors from data/tensors.
2. Build the multimodal Keras model.
3. Train with early stopping and learning-rate reduction.
4. Evaluate on the test split.
5. Save the model to models/multimodal_vol_model.keras.

## Outputs

- data/processed_market_news.csv: merged market + news dataset
- data/tensors/{train,val,test}_data.npz: NumPy arrays for each split
- data/tensors/scaler.pkl: fitted MinMaxScaler
- models/multimodal_vol_model.keras: trained Keras model

## Environment setup

### Python and dependencies

The project requires common scientific Python libraries plus TensorFlow:

- transformers
- tensorflow
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies:

```
python -m pip install -r requirement.txt
```

Notes:

- The first run of FinBERT will download model weights from Hugging Face.

## Running the pipeline

All commands below assume your working directory is the project root.

### Run one file at a time

```
# Phase 1: build processed dataset
python src/data_loader.py

# Phase 2: preprocess and save tensors
python src/preprocessing.py

# Phase 3: train model
python src/train.py

# Optional: model architecture smoke test
python src/model.py
```

### Train options

```
python src/train.py \
	--tensors-dir data/tensors \
	--model-path models/multimodal_vol_model.keras \
	--epochs 50 \
	--batch-size 32 \
	--learning-rate 0.001 \
	--dropout-rate 0.2 \
	--text-units 128 \
	--lstm-units 96 \
	--fusion-units 64 \
	--l2-reg 0.0001 \
	--early-stop-patience 10 \
	--reduce-lr-patience 4 \
	--min-delta 0.0001
```

### Data loader details (src/data_loader.py)

- Market data is downloaded via yfinance with interval="1d" and auto_adjust=False.
- The data loader keeps Close, High, Low, Open, Volume when present.
- News headlines are grouped by date and concatenated with " | ".
- News on non-trading days is rolled forward to the next trading day.
- Rows with missing ^GSPC_Close are dropped, and missing headlines are filled with an empty string.

### Preprocessing details (src/preprocessing.py)

Target engineering:

- target_volatility = rolling 10-day standard deviation of ^GSPC_Close. (A 10-day window represents two trading weeks, chosen because the relevance of financial news typically decays significantly after this period.)
- next_day_volatility = target_volatility shifted by -1 day.

Tokenization:

- FinBERT model: ProsusAI/finbert (AutoTokenizer).
- Tokenization outputs: input_ids and attention_mask (padded to 64 tokens).

Feature scaling:

- All numeric columns except headline, target_volatility, and next_day_volatility are scaled with MinMaxScaler.
- The scaler is saved to data/tensors/scaler.pkl.

Sequence construction:

- Each sample contains a numeric window of length seq_length (default 10).
- The text embedding used is the embedding for the last day in the window.
- The target label is next_day_volatility for the same last day.

Splits:

- Train: first 70 percent of samples
- Val: next 15 percent of samples
- Test: remaining 15 percent

### Model details (src/model.py)

Inputs:

- Tokenized text: input_ids and attention_mask vectors (shape max_text_length)
- Numeric time series window: shape (seq_length, num_features)

Architecture:

- Text branch: LayerNorm -> Dense(ReLU) -> Dropout
- Time series branch: LayerNorm -> LSTM
- Fusion: Concatenate -> Dense(ReLU) -> Dropout -> Dense(1)

Loss and metrics:

- Loss: mean squared error (MSE)
- Metrics: MSE and MAE

### Training details (src/train.py)

- Loads NumPy arrays from data/tensors.
- Uses EarlyStopping and ReduceLROnPlateau callbacks.
- Prints test MSE after training.
- Saves model to models/multimodal_vol_model.keras.

## Reproducibility notes

- Random seeds are not set in the current code.
- FinBERT tokenization is deterministic for identical inputs.
- TensorFlow training may still be nondeterministic unless you set explicit seeds and deterministic ops.

## Limitations and missing pieces

- Data acquisition is manual. The pipeline expects data/DFN/raw_partner_headlines.csv to be present.
- There is no inference or evaluation report script beyond the printed test MSE.
- Hyperparameter tuning, feature selection, and benchmarking are not implemented.

## Troubleshooting

- Missing news file: ensure data/DFN/raw_partner_headlines.csv exists and has date + headline columns.
- Empty market data: yfinance may return empty data if there is no network or the ticker is unavailable.
- FinBERT download issues: check your internet connection and Hugging Face access.
- Memory errors: reduce batch size or run on CPU-only.

## Ethical analysis

See ethical_analysis.md for a discussion of systemic risks, feedback loops, manipulation, and mitigation strategies in automated sentiment-driven trading systems.

## License

No license is specified. Add one if you plan to distribute this project.

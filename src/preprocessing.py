import os
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path='data/processed_market_news.csv'):
    """
    Loads the processed market and news data.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df

def _normalize_headlines(headlines):
    return [
        str(text) if pd.notna(text) and str(text).strip() != "" else "No news today."
        for text in headlines
    ]


def tokenize_finbert_headlines(
    headlines,
    max_length: int = 64,
):
    """
    Tokenize headlines for FinBERT, returning (N, 2, max_length) array 
    where index 0 is input_ids and index 1 is attention_mask.
    """
    normalized_headlines = _normalize_headlines(headlines)
    
    print(f"Tokenizing {len(normalized_headlines)} days of news...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    inputs = tokenizer(
        normalized_headlines,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Stack along axis 1 -> (N, 2, max_length)
    tokens_stacked = np.stack([input_ids, attention_mask], axis=1)
    return tokens_stacked

def preprocess_features(
    df,
    target_col='^GSPC_Close',
    seq_length=10,
    train_ratio: float = 0.7,
):
    """
    Calculates target volatility, scales features, and creates sequence tensors.
    Each sample includes a numeric window and the corresponding day's news embedding.
    """
    print("Preprocessing numerical features...")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if 'headline' not in df.columns:
        raise KeyError("Missing 'headline' column in dataframe.")

    df = df.copy()

    # 1. Calculate the 10-day rolling standard deviation of the close prices as the target volatility
    df['target_volatility'] = df[target_col].rolling(window=10).std()
    
    # Shift target to be next-day's volatility (t+1)
    df['next_day_volatility'] = df['target_volatility'].shift(-1)
    
    # Drop rows that have NaN values (initial rolling window period and the last row)
    df = df.dropna(subset=['target_volatility', 'next_day_volatility']).copy()
    
    # 2. Extract news tokens
    tokens = tokenize_finbert_headlines(
        df['headline'].tolist(),
        max_length=64,
    )
    
    # 3. Scale numerical features (Fix: Fit scaler only on train split)
    feature_cols = [c for c in df.columns if c not in ['headline', 'target_volatility', 'next_day_volatility']]
    
    print(f"Numerical features used: {feature_cols}")
    
    # Calculate chronological train end index based on sequences
    n_samples = len(df) - seq_length + 1
    train_end_seq = int(n_samples * train_ratio)
    # The training sequences cover df indices up to train_end_seq + seq_length - 1
    train_end_df = train_end_seq + seq_length - 1
    
    scaler = MinMaxScaler()
    # Fit only on the training portion to prevent data leakage
    scaler.fit(df.iloc[:train_end_df][feature_cols])
    # Transform entire dataset
    scaled_features = scaler.transform(df[feature_cols])
    
    # 4. Create Sequences
    print(f"Creating sequences of length {seq_length}...")
    X_num = []
    X_text = []
    y = []

    for i in range(n_samples):
        X_num.append(scaled_features[i : i + seq_length])
        X_text.append(tokens[i + seq_length - 1])
        y.append(df['next_day_volatility'].iloc[i + seq_length - 1])

    X_num = np.array(X_num)
    X_text = np.array(X_text)
    y = np.array(y)
    
    return X_num, X_text, y, scaler

def create_data_splits(X_num, X_text, y, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the data into training, validation, and testing sets temporally.
    """
    n_samples = len(y)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    splits = {
        'train': {
            'X_num': X_num[:train_end].astype(np.float32),
            'X_text': X_text[:train_end].astype(np.int32),
            'y': y[:train_end].astype(np.float32)
        },
        'val': {
            'X_num': X_num[train_end:val_end].astype(np.float32),
            'X_text': X_text[train_end:val_end].astype(np.int32),
            'y': y[train_end:val_end].astype(np.float32)
        },
        'test': {
            'X_num': X_num[val_end:].astype(np.float32),
            'X_text': X_text[val_end:].astype(np.int32),
            'y': y[val_end:].astype(np.float32)
        }
    }
    
    print(f"Data splits created:")
    print(f"Train: {splits['train']['X_num'].shape[0]} samples")
    print(f"Val:   {splits['val']['X_num'].shape[0]} samples")
    print(f"Test:  {splits['test']['X_num'].shape[0]} samples")
    print(f"Shape of X_num: {splits['train']['X_num'].shape[1:]}")
    print(f"Shape of X_text: {splits['train']['X_text'].shape[1:]}")
    
    return splits


def save_splits(splits, scaler, output_dir='data/tensors'):
    """
    Save numpy arrays and the scaler for downstream training.
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in splits.items():
        np.savez(os.path.join(output_dir, f"{split_name}_data.npz"), **data)

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as handle:
        pickle.dump(scaler, handle)

if __name__ == "__main__":
    df = load_data()
    
    X_num, X_text, y, scaler = preprocess_features(df, target_col='^GSPC_Close', seq_length=10)
    
    splits = create_data_splits(X_num, X_text, y)
    
    save_splits(splits, scaler)
    print("Preprocessing complete. Tensors and scaler saved to data/tensors/")

import os
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path='data/processed_market_news.csv'):
    # load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df

def _normalize_headlines(headlines):
    return [
        str(t) if pd.notna(t) and str(t).strip() != "" else "No news today."
        for t in headlines
    ]


def tokenize_finbert_headlines(
    headlines,
    max_length=64,
):
    # tokenize for finbert
    norm_heads = _normalize_headlines(headlines)
    
    print(f"Tokenizing {len(norm_heads)} days of news...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    
    inputs = tokenizer(
        norm_heads,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    
    in_ids = inputs["input_ids"]
    mask = inputs['attention_mask']
    
    # stack them up
    toks_stacked = np.stack([in_ids, mask], axis=1)
    return toks_stacked

def prep_feats(
    df,
    target_col='^GSPC_Close',
    seq_length=10,
    train_ratio=0.7,
):
    print("Preprocessing numerical features...")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if 'headline' not in df.columns:
        raise KeyError("Missing 'headline' column in dataframe.")

    df = df.copy()

    # calc 10 day rolling std dev of returns
    df['returns'] = df[target_col].pct_change()
    df['targ_vol'] = df['returns'].rolling(window=10).std()
    
    # shift to next day
    df['next_day_vol'] = df['targ_vol'].shift(-1)
    
    # drop nans
    df = df.dropna(subset=['targ_vol', 'next_day_vol']).copy()
    
    # get news tokens
    toks = tokenize_finbert_headlines(
        df['headline'].tolist(),
        max_length=64,
    )
    
    # fit scaler only on train
    feat_cols = [c for c in df.columns if c not in ['headline', 'targ_vol', 'next_day_vol']]
    
    print(f"Numerical features used: {feat_cols}")
    
    # find split idx
    n_samps = len(df) - seq_length + 1
    tr_end_seq = int(n_samps * train_ratio)
    tr_end_df = tr_end_seq + seq_length - 1
    
    scaler = MinMaxScaler()
    scaler.fit(df.iloc[:tr_end_df][feat_cols])
    
    scaled_feats = scaler.transform(df[feat_cols])
    
    # make seqs
    print(f"Creating sequences of length {seq_length}...")
    x_num = []
    x_text = []
    y = []

    for i in range(n_samps):
        x_num.append(scaled_feats[i : i + seq_length])
        x_text.append(toks[i + seq_length - 1])
        y.append(df['next_day_vol'].iloc[i + seq_length - 1])

    x_num = np.array(x_num)
    x_text = np.array(x_text)
    y = np.array(y)
    
    return x_num, x_text, y, scaler

def create_data_splits(x_num, x_text, y, train_ratio=0.7, val_ratio=0.15):
    # split temporally
    n_samps = len(y)
    tr_end = int(n_samps * train_ratio)
    val_end = int(n_samps * (train_ratio + val_ratio))
    
    splits = {
        'train': {
            'X_num': x_num[:tr_end].astype(np.float32),
            'X_text': x_text[:tr_end].astype(np.int32),
            'y': y[:tr_end].astype(np.float32)
        },
        'val': {
            'X_num': x_num[tr_end:val_end].astype(np.float32),
            'X_text': x_text[tr_end:val_end].astype(np.int32),
            'y': y[tr_end:val_end].astype(np.float32)
        },
        'test': {
            'X_num': x_num[val_end:].astype(np.float32),
            'X_text': x_text[val_end:].astype(np.int32),
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
    # save arrays and scaler
    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in splits.items():
        np.savez(os.path.join(output_dir, f"{split_name}_data.npz"), **data)

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    df = load_data()
    
    x_num, x_text, y, scaler = prep_feats(df, target_col='^GSPC_Close', seq_length=10)
    
    splits = create_data_splits(x_num, x_text, y)
    
    save_splits(splits, scaler)
    print("Preprocessing complete. Tensors and scaler saved to data/tensors/")

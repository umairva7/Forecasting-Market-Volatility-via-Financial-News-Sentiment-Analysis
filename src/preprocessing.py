import os
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_embedding_cache(cache_path: str) -> dict:
    if not cache_path:
        return {}

    path = Path(cache_path)
    if not path.exists():
        return {}

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return {}

    if isinstance(payload, dict) and "embeddings" in payload:
        return payload["embeddings"]
    if isinstance(payload, dict):
        return payload

    return {}


def _save_embedding_cache(cache_path: str, cache_map: dict) -> None:
    if not cache_path:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"embeddings": cache_map}, handle)


def extract_finbert_embeddings(
    headlines,
    batch_size: int = 32,
    cache_path: str = "data/embeddings/finbert_embeddings.pkl",
    use_cache: bool = True,
):
    """
    Extract 768-dimensional sentence embeddings using the pre-trained FinBERT model.
    """
    normalized_headlines = _normalize_headlines(headlines)
    hashes = [_hash_text(text) for text in normalized_headlines]
    cache_map = _load_embedding_cache(cache_path) if use_cache else {}

    embeddings = [None] * len(normalized_headlines)
    missing_indices = []

    for idx, text_hash in enumerate(hashes):
        if use_cache and text_hash in cache_map:
            embeddings[idx] = np.asarray(cache_map[text_hash])
        else:
            missing_indices.append(idx)

    if missing_indices:
        print("Loading FinBERT model from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModel.from_pretrained("ProsusAI/finbert")

        # Freeze the model parameters
        for param in model.parameters():
            param.requires_grad = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        model.eval()

        missing_texts = [normalized_headlines[i] for i in missing_indices]
        print(f"Extracting embeddings for {len(missing_texts)} days of news...")

        for batch_start in range(0, len(missing_texts), batch_size):
            batch_texts = missing_texts[batch_start : batch_start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for offset, embedding in enumerate(cls_embeddings):
                idx = missing_indices[batch_start + offset]
                embeddings[idx] = embedding.astype(np.float32)
                if use_cache:
                    cache_map[hashes[idx]] = embeddings[idx]

        if use_cache:
            _save_embedding_cache(cache_path, cache_map)
    else:
        print("Using cached FinBERT embeddings.")

    if embeddings:
        return np.vstack(embeddings)

    return np.empty((0, 768))

def preprocess_features(
    df,
    target_col='^GSPC_Close',
    seq_length=10,
    embedding_cache_path: str = "data/embeddings/finbert_embeddings.pkl",
    use_embedding_cache: bool = True,
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
    
    # 2. Extract news embeddings (We do this after dropna to save compute on unused rows)
    embeddings = extract_finbert_embeddings(
        df['headline'],
        cache_path=embedding_cache_path,
        use_cache=use_embedding_cache,
    )
    
    # 3. Scale numerical features
    feature_cols = [c for c in df.columns if c not in ['headline', 'target_volatility', 'next_day_volatility']]
    
    print(f"Numerical features used: {feature_cols}")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    # 4. Create Sequences
    print(f"Creating sequences of length {seq_length}...")
    X_num = []
    X_text = []
    y = []

    for i in range(len(df) - seq_length + 1):
        X_num.append(scaled_features[i : i + seq_length])
        X_text.append(embeddings[i + seq_length - 1])
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
            'X_num': torch.tensor(X_num[:train_end], dtype=torch.float32),
            'X_text': torch.tensor(X_text[:train_end], dtype=torch.float32),
            'y': torch.tensor(y[:train_end], dtype=torch.float32)
        },
        'val': {
            'X_num': torch.tensor(X_num[train_end:val_end], dtype=torch.float32),
            'X_text': torch.tensor(X_text[train_end:val_end], dtype=torch.float32),
            'y': torch.tensor(y[train_end:val_end], dtype=torch.float32)
        },
        'test': {
            'X_num': torch.tensor(X_num[val_end:], dtype=torch.float32),
            'X_text': torch.tensor(X_text[val_end:], dtype=torch.float32),
            'y': torch.tensor(y[val_end:], dtype=torch.float32)
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
    Save torch tensors and the scaler for downstream training.
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in splits.items():
        torch.save(data, os.path.join(output_dir, f"{split_name}_data.pt"))

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as handle:
        pickle.dump(scaler, handle)

if __name__ == "__main__":
    df = load_data()
    
    X_num, X_text, y, scaler = preprocess_features(df, target_col='^GSPC_Close', seq_length=10)
    
    splits = create_data_splits(X_num, X_text, y)
    
    save_splits(splits, scaler)
    print("Preprocessing complete. Tensors and scaler saved to data/tensors/")

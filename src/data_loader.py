"""Load market data and align daily news to trading days."""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ("^GSPC", "^VIX")
DEFAULT_NEWS_PATH = os.path.join("data", "DFN", "raw_partner_headlines.csv")


def download_market_data(tickers: Iterable[str] = DEFAULT_TICKERS, period: str = "5y") -> pd.DataFrame:
    """Download daily market data for the requested tickers."""
    market_frames = []

    for ticker in tickers:
        df_ticker = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        if df_ticker.empty:
            continue

        if isinstance(df_ticker.columns, pd.MultiIndex):
            df_ticker.columns = [col[0] for col in df_ticker.columns]

        cols_to_keep = ["Close", "High", "Low", "Open", "Volume"]
        cols_to_keep = [c for c in cols_to_keep if c in df_ticker.columns]
        df_ticker = df_ticker[cols_to_keep]
        df_ticker.columns = [f"{ticker}_{col}" for col in df_ticker.columns]
        market_frames.append(df_ticker)

    if not market_frames:
        return pd.DataFrame()

    market_df = pd.concat(market_frames, axis=1, join="outer")
    market_df.index = pd.to_datetime(market_df.index).normalize()
    return market_df.sort_index()


def load_kaggle_news(file_path: str = DEFAULT_NEWS_PATH) -> pd.DataFrame:
    """Load the Kaggle Daily Financial News dataset and group headlines by date."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"News file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    headline_col = "headline" if "headline" in df.columns else "title" if "title" in df.columns else None
    date_col = "date" if "date" in df.columns else None
    if not headline_col or not date_col:
        raise ValueError("News file must contain 'headline' and 'date' columns.")

    df = df[[date_col, headline_col]].dropna()
    df = df.rename(columns={date_col: "date", headline_col: "headline"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()

    grouped = df.groupby("date")["headline"].apply(lambda x: " | ".join(x.astype(str))).to_frame()
    grouped.index.name = "date"
    return grouped.sort_index()


def align_news_to_trading_days(market_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """Roll weekend/holiday headlines forward to the next trading day."""
    if market_df.empty:
        return market_df

    trading_days = pd.DatetimeIndex(market_df.index).sort_values()

    if news_df.empty:
        aligned_news = pd.DataFrame(index=trading_days, data={"headline": ""})
        return market_df.join(aligned_news, how="left")

    news_dates = pd.DatetimeIndex(news_df.index)
    positions = trading_days.searchsorted(news_dates, side="left")
    valid_mask = positions < len(trading_days)

    mapped_news = news_df.iloc[valid_mask].copy()
    mapped_news["trading_date"] = trading_days.take(positions[valid_mask])
    grouped = mapped_news.groupby("trading_date")["headline"].apply(lambda x: " | ".join(x.astype(str))).to_frame()
    aligned_news = grouped.reindex(trading_days).fillna("")

    return market_df.join(aligned_news, how="left")


def build_processed_dataset(
    news_path: str = DEFAULT_NEWS_PATH,
    output_path: str = os.path.join("data", "processed_market_news.csv"),
    period: str = "5y",
) -> pd.DataFrame:
    """Download market data, align news, and persist a merged dataset."""
    market_data = download_market_data(tickers=DEFAULT_TICKERS, period=period)
    news_data = load_kaggle_news(news_path)
    merged = align_news_to_trading_days(market_data, news_data)

    merged = merged.dropna(subset=["^GSPC_Close"])
    merged["headline"] = merged["headline"].fillna("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_csv(output_path)
    return merged


if __name__ == "__main__":
    merged_data = build_processed_dataset()
    print(f"Processed dataset saved with shape: {merged_data.shape}")

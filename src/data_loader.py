from __future__ import annotations

import os
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ('^GSPC', "^VIX")
DEFAULT_NEWS_PATH = os.path.join("data", "DFN", 'raw_partner_headlines.csv')

def download_market_data(tickers=DEFAULT_TICKERS, period="5y"):
    mkt_frames = []

    for tick in tickers:
        df_tick = yf.download(tick, period=period, interval="1d", auto_adjust=False, progress=False)
        if df_tick.empty:
            continue

        if isinstance(df_tick.columns, pd.MultiIndex):
            df_tick.columns = [col[0] for col in df_tick.columns]


        cols_keep = ['Close', "High", "Low", "Open", 'Volume']
        cols_keep = [c for c in cols_keep if c in df_tick.columns]
        df_tick = df_tick[cols_keep]
        df_tick.columns = [f"{tick}_{col}" for col in df_tick.columns]
        mkt_frames.append(df_tick)

    if not mkt_frames:
        return pd.DataFrame()

    mkt_data = pd.concat(mkt_frames, axis=1, join="outer")
    mkt_data.index = pd.to_datetime(mkt_data.index).normalize()
    return mkt_data.sort_index()


def load_kaggle_news(file_path=DEFAULT_NEWS_PATH):
    # load news
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"News file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    head_col = "headline" if "headline" in df.columns else "title" if 'title' in df.columns else None
    date_col = 'date' if "date" in df.columns else None
    if not head_col or not date_col:
        raise ValueError("News file must contain 'headline' and 'date' columns.")

    df = df[[date_col, head_col]].dropna()
    df = df.rename(columns={date_col: "date", head_col: 'headline'})

    df['date'] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=['date'])
    df["date"] = df['date'].dt.tz_localize(None).dt.normalize()

    grouped = df.groupby("date")["headline"].apply(lambda x: " | ".join(x.astype(str))).to_frame()
    grouped.index.name = "date"
    return grouped.sort_index()


def align_news_to_trading_days(mkt_data, nws_data):
    # align dates
    if mkt_data.empty:
        return mkt_data

    trading_days = pd.DatetimeIndex(mkt_data.index).sort_values()

    if nws_data.empty:
        aligned_nws = pd.DataFrame(index=trading_days, data={"headline": ""})
        return mkt_data.join(aligned_nws, how="left")

    nws_dates = pd.DatetimeIndex(nws_data.index)
    pos = trading_days.searchsorted(nws_dates, side='left')
    valid = pos < len(trading_days)

    mapped_nws = nws_data.iloc[valid].copy()
    mapped_nws['trading_date'] = trading_days.take(pos[valid])
    grouped = mapped_nws.groupby("trading_date")["headline"].apply(lambda x: " | ".join(x.astype(str))).to_frame()
    
    aligned_nws = grouped.reindex(trading_days).fillna("")

    return mkt_data.join(aligned_nws, how='left')


def build_processed_dataset(
    news_path=DEFAULT_NEWS_PATH,
    output_path=os.path.join("data", "processed_market_news.csv"),
    period='5y',
):
    
    mkt = download_market_data(tickers=DEFAULT_TICKERS, period=period)
    nws = load_kaggle_news(news_path)
    merged = align_news_to_trading_days(mkt, nws)

    merged = merged.dropna(subset=['^GSPC_Close'])
    merged['headline'] = merged['headline'].fillna("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_csv(output_path)
    return merged


if __name__ == "__main__":
    m_data = build_processed_dataset()
    print(f"Processed dataset saved with shape: {m_data.shape}")

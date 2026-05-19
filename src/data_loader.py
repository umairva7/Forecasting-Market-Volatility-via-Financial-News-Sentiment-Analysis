import os
import pandas as pd
import yfinance as yf
from datetime import timedelta

def download_market_data(tickers=["^GSPC", "^VIX"], period="5y"):
    """
    Downloads historical market data using yfinance.
    Returns a unified DataFrame with dates as the index.
    """
    print(f"Downloading market data for {tickers} over the past {period}...")
    market_df = pd.DataFrame()
    
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df_ticker = yf.download(ticker, period=period)
        
        # Rename columns to include ticker
        cols_to_keep = ['Close', 'High', 'Low', 'Open', 'Volume']
        # Filter only existing columns just in case
        cols_to_keep = [c for c in cols_to_keep if c in df_ticker.columns]
        df_ticker = df_ticker[cols_to_keep]
        df_ticker.columns = [f"{ticker}_{col}" for col in df_ticker.columns]
        
        if market_df.empty:
            market_df = df_ticker
        else:
            market_df = market_df.join(df_ticker, how='outer')
            
    # Normalize index to date (removing time components)
    market_df.index = pd.to_datetime(market_df.index).normalize()
    return market_df

def load_news_data(file_paths):
    """
    Loads and combines news from given CSV files.
    """
    print(f"Loading news data from {file_paths}...")
    dfs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found.")
            continue
        try:
            print(f"Parsing {path}...")
            # Using on_bad_lines for modern pandas, or error_bad_lines for older versions
            try:
                df = pd.read_csv(path, on_bad_lines='skip')
            except TypeError:
                df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False)
            
            # Identify headline and date columns
            headline_col = 'title' if 'title' in df.columns else 'headline' if 'headline' in df.columns else None
            date_col = 'date' if 'date' in df.columns else None
            
            if headline_col and date_col:
                df = df[[date_col, headline_col]].dropna()
                df.rename(columns={date_col: 'date', headline_col: 'headline'}, inplace=True)
                dfs.append(df)
            else:
                print(f"Could not find required columns in {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    if not dfs:
        print("No valid data loaded from the provided CSV files.")
        return pd.DataFrame(columns=['date', 'headline']).set_index('date')
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Parse dates
    print("Parsing dates. This might take a moment...")
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce', utc=True)
    combined_df = combined_df.dropna(subset=['date'])
    
    # Normalize to date only
    combined_df['date'] = combined_df['date'].dt.tz_localize(None).dt.normalize()
    
    # Group by date and join headlines
    print("Grouping news by date...")
    grouped_news = combined_df.groupby('date')['headline'].apply(lambda x: ' | '.join(x.astype(str))).reset_index()
    grouped_news.set_index('date', inplace=True)
    
    return grouped_news

def merge_and_align_data(market_df, news_df):
    """
    Aligns news data with trading days.
    News from weekends/holidays are rolled forward and accumulated 
    to the next available trading day.
    """
    print("Aligning news with trading days...")
    
    # Ensure both dataframes have valid indices
    if market_df.empty or news_df.empty:
        print("Market data or news data is empty. Cannot merge.")
        return market_df
    
    # Create a unified date range spanning the minimum and maximum dates
    min_date = min(market_df.index.min(), news_df.index.min())
    max_date = max(market_df.index.max(), news_df.index.max())
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
                              
    trading_days = set(market_df.index)
    aligned_data = []
    accumulated_headlines = []
    
    # Iterate through all dates in chronological order
    for current_date in all_dates:
        if current_date in news_df.index:
            headline = news_df.loc[current_date, 'headline']
            if pd.notna(headline) and str(headline).strip() != "":
                accumulated_headlines.append(str(headline))
                
        if current_date in trading_days:
            # It's a trading day, assign accumulated headlines (which includes today's + past weekends)
            if accumulated_headlines:
                combined_headline = " | ".join(accumulated_headlines)
                aligned_data.append({'date': current_date, 'headline': combined_headline})
                accumulated_headlines = [] # Reset after assigning
            else:
                aligned_data.append({'date': current_date, 'headline': ""})
                
    aligned_news_df = pd.DataFrame(aligned_data)
    if not aligned_news_df.empty:
        aligned_news_df.set_index('date', inplace=True)
    else:
        aligned_news_df = pd.DataFrame(columns=['headline'], index=market_df.index)
        
    # Merge with market data
    final_df = market_df.join(aligned_news_df, how='left')
    
    return final_df

if __name__ == "__main__":
    # 1. Download Market Data
    market_data = download_market_data(tickers=["^GSPC", "^VIX"], period="5y")
    print(f"Market data shape: {market_data.shape}")
    
    # 2. Load News Data
    # Path assumes we are running from project root
    news_files = [
        'data/DFN/analyst_ratings_processed.csv',
        'data/DFN/raw_partner_headlines.csv',
        'data/DFN/raw_analyst_ratings.csv'
    ]
    
    news_data = load_news_data(news_files)
    print(f"News data shape: {news_data.shape}")
    
    # 3. Merge and Align
    merged_data = merge_and_align_data(market_data, news_data)
    
    # Filter to only trading days where we have market data
    merged_data = merged_data.dropna(subset=['^GSPC_Close'])
    
    # Fill missing headlines with empty string
    merged_data['headline'] = merged_data['headline'].fillna("")
    
    # Save the output
    os.makedirs('data', exist_ok=True)
    output_path = 'data/processed_market_news.csv'
    merged_data.to_csv(output_path)
    print(f"Data processing complete. Saved to {output_path}")
    print(f"Final data shape: {merged_data.shape}")
    print(merged_data.head())

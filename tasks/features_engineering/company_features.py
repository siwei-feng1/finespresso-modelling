import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import yfinance as yf
import pickle
from textblob import TextBlob

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs', 'features_engineering')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'company_features.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger



logger = setup_logger(__name__)

def validate_ticker(ticker: str, cache: Dict) -> bool:
    """Check if a ticker is valid using cached Yahoo Finance data."""
    if ticker in cache:
        return cache[ticker]
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        cache[ticker] = bool(info.get('symbol'))
        return cache[ticker]
    except Exception:
        cache[ticker] = False
        return False

def add_company_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add company-specific features to DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with required columns.

    Returns:
        pd.DataFrame: DataFrame with added company features.
    """
    required_cols = ['ticker', 'market_cap', 'published_date', 'sector']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns: {required_cols}")
        raise ValueError("Missing required columns")

    df = df.copy()
    
    # Add combined_sentiment if missing
    if 'combined_sentiment' not in df.columns:
        logger.info("Calculating combined_sentiment using TextBlob")
        df['combined_sentiment'] = df.apply(
            lambda x: TextBlob(str(x['title']) + ' ' + str(x['content'])).sentiment.polarity,
            axis=1
        )

    ticker_cache = {}
    features = ['market_cap_category', 'volatility', 'sector_relative_volatility', 'prev_news_sentiment']
    for feature in features:
        df[feature] = np.nan

    # Market cap category
    df['market_cap_category'] = pd.cut(
        df['market_cap'],
        bins=[0, 2e9, 10e9, float('inf')],
        labels=['Small', 'Mid', 'Large'],
        include_lowest=True
    )
    df['market_cap_category'] = df['market_cap_category'].cat.add_categories(['Unknown']).fillna('Unknown')
    logger.info("Added market cap category feature")

    # Volatility and sector-relative volatility
    sector_indices = {
        'Technology': '^IXIC',
        'Finance': '^IXBK',
        'Healthcare': '^IXHC',
        'default': '^GSPC'
    }
    for idx, row in df.iterrows():
        ticker = row['ticker']
        date = pd.to_datetime(row['published_date']).tz_localize(None)
        if not validate_ticker(ticker, ticker_cache):
            logger.warning(f"Skipping invalid ticker {ticker}")
            continue
        try:
            yf_ticker = yf.Ticker(ticker)
            history = yf_ticker.history(period='1y', end=date)
            if not history.empty:
                daily_returns = history['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                df.at[idx, 'volatility'] = volatility
                # Sector-relative volatility
                sector = row['sector'] if pd.notna(row['sector']) else 'default'
                sector_index = sector_indices.get(sector, sector_indices['default'])
                sector_history = yf.Ticker(sector_index).history(period='1y', end=date)
                if not sector_history.empty:
                    sector_returns = sector_history['Close'].pct_change().dropna()
                    sector_vol = sector_returns.std() * np.sqrt(252)
                    df.at[idx, 'sector_relative_volatility'] = volatility / sector_vol if sector_vol > 0 else np.nan
        except Exception as e:
            logger.warning(f"Failed to calculate volatility for {ticker}: {str(e)}")

    # Previous news sentiment
    for ticker in df['ticker'].unique():
        ticker_mask = df['ticker'] == ticker
        dates = pd.to_datetime(df[ticker_mask]['published_date']).dt.tz_localize(None)
        sentiments = df[ticker_mask]['combined_sentiment']
        for idx in df[ticker_mask].index:
            current_date = pd.to_datetime(df.at[idx, 'published_date']).tz_localize(None)
            past_mask = dates < current_date
            if past_mask.any():
                df.at[idx, 'prev_news_sentiment'] = sentiments[past_mask].mean()
            else:
                df.at[idx, 'prev_news_sentiment'] = 0

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats = df[features].describe(include='all').to_dict()
    stats_path = os.path.join(base_dir, 'reports', f'company_features_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    pd.DataFrame(stats).to_csv(stats_path, index=True)
    logger.info(f"Saved company feature statistics to {stats_path}")
    
    return df

def load_cache(cache_path: str) -> Dict:
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache: Dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = pd.read_csv(os.path.join(base_dir, 'data', 'feature_engineering', 'time_features_data.csv'))
    df_with_company_features = add_company_features(df)
    df_with_company_features.to_csv(os.path.join(base_dir, 'data', 'feature_engineering', 'company_features_data.csv'), index=False)
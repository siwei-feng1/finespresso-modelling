import pandas as pd
import yfinance as yf
import logging
import os
from datetime import datetime, time, timedelta
import numpy as np
from utils.date.date_adjuster import get_previous_trading_day, get_next_trading_day
import sys
import pytz
from pathlib import Path

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/info.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Market hours in ET
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

def get_trade_time(published_date, market, market_tz=None):
    """
    Determine the trade entry time based on published date and market period
    Returns time in market's timezone
    """
    # Use the market's timezone from the data, or UTC if not specified
    market_tz = pytz.timezone(market_tz) if market_tz else pytz.UTC
    market_date = published_date.astimezone(market_tz)
    
    if market == 'regular_market':
        # For regular market, enter at market open if news is before open
        if market_date.time() < MARKET_OPEN:
            return datetime.combine(market_date.date(), MARKET_OPEN, market_tz)
        # Otherwise enter at next 5-minute mark, but not after close
        entry_time = market_date + timedelta(minutes=(5 - market_date.minute % 5))
        market_close = datetime.combine(market_date.date(), MARKET_CLOSE, market_tz)
        return min(entry_time, market_close)
    elif market == 'pre_market':
        # Always enter exactly at market open (9:30) same day
        return datetime.combine(market_date.date(), MARKET_OPEN, market_tz)
    elif market == 'after_market':
        # Always enter exactly at market open (9:30) next day
        next_trading_day = get_next_trading_day(market_date.date())
        return datetime.combine(next_trading_day, MARKET_OPEN, market_tz)
    
    return None

def determine_market_period(published_date, market_tz=None):
    """
    Determine if the publication time falls in pre-market, regular market, or after-market
    Uses market's local timezone for:
    - Pre-market: 00:00-9:30
    - Regular: 9:30-16:00
    - After-market: 16:00-23:59
    """
    # Use the market's timezone from the data, or UTC if not specified
    market_tz = pytz.timezone(market_tz) if market_tz else pytz.UTC
    market_date = published_date.astimezone(market_tz)
    market_time = market_date.time()
    
    if market_time < MARKET_OPEN:
        return 'pre_market'
    elif MARKET_OPEN <= market_time < MARKET_CLOSE:
        return 'regular_market'
    else:  # MARKET_CLOSE <= market_time
        return 'after_market'

def get_intraday_prices(symbol, date, interval='1m'):
    """Get intraday price data for a specific date"""
    try:
        start_date = date.strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            logger.warning(f"No intraday data available for {symbol} on {date}")
        
        return data
    except Exception as e:
        logger.error(f"Error fetching intraday data for {symbol}: {e}")
        return pd.DataFrame()

def set_prices(row):
    """Get entry and exit prices for a trade based on the news publication time"""
    row = row.copy()
    
    symbol = row['yf_ticker']
    logger.info(f"Processing price data for {symbol}")

    if not symbol:
        logger.warning(f"No ticker symbol found for news_id {row.get('news_id')}")
        return row

    # Get timezone from the row data
    news_tz = pytz.timezone(row['timezone']) if row.get('timezone') else pytz.UTC
    
    # Convert published_date to datetime with proper timezone
    if isinstance(row['published_date'], pd.Timestamp):
        pub_date = row['published_date'].to_pydatetime()
    else:
        pub_date = datetime.strptime(row['published_date'], '%Y-%m-%d %H:%M:%S%z')
    
    # Ensure datetime is in the correct timezone
    if pub_date.tzinfo is None:
        pub_date = news_tz.localize(pub_date)
    else:
        pub_date = pub_date.astimezone(news_tz)
    
    # Determine market period using the news timezone
    market = determine_market_period(pub_date, row['timezone'])
    row['market'] = market
    
    pub_date_only = pub_date.date()
    next_trading_day = get_next_trading_day(pub_date_only)

    try:
        if market == 'pre_market':
            intraday_data = get_intraday_prices(symbol, pub_date_only)
            if not intraday_data.empty:
                # Exactly 9:30 for entry, 16:00 for exit
                market_open = datetime.combine(pub_date_only, MARKET_OPEN, news_tz)
                market_close = datetime.combine(pub_date_only, MARKET_CLOSE, news_tz)
                
                trading_data = intraday_data[
                    (intraday_data.index >= market_open) & 
                    (intraday_data.index <= market_close)
                ]
                
                if not trading_data.empty:
                    row['begin_price'] = float(trading_data['Open'].iloc[0])
                    row['end_price'] = float(trading_data['Close'].iloc[-1])
                    row['intraday_prices'] = trading_data
                    row['entry_time'] = market_open.replace(second=0, microsecond=0)
                    row['exit_time'] = market_close.replace(second=0, microsecond=0)
                else:
                    logger.warning(f"No trading hours data for {symbol} on {pub_date_only}")
                    return row
            else:
                logger.warning(f"No intraday data for {symbol} on {pub_date_only}")
                return row
                
        elif market == 'after_market':
            intraday_data = get_intraday_prices(symbol, next_trading_day)
            if not intraday_data.empty:
                # Exactly 9:30 for entry, 16:00 for exit
                market_open = datetime.combine(next_trading_day, MARKET_OPEN, news_tz)
                market_close = datetime.combine(next_trading_day, MARKET_CLOSE, news_tz)
                
                trading_data = intraday_data[
                    (intraday_data.index >= market_open) & 
                    (intraday_data.index <= market_close)
                ]
                
                if not trading_data.empty:
                    row['begin_price'] = float(trading_data['Open'].iloc[0])
                    row['end_price'] = float(trading_data['Close'].iloc[-1])
                    row['intraday_prices'] = trading_data
                    row['entry_time'] = market_open.replace(second=0, microsecond=0)
                    row['exit_time'] = market_close.replace(second=0, microsecond=0)
                else:
                    logger.warning(f"No trading hours data for {symbol} on {next_trading_day}")
                    return row
            else:
                logger.warning(f"No intraday data for {symbol} on {next_trading_day}")
                return row
                
        else:  # regular_market
            intraday_data = get_intraday_prices(symbol, pub_date_only)
            if not intraday_data.empty:
                market_close = datetime.combine(pub_date_only, MARKET_CLOSE, news_tz)
                entry_time = get_trade_time(pub_date, market, row['timezone'])
                
                trading_data = intraday_data[
                    (intraday_data.index >= entry_time) & 
                    (intraday_data.index <= market_close)
                ]
                
                if not trading_data.empty:
                    row['begin_price'] = float(trading_data['Open'].iloc[0])
                    row['end_price'] = float(trading_data['Close'].iloc[-1])
                    row['intraday_prices'] = trading_data
                    row['entry_time'] = entry_time.replace(second=0, microsecond=0)
                    row['exit_time'] = market_close.replace(second=0, microsecond=0)
                else:
                    logger.warning(f"No valid intraday data after publication for {symbol}")
                    return row
            else:
                logger.warning(f"No intraday data for {symbol} on {pub_date_only}")
                return row

        # Calculate basic price metrics
        if row.get('begin_price') is not None and row.get('end_price') is not None:
            row['price_change'] = row['end_price'] - row['begin_price']
            row['price_change_percentage'] = (row['price_change'] / row['begin_price']) * 100
            row['actual_side'] = 'UP' if row['price_change_percentage'] >= 0 else 'DOWN'

    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        logger.exception("Detailed traceback:")
        
    return row

def create_price_moves(news_df):
    """Process each news item to get price data"""
    logger.info(f"Starting to create price moves for {len(news_df)} news items")
    
    if news_df.empty:
        logger.warning("Empty news DataFrame provided")
        return pd.DataFrame()
        
    # Check if required columns exist in input DataFrame
    required_input_columns = ['yf_ticker', 'published_date', 'timezone']
    missing_input_columns = [col for col in required_input_columns if col not in news_df.columns]
    if missing_input_columns:
        logger.error(f"Missing required input columns: {missing_input_columns}")
        return pd.DataFrame()
    
    news_df = news_df.reset_index(drop=True)
    
    try:
        # Process each row
        processed_rows = []
        for idx, row in news_df.iterrows():
            try:
                processed_row = set_prices(row)
                if processed_row.get('begin_price') is not None and processed_row.get('end_price') is not None:
                    processed_rows.append(processed_row)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        if not processed_rows:
            logger.warning("No rows were successfully processed")
            return pd.DataFrame()
            
        processed_df = pd.DataFrame(processed_rows)
        
        # Check if required output columns exist
        required_output_columns = ['begin_price', 'end_price', 'entry_time', 'exit_time']
        missing_output_columns = [col for col in required_output_columns if col not in processed_df.columns]
        if missing_output_columns:
            logger.error(f"Missing required output columns after processing: {missing_output_columns}")
            return pd.DataFrame()
        
        # Remove rows with missing price data
        original_len = len(processed_df)
        processed_df.dropna(subset=required_output_columns, inplace=True)
        logger.info(f"Removed {original_len - len(processed_df)} rows with missing price data")
        
        if processed_df.empty:
            logger.warning("No valid rows remain after processing")
            return pd.DataFrame()
            
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in create_price_moves: {str(e)}")
        logger.exception("Detailed traceback:")
        return pd.DataFrame()

def load_news_data(data_dir="data"):
    """Load news data from CSV files in the data directory"""
    news_file = Path(data_dir) / "all_news.csv"
    logger = logging.getLogger(__name__)
    if not news_file.exists():
        logger.error(f"News file not found: {news_file}")
        return None
    try:
        logger.info(f"Loading news data from {news_file}")
        news_df = pd.read_csv(news_file)
        # Convert published_date to datetime
        if 'published_date' in news_df.columns:
            news_df['published_date'] = pd.to_datetime(news_df['published_date'])
        logger.info(f"Loaded {len(news_df)} news records")
        return news_df
    except Exception as e:
        logger.error(f"Error loading news data: {e}")
        return None

def load_price_moves_csv(data_dir="data"):
    """Load price moves from CSV file (data/all_price_moves.csv)"""
    price_moves_file = Path(data_dir) / "all_price_moves.csv"
    logger = logging.getLogger(__name__)
    if not price_moves_file.exists():
        logger.error(f"Price moves file not found: {price_moves_file}")
        return None
    try:
        logger.info(f"Loading price moves from {price_moves_file}")
        price_moves_df = pd.read_csv(price_moves_file)
        logger.info(f"Loaded {len(price_moves_df)} price move records")
        return price_moves_df
    except Exception as e:
        logger.error(f"Error loading price moves: {e}")
        return None

def load_price_moves_db(*args, **kwargs):
    """Load price moves from the database table price_moves"""
    from utils.db.price_move_db_util import get_price_moves
    try:
        price_moves_df = get_price_moves()
        return price_moves_df
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading price moves from DB: {e}")
        return None

def load_news_data_csv(data_dir="data"):
    """Load news data from CSV files in the data directory (robust date parsing)"""
    news_file = Path(data_dir) / "all_news.csv"
    logger = logging.getLogger(__name__)
    if not news_file.exists():
        logger.error(f"News file not found: {news_file}")
        return None
    try:
        logger.info(f"Loading news data from {news_file}")
        news_df = pd.read_csv(news_file)
        # Robust date parsing for ISO8601 and mixed formats
        if 'published_date' in news_df.columns:
            news_df['published_date'] = pd.to_datetime(news_df['published_date'], errors='coerce', utc=True)
        logger.info(f"Loaded {len(news_df)} news records")
        return news_df
    except Exception as e:
        logger.error(f"Error loading news data: {e}")
        return None 
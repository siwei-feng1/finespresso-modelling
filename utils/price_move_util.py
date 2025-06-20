import pandas as pd
import yfinance as yf
import logging
import os
from datetime import datetime, time
import numpy as np
from utils.db.price_move_db_util import store_price_move, PriceMove
from utils.date.date_adjuster import get_previous_trading_day, get_next_trading_day
import sys

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

index_symbol = 'SPY'
EXCHANGE = 'NASDAQ'

def get_price_data(ticker, published_date):
    logger.info(f"Getting price data for {ticker} on {published_date}")
    row = pd.Series({'ticker': ticker, 'published_date': published_date, 'market': 'market_open'})  # Assume market_open, adjust if needed
    processed_row = set_prices(row)
    
    if processed_row['begin_price'] is None or processed_row['end_price'] is None:
        logger.warning(f"Unable to get price data for {ticker} on {published_date}")
        return None

def set_prices(row):
    # Convert the input row to a copy to avoid SettingWithCopyWarning
    row = row.copy()
    
    try:
        symbol = row['yf_ticker']
        if pd.isna(symbol) or not isinstance(symbol, str):
            logger.error(f"Invalid symbol: {symbol}")
            return None
            
        logger.info(f"Processing price data for {symbol}")

        # Validate and parse published_date
        if pd.isna(row['published_date']):
            logger.error(f"Invalid published_date for {symbol}")
            return None

        if isinstance(row['published_date'], pd.Timestamp):
            today_date = row['published_date'].to_pydatetime()
        else:
            try:
                today_date = datetime.strptime(str(row['published_date']), '%Y-%m-%d %H:%M:%S%z')
            except ValueError:
                logger.error(f"Invalid date format for {symbol}: {row['published_date']}")
                return None

        # Get market timing and trading days
        pub_time = today_date.time()
        today_date_only = today_date.date()
        
        # Determine market based on publication time
        if time(9, 30) <= pub_time < time(16, 0):
            market = 'regular_market'
        elif time(16, 0) <= pub_time:
            market = 'after_market'
        elif time(0, 0) <= pub_time < time(9, 30):
            market = 'pre_market'
        else:
            market = 'regular_market'
            logger.warning(f"Unknown market time for {symbol}, defaulting to regular_market")

        # Get trading days
        previous_trading_day = get_previous_trading_day(today_date_only)
        next_trading_day = get_next_trading_day(today_date_only)
        
        yf_previous_date = previous_trading_day.strftime('%Y-%m-%d')
        yf_today_date = today_date_only.strftime('%Y-%m-%d')
        yf_next_date = next_trading_day.strftime('%Y-%m-%d')

        # Log the date range we're querying
        logger.info(f"Querying {symbol} data from {yf_previous_date} to {yf_next_date}")

        try:
            # Download price data with explicit interval
            data = yf.download(symbol, start=yf_previous_date, end=yf_next_date, interval='1d')
            index_data = yf.download(index_symbol, start=yf_previous_date, end=yf_next_date, interval='1d')
            
            # Log the retrieved data shape
            logger.info(f"Retrieved data shape - {symbol}: {data.shape}, {index_symbol}: {index_data.shape}")
            
            if data.empty or index_data.empty:
                logger.warning(f"No data available for {symbol} or index")
                return None

            # Log available dates in the data
            logger.info(f"Available dates for {symbol}: {data.index.strftime('%Y-%m-%d').tolist()}")

            # Extract single price values based on market timing
            try:
                # Add debug logging for price extraction
                logger.debug(f"Market timing: {market}")
                logger.debug(f"Data for {yf_today_date}:\n{data.loc[yf_today_date] if yf_today_date in data.index else 'No data'}")

                if market == 'pre_market':
                    if yf_previous_date not in data.index or yf_today_date not in data.index:
                        logger.error(f"Missing required dates for pre_market: prev={yf_previous_date}, today={yf_today_date}")
                        return None
                    begin_price = float(data.loc[yf_previous_date, 'Close'].iloc[0])
                    end_price = float(data.loc[yf_today_date, 'Open'].iloc[0])
                    index_begin_price = float(index_data.loc[yf_previous_date, 'Close'].iloc[0])
                    index_end_price = float(index_data.loc[yf_today_date, 'Open'].iloc[0])
                elif market == 'regular_market':
                    if yf_today_date not in data.index:
                        logger.error(f"Missing required date for regular_market: {yf_today_date}")
                        return None
                    begin_price = float(data.loc[yf_today_date, 'Open'].iloc[0])
                    end_price = float(data.loc[yf_today_date, 'Close'].iloc[0])
                    index_begin_price = float(index_data.loc[yf_today_date, 'Open'].iloc[0])
                    index_end_price = float(index_data.loc[yf_today_date, 'Close'].iloc[0])
                else:  # after_market
                    if yf_today_date not in data.index or yf_next_date not in data.index:
                        logger.error(f"Missing required dates for after_market: today={yf_today_date}, next={yf_next_date}")
                        return None
                    begin_price = float(data.loc[yf_today_date, 'Close'].iloc[0])
                    end_price = float(data.loc[yf_next_date, 'Open'].iloc[0])
                    index_begin_price = float(index_data.loc[yf_today_date, 'Close'].iloc[0])
                    index_end_price = float(index_data.loc[yf_next_date, 'Open'].iloc[0])

                # Log the extracted prices
                logger.info(f"Extracted prices for {symbol}: begin={begin_price}, end={end_price}")

                # Validate prices
                if any(pd.isna([begin_price, end_price, index_begin_price, index_end_price])):
                    logger.warning(f"NaN values in price data for {symbol}")
                    return None

                if begin_price <= 0 or index_begin_price <= 0:
                    logger.warning(f"Invalid begin prices for {symbol}: stock={begin_price}, index={index_begin_price}")
                    return None

                # Calculate price changes
                price_change = end_price - begin_price
                index_price_change = index_end_price - index_begin_price
                
                # Calculate percentages
                price_change_percentage = (price_change / begin_price) * 100
                index_price_change_percentage = (index_price_change / index_begin_price) * 100
                
                # Update row with calculated values
                row.update({
                    'begin_price': begin_price,
                    'end_price': end_price,
                    'index_begin_price': index_begin_price,
                    'index_end_price': index_end_price,
                    'price_change': price_change,
                    'price_change_percentage': price_change_percentage,
                    'index_price_change': index_price_change,
                    'index_price_change_percentage': index_price_change_percentage,
                    'daily_alpha': price_change_percentage - index_price_change_percentage,
                    'actual_side': 'UP' if price_change_percentage >= 0 else 'DOWN',
                    'Volume': float(data.loc[yf_today_date, 'Volume'].iloc[0]),
                    'market': market
                })

            except (IndexError, KeyError) as e:
                logger.error(f"Error accessing price data for {symbol}: {e}")
                return None
                
        except Exception as e:
            if "YFTzMissingError" in str(e):
                logger.warning(f"Stock {symbol} may be delisted or has no timezone data")
            else:
                logger.error(f"Error downloading data for {symbol}: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error processing {symbol}: {e}")
        return None

    logger.info(f"Successfully processed price move for {symbol}")
    return row

def create_price_moves(news_df, price_source='yfinance'):
    logger.info(f"Starting to create price moves for {len(news_df)} news items")
    news_df = news_df.reset_index(drop=True)
    processed_rows = []

    for index, row in news_df.iterrows():
        try:
            logger.info(f"Processing row {index} for ticker {row['yf_ticker']}")
            processed_row = set_prices(row)
            if processed_row is not None:  # Only append valid processed rows
                processed_rows.append(processed_row)
        except Exception as e:
            logger.error(f"Error processing row {index} for ticker {row['yf_ticker']}: {e}")
            logger.exception("Detailed traceback:")
            continue

    if not processed_rows:
        logger.warning("No valid rows were processed")
        return pd.DataFrame()

    processed_df = pd.DataFrame(processed_rows)
    logger.info(f"Processed {len(processed_df)} rows successfully")

    required_price_columns = ['begin_price', 'end_price', 'index_begin_price', 'index_end_price']
    missing_columns = [col for col in required_price_columns if col not in processed_df.columns]
    if missing_columns:
        logger.warning(f"Missing columns in the DataFrame: {missing_columns}")
        return processed_df

    original_len = len(processed_df)
    processed_df.dropna(subset=required_price_columns, inplace=True)
    logger.info(f"Removed {original_len - len(processed_df)} rows with NaN values")

    try:
        logger.info("Calculating alpha and setting actual side")
        processed_df['daily_alpha'] = processed_df['price_change_percentage'] - processed_df['index_price_change_percentage']
        processed_df['actual_side'] = np.where(processed_df['price_change_percentage'] >= 0, 'UP', 'DOWN')
        logger.info("Successfully calculated alpha and set actual side")
    except Exception as e:
        logger.error(f"Error in calculations: {e}")

    logger.info(f"Finished creating price moves. Final DataFrame has {len(processed_df)} rows")
    
    # Store price moves in the database
    for _, row in processed_df.iterrows():
        try:
            price_move = create_price_move(
                news_id=row['news_id'],
                ticker=row['yf_ticker'],  # Changed from 'ticker' to 'yf_ticker'
                published_date=row['published_date'],
                begin_price=row['begin_price'],
                end_price=row['end_price'],
                index_begin_price=row['index_begin_price'],
                index_end_price=row['index_end_price'],
                volume=row.get('Volume'),
                market=row['market'],
                price_change=row['price_change'],
                price_change_percentage=row['price_change_percentage'],
                index_price_change=row['index_price_change'],
                index_price_change_percentage=row['index_price_change_percentage'],
                actual_side=row['actual_side'],
                price_source=price_source
            )
            store_price_move(price_move)
        except Exception as e:
            logger.error(f"Error storing price move for news_id {row['news_id']}: {e}")

    return processed_df

def create_price_move(news_id, ticker, published_date, begin_price, end_price, index_begin_price, index_end_price, volume, market, price_change, price_change_percentage, index_price_change, index_price_change_percentage, actual_side, predicted_side=None, price_source='yfinance'):
    daily_alpha = price_change_percentage - index_price_change_percentage
    return PriceMove(
        news_id=news_id,
        ticker=ticker,
        published_date=published_date,
        begin_price=begin_price,
        end_price=end_price,
        index_begin_price=index_begin_price,
        index_end_price=index_end_price,
        volume=volume,
        market=market,
        price_change=price_change,
        price_change_percentage=price_change_percentage,
        index_price_change=index_price_change,
        index_price_change_percentage=index_price_change_percentage,
        daily_alpha=daily_alpha,
        actual_side=actual_side,
        predicted_side=predicted_side,
        downloaded_at=datetime.utcnow(),
        price_source=price_source
    )

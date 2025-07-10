#!/usr/bin/env python3
"""
Backfill script for processing existing news from database and calculating price moves.

This script:
1. Gets news from database for a specific month and publisher using news_db_util
2. For news with yf_ticker: calculates price moves using yfinance and stores to DB with runid
3. Uses SPY as the market index
4. Provides comprehensive logging and statistics
"""

import pandas as pd
import logging
import os
import sys
import argparse
from datetime import datetime, time, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from dotenv import load_dotenv
import time as time_module

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.date.date_adjuster import get_previous_trading_day, get_next_trading_day
from utils.db.news_db_util import get_news_df_date_range
from utils.db.price_move_db_util import store_price_move, PriceMove

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backfill_db.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class YFinanceBackfillProcessor:
    def __init__(self):
        """Initialize the YFinance backfill processor."""
        self.index_symbol = 'SPY'  # S&P 500 ETF as market index
        self.stats = {
            'total_news': 0,
            'with_ticker': 0,
            'price_moves_calculated': 0,
            'price_moves_stored_db': 0,
            'errors': 0
        }
        
    def generate_run_id(self) -> int:
        """Generate a unique run ID based on current timestamp."""
        return int(datetime.now().timestamp())
    
    def get_market_timing(self, published_date: datetime) -> str:
        """
        Determine market timing based on publication time.
        
        Args:
            published_date: Publication date
            
        Returns:
            Market timing: 'pre_market', 'regular_market', or 'after_market'
        """
        # Handle timezone-aware datetime
        if published_date.tzinfo is None:
            # Assume UTC if no timezone info
            published_date = published_date.replace(tzinfo=None)
        
        pub_time = published_date.time()
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if pub_time < market_open:
            return 'pre_market'
        elif market_open <= pub_time < market_close:
            return 'regular_market'
        else:  # pub_time >= market_close
            return 'after_market'
    
    def get_price_data(self, ticker: str, published_date: datetime) -> Optional[Dict]:
        """Get price data for ticker around the published date using yfinance."""
        try:
            # Handle timezone-aware datetime
            if published_date.tzinfo is None:
                # Assume UTC if no timezone info
                published_date = published_date.replace(tzinfo=None)
            
            pub_time = published_date.time()
            pub_date = published_date.date()
            
            # Determine market timing based on publication time
            market_timing = self.get_market_timing(published_date)
            
            # Get trading days
            previous_trading_day = get_previous_trading_day(pub_date)
            next_trading_day = get_next_trading_day(pub_date)
            
            # Format dates for yfinance
            yf_prev_date = previous_trading_day.strftime('%Y-%m-%d')
            yf_today_date = pub_date.strftime('%Y-%m-%d')
            yf_next_date = next_trading_day.strftime('%Y-%m-%d')
            
            logger.debug(f"Getting price data for {ticker} on {yf_today_date}, market: {market_timing}")
            
            # Download price data
            data = yf.download(ticker, start=yf_prev_date, end=yf_next_date, interval='1d', auto_adjust=True)
            index_data = yf.download(self.index_symbol, start=yf_prev_date, end=yf_next_date, interval='1d', auto_adjust=True)
            
            if data.empty or index_data.empty:
                logger.warning(f"No price data available for {ticker}")
                return None
            
            # Extract prices based on market timing
            if market_timing == 'pre_market':
                if yf_prev_date not in data.index or yf_today_date not in data.index:
                    return None
                begin_price = float(data.loc[yf_prev_date, 'Close'].iloc[0])
                end_price = float(data.loc[yf_today_date, 'Open'].iloc[0])
                index_begin_price = float(index_data.loc[yf_prev_date, 'Close'].iloc[0])
                index_end_price = float(index_data.loc[yf_today_date, 'Open'].iloc[0])
            elif market_timing == 'regular_market':
                if yf_today_date not in data.index:
                    return None
                begin_price = float(data.loc[yf_today_date, 'Open'].iloc[0])
                end_price = float(data.loc[yf_today_date, 'Close'].iloc[0])
                index_begin_price = float(index_data.loc[yf_today_date, 'Open'].iloc[0])
                index_end_price = float(index_data.loc[yf_today_date, 'Close'].iloc[0])
            else:  # after_market
                if yf_today_date not in data.index or yf_next_date not in data.index:
                    return None
                begin_price = float(data.loc[yf_today_date, 'Close'].iloc[0])
                end_price = float(data.loc[yf_next_date, 'Open'].iloc[0])
                index_begin_price = float(index_data.loc[yf_today_date, 'Close'].iloc[0])
                index_end_price = float(index_data.loc[yf_next_date, 'Open'].iloc[0])
            
            # Calculate price changes
            price_change = end_price - begin_price
            index_price_change = index_end_price - index_begin_price
            
            price_change_percentage = (price_change / begin_price) * 100
            index_price_change_percentage = (index_price_change / index_begin_price) * 100
            
            volume = float(data.loc[yf_today_date, 'Volume'].iloc[0]) if yf_today_date in data.index and not pd.isna(data.loc[yf_today_date, 'Volume'].iloc[0]) else 0
            
            return {
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
                'volume': volume,
                'market': market_timing
            }
            
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {e}")
            return None
    
    def calculate_price_move(self, row: pd.Series, run_id: int) -> Optional[PriceMove]:
        """
        Calculate price move for a news item based on market timing using yfinance.
        
        Args:
            row: News item row from DataFrame
            run_id: Run ID for this calculation batch
            
        Returns:
            PriceMove object or None if calculation fails
        """
        try:
            ticker = row['yf_ticker']
            published_date = row['published_date']
            news_id = row['news_id']
            
            if pd.isna(ticker) or not isinstance(ticker, str) or ticker.strip() == '':
                logger.warning(f"Invalid ticker for news_id {news_id}: {ticker}")
                return None
            
            # Convert published_date to datetime if it's a string
            if isinstance(published_date, str):
                published_date = pd.to_datetime(published_date)
            
            # Determine market timing
            market_timing = self.get_market_timing(published_date)
            logger.info(f"Processing {ticker} (news_id: {news_id}) - Market timing: {market_timing}")
            
            # Get price data
            price_data = self.get_price_data(ticker, published_date)
            if not price_data:
                logger.warning(f"No price data available for {ticker}")
                return None
            
            # Create PriceMove object
            price_move = PriceMove(
                news_id=int(news_id),
                ticker=ticker,
                published_date=published_date,
                begin_price=float(price_data['begin_price']),
                end_price=float(price_data['end_price']),
                index_begin_price=float(price_data['index_begin_price']),
                index_end_price=float(price_data['index_end_price']),
                volume=int(price_data['volume']) if price_data['volume'] else None,
                market=price_data['market'],
                price_change=float(price_data['price_change']),
                price_change_percentage=float(price_data['price_change_percentage']),
                index_price_change=float(price_data['index_price_change']),
                index_price_change_percentage=float(price_data['index_price_change_percentage']),
                daily_alpha=float(price_data['daily_alpha']),
                actual_side=price_data['actual_side'],
                predicted_side=row.get('predicted_side'),
                predicted_move=float(row.get('predicted_move')) if row.get('predicted_move') is not None else None,
                price_source='yfinance',
                runid=int(run_id)
            )
            
            # Set downloaded_at timestamp
            price_move.downloaded_at = datetime.now(timezone.utc)
            
            logger.info(f"Successfully calculated price move for {ticker}: {price_data['price_change_percentage']:.2f}% (alpha: {price_data['daily_alpha']:.2f}%)")
            return price_move
            
        except Exception as e:
            logger.error(f"Error calculating price move for news_id {row.get('news_id')}: {e}")
            return None
    
    def calculate_price_moves_for_date_range(self, start_month: str, end_month: str, publisher: str = 'baltics') -> pd.DataFrame:
        """
        Calculate price moves for all news items in a date range using yfinance.
        
        Args:
            start_month: Start month in YYYY-MM format (e.g., "2025-06")
            end_month: End month in YYYY-MM format (e.g., "2025-08")
            publisher: Publisher to filter by (default: globenewswire_biotech)
            
        Returns:
            DataFrame with calculated price moves
        """
        # Generate run ID
        run_id = self.generate_run_id()
        logger.info(f"Starting YFinance price move calculation from {start_month} to {end_month} with run_id: {run_id}")
        
        # Parse start and end months
        try:
            start_year, start_month_num = map(int, start_month.split('-'))
            end_year, end_month_num = map(int, end_month.split('-'))
        except ValueError as e:
            logger.error(f"Invalid date format. Use YYYY-MM format: {e}")
            return pd.DataFrame()
        
        # Calculate date range
        start_date = datetime(start_year, start_month_num, 1)
        if end_month_num == 12:
            end_date = datetime(end_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(end_year, end_month_num + 1, 1) - timedelta(days=1)
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Get news data from database
        logger.info(f"Fetching news data for publisher: {publisher}")
        news_df = get_news_df_date_range(
            publishers=[publisher],
            start_date=start_date,
            end_date=end_date
        )
        
        if news_df.empty:
            logger.warning(f"No news found for {publisher} in {start_month}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(news_df)} news items for processing")
        
        # Filter for items with valid tickers
        news_df = news_df.dropna(subset=['yf_ticker'])
        # Also filter out empty string tickers
        news_df = news_df[news_df['yf_ticker'].str.strip() != '']
        logger.info(f"After filtering for valid tickers: {len(news_df)} items")
        
        self.stats['total_news'] = len(news_df)
        self.stats['with_ticker'] = len(news_df)
        
        # Calculate price moves with batch processing
        successful_calculations = 0
        failed_calculations = 0
        price_moves = []
        batch_size = 50
        current_batch = []
        
        for index, row in news_df.iterrows():
            try:
                price_move = self.calculate_price_move(row, run_id)
                if price_move:
                    current_batch.append(price_move)
                    successful_calculations += 1
                    logger.debug(f"Calculated price move for news_id {row['news_id']}")
                else:
                    failed_calculations += 1
                    
                # Process batch when it reaches batch_size or at the end
                if len(current_batch) >= batch_size or index == len(news_df) - 1:
                    # Store batch to database
                    batch_stored = 0
                    for pm in current_batch:
                        if store_price_move(pm):
                            batch_stored += 1
                            self.stats['price_moves_stored_db'] += 1
                        else:
                            logger.warning(f"Failed to store price move for news_id {pm.news_id}")
                    
                    logger.info(f"Batch processed: {len(current_batch)} calculated, {batch_stored} stored to DB")
                    price_moves.extend(current_batch)
                    current_batch = []
                    
                # Add small delay to avoid rate limiting
                time_module.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                failed_calculations += 1
                self.stats['errors'] += 1
                continue
        
        self.stats['price_moves_calculated'] = successful_calculations
        
        logger.info(f"YFinance price move calculation completed:")
        logger.info(f"  - Successful calculations: {successful_calculations}")
        logger.info(f"  - Failed calculations: {failed_calculations}")
        logger.info(f"  - Run ID: {run_id}")
        
        # Convert to DataFrame for return
        if price_moves:
            df_data = []
            for pm in price_moves:
                df_data.append({
                    'news_id': pm.news_id,
                    'ticker': pm.ticker,
                    'published_date': pm.published_date,
                    'begin_price': pm.begin_price,
                    'end_price': pm.end_price,
                    'index_begin_price': pm.index_begin_price,
                    'index_end_price': pm.index_end_price,
                    'volume': pm.volume,
                    'market': pm.market,
                    'price_change': pm.price_change,
                    'price_change_percentage': pm.price_change_percentage,
                    'index_price_change': pm.index_price_change,
                    'index_price_change_percentage': pm.index_price_change_percentage,
                    'daily_alpha': pm.daily_alpha,
                    'actual_side': pm.actual_side,
                    'runid': pm.runid
                })
            
            result_df = pd.DataFrame(df_data)
            return result_df
        else:
            return pd.DataFrame()
    
    def print_statistics(self):
        """Print comprehensive statistics about the backfill process."""
        logger.info("=" * 60)
        logger.info("BACKFILL PROCESS STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total news items processed: {self.stats['total_news']}")
        logger.info(f"News with existing tickers: {self.stats['with_ticker']}")
        logger.info(f"Price moves calculated: {self.stats['price_moves_calculated']}")
        logger.info(f"Price moves stored to database: {self.stats['price_moves_stored_db']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info("=" * 60)

def main():
    """Main function to run the backfill process."""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate price moves using YFinance')
    parser.add_argument('--start-month', type=str, default='2025-06', 
                       help='Start month in YYYY-MM format (default: 2025-06)')
    parser.add_argument('--end-month', type=str, default='2025-06', 
                       help='End month in YYYY-MM format (default: 2025-06)')
    parser.add_argument('--publisher', type=str, default='baltics',
                       help='Publisher to filter news (default: baltics)')
    
    args = parser.parse_args()
    
    try:
        processor = YFinanceBackfillProcessor()
        
        logger.info(f"Starting YFinance price move calculation from {args.start_month} to {args.end_month}")
        
        # Calculate price moves
        result_df = processor.calculate_price_moves_for_date_range(
            start_month=args.start_month,
            end_month=args.end_month,
            publisher=args.publisher
        )
        
        if not result_df.empty:
            print(f"\nYFinance Price Move Calculation Summary:")
            print(f"Total price moves calculated: {len(result_df)}")
            
            # Handle timezone-aware datetime comparison
            try:
                min_date = result_df['published_date'].min()
                max_date = result_df['published_date'].max()
                print(f"Date range: {min_date} to {max_date}")
            except Exception as e:
                print(f"Date range: Available (timezone comparison issue: {e})")
            
            print(f"Unique tickers: {result_df['ticker'].nunique()}")
            print(f"Market timing distribution:")
            print(result_df['market'].value_counts())
            print(f"\nSample results:")
            print(result_df[['ticker', 'published_date', 'market', 'price_change_percentage', 'daily_alpha', 'actual_side']].head())
            
            # Print statistics
            processor.print_statistics()
        else:
            print("No price moves were calculated.")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
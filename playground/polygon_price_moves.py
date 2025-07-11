import os
import pandas as pd
import requests
import logging
from datetime import datetime, time, timedelta, timezone
import numpy as np
from typing import List, Dict, Optional
import pytz
from dotenv import load_dotenv

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db.news_db_util import get_news_df_date_range
from utils.db.price_move_db_util import store_price_move, PriceMove
from utils.date.date_adjuster import get_previous_trading_day, get_next_trading_day

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/polygon_price_moves.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
INDEX_SYMBOL = 'SPY'
US_EASTERN_TZ = pytz.timezone('US/Eastern')

class PolygonPriceMoveCalculator:
    def __init__(self):
        """Initialize the Polygon API price move calculator."""
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.index_symbol = INDEX_SYMBOL
        
    def generate_run_id(self) -> int:
        """Generate a unique run ID based on current timestamp."""
        return int(datetime.now().timestamp())
    
    def get_market_timing(self, published_date: datetime) -> str:
        """
        Determine market timing based on publication time.
        
        Args:
            published_date: Publication date in US Eastern time
            
        Returns:
            Market timing: 'pre_market', 'regular_market', or 'after_market'
        """
        # Convert to US Eastern time if not already
        if published_date.tzinfo is None:
            published_date = US_EASTERN_TZ.localize(published_date)
        elif published_date.tzinfo != US_EASTERN_TZ:
            published_date = published_date.astimezone(US_EASTERN_TZ)
        
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
    
    def get_daily_prices(self, ticker: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Get daily prices for a ticker from Polygon API.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with price data or None if error
        """
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                return data
            else:
                logger.warning(f"No data returned for {ticker}: {data.get('status')}, full response: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {ticker}: {e}")
            return None
    
    def process_price_data(self, raw_data: Dict, ticker: str) -> pd.DataFrame:
        """
        Process raw price data from Polygon API into a DataFrame.
        
        Args:
            raw_data: Raw data from Polygon API
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with processed price data
        """
        results = raw_data.get('results', [])
        
        if not results:
            return pd.DataFrame()
        
        # Extract data from results
        data = []
        for result in results:
            data.append({
                'ticker': ticker,
                'date': datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'close': result['c'],
                'volume': result['v'],
                'vwap': result.get('vw', None),
                'transactions': result.get('n', None)
            })
        
        return pd.DataFrame(data)
    
    def get_price_data_for_date(self, ticker: str, date: datetime.date) -> Optional[Dict]:
        """
        Get price data for a specific date using Polygon API.
        
        Args:
            ticker: Stock ticker symbol
            date: Date to get price data for
            
        Returns:
            Dictionary with price data or None if error
        """
        date_str = date.strftime('%Y-%m-%d')
        
        # Get stock data
        stock_data = self.get_daily_prices(ticker, date_str, date_str)
        if not stock_data:
            return None
        
        # Get index data
        index_data = self.get_daily_prices(self.index_symbol, date_str, date_str)
        if not index_data:
            return None
        
        # Process the data
        stock_df = self.process_price_data(stock_data, ticker)
        index_df = self.process_price_data(index_data, self.index_symbol)
        
        if stock_df.empty or index_df.empty:
            return None
        
        # Get the first (and only) row
        stock_row = stock_df.iloc[0]
        index_row = index_df.iloc[0]
        
        return {
            'open': stock_row['open'],
            'close': stock_row['close'],
            'volume': stock_row['volume'],
            'index_open': index_row['open'],
            'index_close': index_row['close']
        }
    
    def calculate_price_move(self, row: pd.Series, run_id: int) -> Optional[PriceMove]:
        """
        Calculate price move for a news item based on market timing using Polygon API.
        
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
            
            if pd.isna(ticker) or not isinstance(ticker, str):
                logger.warning(f"Invalid ticker for news_id {news_id}: {ticker}")
                return None
            
            # Convert published_date to datetime if it's a string
            if isinstance(published_date, str):
                published_date = pd.to_datetime(published_date)
            
            # Determine market timing
            market_timing = self.get_market_timing(published_date)
            logger.info(f"Processing {ticker} (news_id: {news_id}) - Market timing: {market_timing}")
            
            # Get the relevant trading days based on market timing
            pub_date = published_date.date()
            
            if market_timing == 'pre_market':
                # price_move = price(t, open) - price(t-1, close)
                prev_trading_day = get_previous_trading_day(pub_date)
                
                # Get previous day's close and current day's open
                prev_data = self.get_price_data_for_date(ticker, prev_trading_day)
                curr_data = self.get_price_data_for_date(ticker, pub_date)
                
                if not prev_data or not curr_data:
                    logger.warning(f"Missing price data for {ticker} - prev: {prev_trading_day}, curr: {pub_date}")
                    return None
                
                begin_price = prev_data['close']  # Previous day's close
                end_price = curr_data['open']     # Current day's open
                index_begin_price = prev_data['index_close']
                index_end_price = curr_data['index_open']
                volume = curr_data['volume']
                
            elif market_timing == 'regular_market':
                # price_move = price(t, close) - price(t, open)
                curr_data = self.get_price_data_for_date(ticker, pub_date)
                
                if not curr_data:
                    logger.warning(f"Missing price data for {ticker} on {pub_date}")
                    return None
                
                begin_price = curr_data['open']   # Current day's open
                end_price = curr_data['close']    # Current day's close
                index_begin_price = curr_data['index_open']
                index_end_price = curr_data['index_close']
                volume = curr_data['volume']
                
            else:  # after_market
                # price_move = price(t+1, open) - price(t, close)
                next_trading_day = get_next_trading_day(pub_date)
                
                # Get current day's close and next day's open
                curr_data = self.get_price_data_for_date(ticker, pub_date)
                next_data = self.get_price_data_for_date(ticker, next_trading_day)
                
                if not curr_data or not next_data:
                    logger.warning(f"Missing price data for {ticker} - curr: {pub_date}, next: {next_trading_day}")
                    return None
                
                begin_price = curr_data['close']  # Current day's close
                end_price = next_data['open']     # Next day's open
                index_begin_price = curr_data['index_close']
                index_end_price = next_data['index_open']
                volume = next_data['volume']
            
            # Validate prices
            if any(pd.isna([begin_price, end_price, index_begin_price, index_end_price])):
                logger.warning(f"NaN values in price data for {ticker}")
                return None
            
            if begin_price <= 0 or index_begin_price <= 0:
                logger.warning(f"Invalid begin prices for {ticker}: stock={begin_price}, index={index_begin_price}")
                return None
            
            # Calculate price changes
            price_change = end_price - begin_price
            index_price_change = index_end_price - index_begin_price
            
            # Calculate percentages
            price_change_percentage = (price_change / begin_price) * 100
            index_price_change_percentage = (index_price_change / index_begin_price) * 100
            
            # Calculate daily alpha
            daily_alpha = price_change_percentage - index_price_change_percentage
            
            # Determine actual side
            actual_side = 'UP' if price_change_percentage >= 0 else 'DOWN'
            
            # Create PriceMove object
            price_move = PriceMove(
                news_id=int(news_id),  # Convert to regular int
                ticker=ticker,
                published_date=published_date,
                begin_price=float(begin_price),
                end_price=float(end_price),
                index_begin_price=float(index_begin_price),
                index_end_price=float(index_end_price),
                volume=int(volume) if volume else None,  # Convert to regular int
                market=market_timing,
                price_change=float(price_change),
                price_change_percentage=float(price_change_percentage),
                index_price_change=float(index_price_change),
                index_price_change_percentage=float(index_price_change_percentage),
                daily_alpha=float(daily_alpha),
                actual_side=actual_side,
                predicted_side=row.get('predicted_side'),
                predicted_move=float(row.get('predicted_move')) if row.get('predicted_move') is not None else None,
                price_source='polygon',
                runid=int(run_id)  # Convert to regular int
            )
            
            logger.info(f"Successfully calculated price move for {ticker}: {price_change_percentage:.2f}% (alpha: {daily_alpha:.2f}%)")
            return price_move
            
        except Exception as e:
            logger.error(f"Error calculating price move for news_id {row.get('news_id')}: {e}")
            return None
    
    def calculate_price_moves_for_date_range(self, start_month: str, end_month: str, publisher: str = 'globenewswire_biotech') -> pd.DataFrame:
        """
        Calculate price moves for all news items in a date range using Polygon API.
        
        Args:
            start_month: Start month in YYYY-MM format (e.g., "2025-06")
            end_month: End month in YYYY-MM format (e.g., "2025-08")
            publisher: Publisher to filter by (default: globenewswire_biotech)
            
        Returns:
            DataFrame with calculated price moves
        """
        # Generate run ID
        run_id = self.generate_run_id()
        logger.info(f"Starting Polygon price move calculation from {start_month} to {end_month} with run_id: {run_id}")
        
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
        logger.info(f"After filtering for valid tickers: {len(news_df)} items")
        
        # Calculate price moves
        successful_calculations = 0
        failed_calculations = 0
        price_moves = []
        
        for index, row in news_df.iterrows():
            try:
                price_move = self.calculate_price_move(row, run_id)
                if price_move:
                    price_moves.append(price_move)
                    successful_calculations += 1
                    
                    # Store in database
                    if store_price_move(price_move):
                        logger.debug(f"Stored price move for news_id {row['news_id']}")
                    else:
                        logger.warning(f"Failed to store price move for news_id {row['news_id']}")
                else:
                    failed_calculations += 1
                    
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                failed_calculations += 1
                continue
        
        logger.info(f"Polygon price move calculation completed:")
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

def main():
    """Main function to run the Polygon price move calculator."""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate price moves using Polygon API')
    parser.add_argument('--start-month', type=str, default='2025-06', 
                       help='Start month in YYYY-MM format (default: 2025-06)')
    parser.add_argument('--end-month', type=str, default='2025-06', 
                       help='End month in YYYY-MM format (default: 2025-06)')
    parser.add_argument('--publisher', type=str, default='globenewswire_biotech',
                       help='Publisher to filter news (default: globenewswire_biotech)')
    
    args = parser.parse_args()
    
    try:
        calculator = PolygonPriceMoveCalculator()
        
        logger.info(f"Starting Polygon price move calculation from {args.start_month} to {args.end_month}")
        
        # Calculate price moves
        result_df = calculator.calculate_price_moves_for_date_range(
            start_month=args.start_month,
            end_month=args.end_month,
            publisher=args.publisher
        )
        
        if not result_df.empty:
            print(f"\nPolygon Price Move Calculation Summary:")
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
            
            # Save to CSV for inspection
            output_file = f"data/polygon_price_moves_{args.start_month}_to_{args.end_month}.csv"
            result_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("No price moves were calculated.")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

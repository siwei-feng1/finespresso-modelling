import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import logging
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolygonPriceDownloader:
    def __init__(self):
        """Initialize the Polygon API price downloader."""
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        
    def load_tickers(self, file_path: str = "data/biotech_tickers.csv") -> List[str]:
        """Load ticker symbols from CSV file."""
        try:
            df = pd.read_csv(file_path)
            tickers = df['yf_ticker'].dropna().tolist()
            logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
            return tickers
        except Exception as e:
            logger.error(f"Error loading tickers from {file_path}: {e}")
            return []
    
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
                logger.warning(f"No data returned for {ticker}: {data.get('status')}")
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
    
    def download_prices(self, 
                       start_date: str, 
                       end_date: str, 
                       tickers_file: str = "data/biotech_tickers.csv",
                       output_file: Optional[str] = None,
                       delay: float = 0.1) -> pd.DataFrame:
        """
        Download daily prices for all tickers in the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            tickers_file: Path to CSV file with ticker symbols
            output_file: Optional output file path to save results
            delay: Delay between API requests in seconds
            
        Returns:
            DataFrame with all price data
        """
        # Load tickers
        tickers = self.load_tickers(tickers_file)
        if not tickers:
            logger.error("No tickers loaded")
            return pd.DataFrame()
        
        # Validate dates
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()
        
        logger.info(f"Starting price download for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data = []
        successful_downloads = 0
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
            
            # Get price data
            raw_data = self.get_daily_prices(ticker, start_date, end_date)
            
            if raw_data:
                # Process the data
                df = self.process_price_data(raw_data, ticker)
                if not df.empty:
                    all_data.append(df)
                    successful_downloads += 1
                    logger.info(f"Successfully downloaded {len(df)} records for {ticker}")
                else:
                    logger.warning(f"No processed data for {ticker}")
            else:
                logger.warning(f"Failed to download data for {ticker}")
            
            # Add delay to respect API rate limits
            if i < len(tickers):
                time.sleep(delay)
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Download completed. {successful_downloads}/{len(tickers)} tickers successful. "
                       f"Total records: {len(combined_df)}")
            
            # Save to file if specified
            if output_file:
                combined_df.to_csv(output_file, index=False)
                logger.info(f"Data saved to {output_file}")
            
            return combined_df
        else:
            logger.warning("No data was downloaded")
            return pd.DataFrame()

def main():
    """Main function to run the price downloader."""
    # Example usage
    downloader = PolygonPriceDownloader()
    
    # Set date range (modify these as needed)
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # Download prices
    df = downloader.download_prices(
        start_date=start_date,
        end_date=end_date,
        output_file=f"data/polygon_prices_{start_date}_to_{end_date}.csv"
    )
    
    if not df.empty:
        print(f"\nDownload Summary:")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        print(f"\nFirst few records:")
        print(df.head())

if __name__ == "__main__":
    main()

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Polygon utility
from utils.price_move_util_polygon import download_with_retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_polygon_price_data():
    # Define biotech tickers from previous runs
    tickers = [
        'MRNA',  # Moderna
        'BNTX',  # BioNTech
        'NVAX',  # Novavax
        'INO',   # Inovio
        'VXRT',  # Vaxart
        'REGN',  # Regeneron
        'GILD',  # Gilead Sciences
        'AMGN',  # Amgen
        'BIIB',  # Biogen
        'VRTX'   # Vertex Pharmaceuticals
    ]
    
    # Use past week's data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Format dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Testing Polygon price data for {tickers} from {start_date_str} to {end_date_str}")
    
    for ticker in tickers:
        logger.info(f"Processing price data for {ticker}")
        try:
            # Download price data with retry
            price_data = download_with_retry(ticker, start_date_str, end_date_str)
            if price_data is not None and not price_data.empty:
                logger.info(f"Successfully retrieved price data for {ticker}:")
                logger.info(f"Number of records: {len(price_data)}")
                logger.info(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
                logger.info(f"Sample data:\n{price_data.head()}")
            else:
                logger.warning(f"No price data found for {ticker}")
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

def test_polygon_specific_cases():
    # List of (ticker, start_date, end_date) tuples from failed cases
    test_cases = [
        ("ANVS", "2020-12-16", "2020-12-18"),
        ("VBLT", "2022-04-25", "2022-04-27"),
        ("RAIN", "2022-05-03", "2022-05-05"),
        ("NEXI", "2022-05-11", "2022-05-13"),
        ("CFRX", "2022-09-09", "2022-09-13"),
        ("RAD", "2023-02-13", "2023-02-15"),  # RAD.AX, but Polygon may use just RAD for US stocks
    ]
    for ticker, start_date, end_date in test_cases:
        logger.info(f"Testing Polygon price data for {ticker} from {start_date} to {end_date}")
        try:
            price_data = download_with_retry(ticker, start_date, end_date)
            if price_data is not None and not price_data.empty:
                logger.info(f"Successfully retrieved price data for {ticker}:")
                logger.info(f"Number of records: {len(price_data)}")
                logger.info(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
                logger.info(f"Sample data:\n{price_data.head()}")
            else:
                logger.warning(f"No price data found for {ticker} in range {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

if __name__ == '__main__':
    test_polygon_price_data()
    test_polygon_specific_cases() 
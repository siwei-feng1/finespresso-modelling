import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary modules
from utils.db import news_db_util, price_move_db_util
from utils.price_move_util_polygon import get_news_with_price_moves
from utils.price_move_util_live import get_news_with_price_moves as get_news_with_price_moves_yf
from utils.db.news_db_util import get_news_df
from utils.price_move_util import create_price_moves, store_price_move
from utils.db.price_move_db_util import get_price_moves

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Schema fields for price_moves and news
PRICE_MOVES_FIELDS = [
    'id', 'news_id', 'ticker', 'published_date', 'begin_price', 'end_price',
    'index_begin_price', 'index_end_price', 'volume', 'market', 'price_change',
    'price_change_percentage', 'index_price_change', 'index_price_change_percentage',
    'daily_alpha', 'actual_side', 'predicted_side', 'predicted_move', 'downloaded_at'
]
NEWS_FIELDS = [
    'id', 'title', 'link', 'company', 'published_date', 'content', 'reason', 'industry',
    'publisher_topic', 'event', 'publisher', 'downloaded_at', 'status', 'instrument_id',
    'yf_ticker', 'ticker', 'published_date_gmt', 'timezone', 'publisher_summary',
    'ticker_url', 'predicted_side', 'predicted_move', 'langauge', 'language',
    'content_en', 'title_en'
]

def test_price_move_calculation():
    logger.info("Starting price move tests...")
    
    # Get news items from globenewswire_biotech
    news_df = get_news_df(publisher='globenewswire_biotech')
    logger.info(f"Retrieved {len(news_df)} news items")
    
    # Take first 5 items for testing
    test_news = news_df.head(5)
    logger.info(f"Testing price move calculation for {len(test_news)} news items from globenewswire_biotech")
    
    # Test Polygon price moves
    logger.info("Testing Polygon price moves...")
    polygon_moves = create_price_moves(test_news)
    if polygon_moves is not None and not polygon_moves.empty:
        logger.info(f"Successfully calculated {len(polygon_moves)} price moves using Polygon")
        logger.info("Sample data:")
        # Only log columns that exist
        expected_cols = ['news_id', 'ticker', 'actual_side', 'price_change_percentage']
        available_cols = [col for col in expected_cols if col in polygon_moves.columns]
        if available_cols:
            logger.info(polygon_moves[available_cols].head())
        else:
            logger.warning(f"Expected columns {expected_cols} not found. Available columns: {polygon_moves.columns.tolist()}")
            logger.info(polygon_moves.head())
    else:
        logger.error("Failed to calculate price moves using Polygon")
    
    logger.info("\nTesting Yahoo Finance price moves...")
    yf_moves = create_price_moves(test_news)
    if yf_moves is not None and not yf_moves.empty:
        logger.info(f"Successfully calculated {len(yf_moves)} price moves using Yahoo Finance")
        logger.info("Sample data:")
        expected_cols = ['news_id', 'ticker', 'actual_side', 'price_change_percentage']
        available_cols = [col for col in expected_cols if col in yf_moves.columns]
        if available_cols:
            logger.info(yf_moves[available_cols].head())
        else:
            logger.warning(f"Expected columns {expected_cols} not found. Available columns: {yf_moves.columns.tolist()}")
            logger.info(yf_moves.head())
    else:
        logger.error("Failed to calculate price moves using Yahoo Finance")
    
    logger.info("\n==================================================\n")

def test_price_move_storage():
    # Get news items from globenewswire_biotech
    news_df = get_news_df(publisher='globenewswire_biotech')
    logger.info(f"Retrieved {len(news_df)} news items")
    
    # Take first 3 items for testing
    test_news = news_df.head(3)
    logger.info(f"Testing price move storage for {len(test_news)} news items from globenewswire_biotech")
    
    # Calculate price moves using Yahoo Finance
    price_moves = create_price_moves(test_news, price_source='yfinance')
    if price_moves is not None and not price_moves.empty:
        # Filter out rows with missing or NaN required price columns
        required_cols = ['begin_price', 'end_price', 'index_begin_price', 'index_end_price']
        missing_cols = [col for col in required_cols if col not in price_moves.columns]
        if missing_cols:
            logger.error(f"No valid price moves to store (missing columns: {missing_cols})")
            return
        valid_moves = price_moves.dropna(subset=required_cols)
        if valid_moves.empty:
            logger.error("No valid price moves to store (all rows have missing price data)")
            return
        # Store each valid price move individually
        success = True
        for _, row in valid_moves.iterrows():
            if not store_price_move(row):
                success = False
        if success:
            logger.info(f"Successfully stored {len(valid_moves)} price moves")
        else:
            logger.error("Failed to store some price moves")
    else:
        logger.error("Failed to calculate price moves for storage test")
    
    logger.info("\n==================================================\n")

def test_price_move_retrieval():
    # Get all price moves
    price_moves_df = get_price_moves()
    if price_moves_df is not None and not price_moves_df.empty:
        logger.info(f"Retrieved {len(price_moves_df)} price moves from the database")
        logger.info("\nSample data:")
        # Only log columns that exist
        expected_cols = ['id', 'ticker', 'actual_side', 'price_change_percentage']
        available_cols = [col for col in expected_cols if col in price_moves_df.columns]
        if available_cols:
            logger.info(price_moves_df[available_cols].head())
        else:
            logger.warning(f"Expected columns {expected_cols} not found. Available columns: {price_moves_df.columns.tolist()}")
            logger.info(price_moves_df.head())
    else:
        logger.error("Failed to retrieve price moves")

if __name__ == "__main__":
    test_price_move_calculation()
    test_price_move_storage()
    test_price_move_retrieval() 
import logging
import os
from datetime import datetime, timedelta
import sys
import pandas as pd
from typing import List
from utils.db import news_db_util, price_move_db_util
from utils.db.price_move_db_util import PriceMove
from utils.price_move_util_live import get_news_with_price_moves as get_news_with_price_moves_yf
from utils.price_move_util_polygon import get_news_with_price_moves as get_news_with_price_moves_polygon

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

BATCH_SIZE = 20  # Increased batch size for better performance

def process_news_batch(news_batch: pd.DataFrame, price_source: str = 'yfinance') -> List[PriceMove]:
    """Process a batch of news items and return list of PriceMove objects"""
    price_moves = []
    successful_items = 0
    failed_items = 0
    
    # Get price moves based on the selected source
    if price_source == 'yfinance':
        processed_df = get_news_with_price_moves_yf(news_batch)
    elif price_source == 'polygon':
        processed_df = get_news_with_price_moves_polygon(news_batch)
    else:
        raise ValueError(f"Invalid price source: {price_source}")
    
    if processed_df is None or processed_df.empty:
        logger.warning("No valid price moves calculated for the batch")
        return []
    
    for _, row in processed_df.iterrows():
        try:
            if pd.isna(row.get('actual_side')):
                logger.warning(f"No valid price data for news_id: {row['news_id']}")
                failed_items += 1
                continue

            # Ensure all required fields are present and have correct types
            required_fields = {
                'news_id': int,
                'yf_ticker': str,
                'published_date': pd.Timestamp,
                'begin_price': float,
                'end_price': float,
                'index_begin_price': float,
                'index_end_price': float,
                'volume': float,
                'market': str,
                'price_change': float,
                'price_change_percentage': float,
                'index_price_change': float,
                'index_price_change_percentage': float,
                'daily_alpha': float,
                'actual_side': str
            }

            # Validate and convert fields
            for field, field_type in required_fields.items():
                if field not in row or pd.isna(row[field]):
                    raise ValueError(f"Missing required field: {field}")
                if field_type == pd.Timestamp:
                    if not isinstance(row[field], pd.Timestamp):
                        row[field] = pd.to_datetime(row[field])
                else:
                    row[field] = field_type(row[field])

            # Create PriceMove object
            price_move = PriceMove(
                news_id=row['news_id'],
                ticker=row['yf_ticker'],
                published_date=row['published_date'],
                begin_price=row['begin_price'],
                end_price=row['end_price'],
                index_begin_price=row['index_begin_price'],
                index_end_price=row['index_end_price'],
                volume=int(row['volume']) if not pd.isna(row['volume']) else None,
                market=row['market'],
                price_change=row['price_change'],
                price_change_percentage=row['price_change_percentage'],
                index_price_change=row['index_price_change'],
                index_price_change_percentage=row['index_price_change_percentage'],
                daily_alpha=row['daily_alpha'],
                actual_side=row['actual_side'],
                downloaded_at=datetime.utcnow(),
                price_source=price_source
            )
            price_moves.append(price_move)
            successful_items += 1
            
        except Exception as e:
            logger.error(f"Error processing news item: {e}")
            logger.exception("Detailed traceback:")
            failed_items += 1
            
    logger.info(f"Batch processing results - Successful: {successful_items}, Failed: {failed_items}")
    return price_moves

def store_price_move_batch(price_moves: List[PriceMove]) -> tuple[int, int]:
    """Store a batch of price moves and return count of successes and failures"""
    successful = 0
    failed = 0
    
    for price_move in price_moves:
        try:
            price_move_db_util.store_price_move(price_move)
            successful += 1
            logger.debug(f"Stored price move for news_id: {price_move.news_id}")
        except Exception as e:
            failed += 1
            logger.error(f"Failed to store price move for news_id {price_move.news_id}: {e}")
            
    return successful, failed

def run_price_move_task(publisher: str = None, price_source: str = 'yfinance'):
    """Run price move task for a publisher with batch processing. If publisher is None, process all publishers."""
    logger.info(f"Starting price move task{' for publisher: ' + publisher if publisher else ' for all publishers'}")
    logger.info(f"Using price source: {price_source}")

    # Get news data
    news_df = news_db_util.get_news_df(publisher)
    total_news = len(news_df)
    logger.info(f"Retrieved {total_news} news items{' for ' + publisher if publisher else ''}")

    if news_df.empty:
        logger.warning(f"No news items found{' for ' + publisher if publisher else ''}")
        return

    # Initialize counters
    total_successful = 0
    total_failed = 0
    batch_count = 0
    start_time = datetime.now()

    # Process in batches
    for start_idx in range(0, len(news_df), BATCH_SIZE):
        batch_count += 1
        end_idx = min(start_idx + BATCH_SIZE, len(news_df))
        news_batch = news_df.iloc[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_count} ({start_idx+1} to {end_idx} of {total_news})")
        batch_start_time = datetime.now()
        
        # Process the batch
        price_moves = process_news_batch(news_batch, price_source)
        
        # Store the batch
        if price_moves:
            successful, failed = store_price_move_batch(price_moves)
            total_successful += successful
            total_failed += failed
            
            logger.info(f"Batch {batch_count} results - Successful: {successful}, Failed: {failed}")
        else:
            logger.warning(f"Batch {batch_count} produced no valid price moves")
            total_failed += len(news_batch)

        # Log batch timing
        batch_duration = datetime.now() - batch_start_time
        logger.info(f"Batch {batch_count} completed in {batch_duration}")

    # Log final results
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    logger.info(f"\n=== Price move task completed{' for ' + publisher if publisher else ''} ===")
    logger.info(f"Price source: {price_source}")
    logger.info(f"Total batches processed: {batch_count}")
    logger.info(f"Final results - Successful: {total_successful}, Failed: {total_failed}")
    logger.info(f"Total news items: {total_news}, Processed: {total_successful + total_failed}")
    logger.info(f"Total duration: {total_duration}")
    logger.info(f"Average batch duration: {total_duration / batch_count if batch_count > 0 else 0}")

def test_price_move_task(news_id: int, price_source: str = 'yfinance'):
    """Test processing for a single news item"""
    logger.info(f"Starting test price move task for news_id: {news_id}")
    logger.info(f"Using price source: {price_source}")

    news_df = news_db_util.get_news_by_id(news_id)
    if news_df.empty:
        logger.error(f"No news item found with id: {news_id}")
        return

    # Process single item as a batch
    price_moves = process_news_batch(news_df, price_source)
    if price_moves:
        successful, failed = store_price_move_batch(price_moves)
        logger.info(f"Test results - Successful: {successful}, Failed: {failed}")
    else:
        logger.warning("Test produced no valid price moves")

if __name__ == "__main__":
    # Get all news items without filtering by publisher
    publisher = 'omx'
    #run_price_move_task(publisher)
    
    # Uncomment to test specific news item:
    test_price_move_task(23436)

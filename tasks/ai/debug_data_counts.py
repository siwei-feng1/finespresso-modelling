#!/usr/bin/env python3
"""
Debug script to check data counts in the database
"""

import os
import sys
import pandas as pd

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.db.news_db_util import get_news_df
from utils.db.price_move_db_util import get_price_moves, get_raw_price_moves
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

def check_data_counts():
    """Check counts in various tables and joins"""
    
    logger.info("=== Database Data Counts Analysis ===")
    
    # Get raw news data
    logger.info("Loading raw news data...")
    news_df = get_news_df()
    logger.info(f"Total news records: {len(news_df)}")
    
    # Get raw price moves data
    logger.info("Loading raw price moves data...")
    raw_price_moves_df = get_raw_price_moves()
    logger.info(f"Total price moves records: {len(raw_price_moves_df)}")
    
    # Get joined data (what we're currently using)
    logger.info("Loading joined price moves data...")
    joined_df = get_price_moves()
    logger.info(f"Total joined records: {len(joined_df)}")
    
    # Calculate missing records
    missing_in_join = len(raw_price_moves_df) - len(joined_df)
    logger.info(f"Records lost in join: {missing_in_join}")
    logger.info(f"Join efficiency: {len(joined_df)/len(raw_price_moves_df)*100:.2f}%")
    
    # Check for orphaned price moves (price moves without corresponding news)
    if not raw_price_moves_df.empty:
        orphaned_price_moves = raw_price_moves_df[~raw_price_moves_df['news_id'].isin(news_df['news_id'])]
        logger.info(f"Orphaned price moves (no corresponding news): {len(orphaned_price_moves)}")
        
        # Check for news without price moves
        news_without_price_moves = news_df[~news_df['news_id'].isin(raw_price_moves_df['news_id'])]
        logger.info(f"News without price moves: {len(news_without_price_moves)}")
    
    # Check event distribution in joined data
    if not joined_df.empty:
        logger.info("\n=== Event Distribution in Joined Data ===")
        event_counts = joined_df['event'].value_counts()
        logger.info(f"Total unique events: {len(event_counts)}")
        logger.info(f"Top 10 events by count:")
        for event, count in event_counts.head(10).items():
            logger.info(f"  {event}: {count}")
    
    # Check actual_side distribution
    if not joined_df.empty:
        logger.info("\n=== Actual Side Distribution ===")
        side_counts = joined_df['actual_side'].value_counts()
        logger.info(f"Actual side counts:")
        for side, count in side_counts.items():
            logger.info(f"  {side}: {count}")
    
    logger.info("\n=== Analysis Complete ===")

if __name__ == "__main__":
    check_data_counts() 
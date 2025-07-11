#!/usr/bin/env python3
"""
Download all news, instrument, and price move data from the database and save as CSV files.
This script provides a comprehensive data export for analysis and backup purposes.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db.news_db_util import get_news_df, get_news_latest_df
from utils.db.instrument_db_util import get_all_instruments
from utils.db.price_move_db_util import get_price_moves, get_price_moves_date_range, get_raw_price_moves
from utils.db_pool import DatabasePool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_data_directory():
    """Ensure the data directory exists"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory: {data_dir}")
    return data_dir

def download_news_data(data_dir):
    """Download all news data and save to CSV"""
    logger.info("Starting news data download...")
    
    try:
        # Get all news data
        logger.info("Downloading all news data...")
        news_df = get_news_df()
        
        if not news_df.empty:
            # Save all news data
            news_file = os.path.join(data_dir, 'all_news.csv')
            news_df.to_csv(news_file, index=False)
            logger.info(f"Saved {len(news_df)} news records to {news_file}")
            
            # Also save latest 1000 news items
            logger.info("Downloading latest 1000 news items...")
            latest_news_df = get_news_latest_df()
            
            if not latest_news_df.empty:
                latest_news_file = os.path.join(data_dir, 'latest_news.csv')
                latest_news_df.to_csv(latest_news_file, index=False)
                logger.info(f"Saved {len(latest_news_df)} latest news records to {latest_news_file}")
            
            # Create summary statistics
            news_summary = {
                'total_records': len(news_df),
                'date_range': {
                    'earliest': news_df['published_date'].min() if 'published_date' in news_df.columns else 'N/A',
                    'latest': news_df['published_date'].max() if 'published_date' in news_df.columns else 'N/A'
                },
                'publishers': news_df['publisher'].value_counts().to_dict() if 'publisher' in news_df.columns else {},
                'events': news_df['event'].value_counts().to_dict() if 'event' in news_df.columns else {},
                'tickers': news_df['ticker'].value_counts().head(20).to_dict() if 'ticker' in news_df.columns else {}
            }
            
            # Save summary as JSON
            import json
            summary_file = os.path.join(data_dir, 'news_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(news_summary, f, indent=2, default=str)
            logger.info(f"Saved news summary to {summary_file}")
            
        else:
            logger.warning("No news data found in the database")
            
    except Exception as e:
        logger.error(f"Error downloading news data: {str(e)}")
        logger.exception("Full traceback:")

def download_instrument_data(data_dir):
    """Download all instrument data and save to CSV"""
    logger.info("Starting instrument data download...")
    
    try:
        # Get all instrument data
        logger.info("Downloading all instrument data...")
        instrument_df = get_all_instruments()
        
        if not instrument_df.empty:
            # Save all instrument data
            instrument_file = os.path.join(data_dir, 'all_instruments.csv')
            instrument_df.to_csv(instrument_file, index=False)
            logger.info(f"Saved {len(instrument_df)} instrument records to {instrument_file}")
            
            # Create summary statistics
            instrument_summary = {
                'total_records': len(instrument_df),
                'asset_classes': instrument_df['asset_class'].value_counts().to_dict() if 'asset_class' in instrument_df.columns else {},
                'exchanges': instrument_df['exchange'].value_counts().to_dict() if 'exchange' in instrument_df.columns else {},
                'countries': instrument_df['country'].value_counts().to_dict() if 'country' in instrument_df.columns else {},
                'sectors': instrument_df['sector'].value_counts().to_dict() if 'sector' in instrument_df.columns else {}
            }
            
            # Save summary as JSON
            import json
            summary_file = os.path.join(data_dir, 'instrument_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(instrument_summary, f, indent=2, default=str)
            logger.info(f"Saved instrument summary to {summary_file}")
            
        else:
            logger.warning("No instrument data found in the database")
            
    except Exception as e:
        logger.error(f"Error downloading instrument data: {str(e)}")
        logger.exception("Full traceback:")

def download_price_move_data(data_dir):
    """Download all price move data and save to CSV"""
    logger.info("Starting price move data download...")
    
    try:
        # Get all raw price move data
        logger.info("Downloading all raw price move data...")
        raw_price_move_df = get_raw_price_moves()
        
        if not raw_price_move_df.empty:
            # Save raw price move data
            raw_price_move_file = os.path.join(data_dir, 'all_price_moves.csv')
            raw_price_move_df.to_csv(raw_price_move_file, index=False)
            logger.info(f"Saved {len(raw_price_move_df)} raw price move records to {raw_price_move_file}")
            
            # Create summary statistics for raw price moves
            price_move_summary = {
                'total_records': len(raw_price_move_df),
                'date_range': {
                    'earliest': raw_price_move_df['published_date'].min() if 'published_date' in raw_price_move_df.columns else 'N/A',
                    'latest': raw_price_move_df['published_date'].max() if 'published_date' in raw_price_move_df.columns else 'N/A'
                },
                'actual_sides': raw_price_move_df['actual_side'].value_counts().to_dict() if 'actual_side' in raw_price_move_df.columns else {},
                'predicted_sides': raw_price_move_df['predicted_side'].value_counts().to_dict() if 'predicted_side' in raw_price_move_df.columns else {},
                'tickers': raw_price_move_df['ticker'].value_counts().head(20).to_dict() if 'ticker' in raw_price_move_df.columns else {},
                'markets': raw_price_move_df['market'].value_counts().to_dict() if 'market' in raw_price_move_df.columns else {},
                'price_sources': raw_price_move_df['price_source'].value_counts().to_dict() if 'price_source' in raw_price_move_df.columns else {},
                'runids': raw_price_move_df['runid'].value_counts().to_dict() if 'runid' in raw_price_move_df.columns else {}
            }
            
            # Calculate some basic statistics for price changes
            if 'price_change_percentage' in raw_price_move_df.columns:
                price_changes = raw_price_move_df['price_change_percentage'].dropna()
                if len(price_changes) > 0:
                    price_move_summary['price_change_stats'] = {
                        'mean': float(price_changes.mean()),
                        'median': float(price_changes.median()),
                        'std': float(price_changes.std()),
                        'min': float(price_changes.min()),
                        'max': float(price_changes.max())
                    }
            
            # Save summary as JSON
            import json
            summary_file = os.path.join(data_dir, 'price_move_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(price_move_summary, f, indent=2, default=str)
            logger.info(f"Saved price move summary to {summary_file}")
            
        else:
            logger.warning("No raw price move data found in the database")
            
    except Exception as e:
        logger.error(f"Error downloading price move data: {str(e)}")
        logger.exception("Full traceback:")

def download_recent_data(data_dir, days=30):
    """Download recent data for the last N days"""
    logger.info(f"Starting recent data download (last {days} days)...")
    
    try:
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Downloading data from {start_date} to {end_date}")
        
        # Get recent price moves
        recent_price_moves = get_price_moves_date_range(start_date, end_date)
        
        if not recent_price_moves.empty:
            recent_file = os.path.join(data_dir, f'recent_price_moves_{days}days.csv')
            recent_price_moves.to_csv(recent_file, index=False)
            logger.info(f"Saved {len(recent_price_moves)} recent price move records to {recent_file}")
        else:
            logger.warning(f"No recent price move data found for the last {days} days")
            
    except Exception as e:
        logger.error(f"Error downloading recent data: {str(e)}")
        logger.exception("Full traceback:")

def create_download_report(data_dir):
    """Create a comprehensive download report"""
    logger.info("Creating download report...")
    
    try:
        report = {
            'download_timestamp': datetime.now().isoformat(),
            'data_directory': data_dir,
            'files_created': []
        }
        
        # Check what files were created
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') or filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                file_size = os.path.getsize(filepath)
                report['files_created'].append({
                    'filename': filename,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
        
        # Save report
        import json
        report_file = os.path.join(data_dir, 'download_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Download report saved to {report_file}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("DOWNLOAD SUMMARY")
        print("="*50)
        print(f"Data directory: {data_dir}")
        print(f"Download timestamp: {report['download_timestamp']}")
        print(f"Files created: {len(report['files_created'])}")
        print("\nFiles:")
        for file_info in report['files_created']:
            print(f"  - {file_info['filename']} ({file_info['size_mb']} MB)")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error creating download report: {str(e)}")

def main():
    """Main function to orchestrate the download process"""
    logger.info("Starting comprehensive data download...")
    
    try:
        # Ensure data directory exists
        data_dir = ensure_data_directory()
        
        # Download all data types
        download_news_data(data_dir)
        download_instrument_data(data_dir)
        download_price_move_data(data_dir)
        
        # Download recent data (last 30 days)
        download_recent_data(data_dir, days=30)
        
        # Create download report
        create_download_report(data_dir)
        
        logger.info("Data download completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main download process: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main() 
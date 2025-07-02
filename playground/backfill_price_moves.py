#!/usr/bin/env python3
"""
Backfill script for processing existing news from database and calculating price moves.

This script:
1. Gets all existing news from the database using news_db_util
2. For news with yf_ticker: calculates price moves and stores to DB/CSV
3. For news without yf_ticker: extracts ticker using OpenAI and stores to DB
4. Uses SPY as the market index
5. Provides comprehensive logging and statistics
"""

import pandas as pd
import logging
import os
import sys
from datetime import datetime, time, timedelta
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv
import json
import time as time_module

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai.openai_util import client
from utils.date.date_adjuster import get_previous_trading_day, get_next_trading_day
from utils.db.news_db_util import get_news_df, update_news_tickers
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

class DatabaseBackfillProcessor:
    def __init__(self):
        self.openai_client = client
        self.index_symbol = 'SPY'  # S&P 500 ETF as market index
        self.stats = {
            'total_news': 0,
            'with_ticker': 0,
            'without_ticker': 0,
            'ticker_extracted': 0,
            'price_moves_calculated': 0,
            'price_moves_stored_db': 0,
            'price_moves_stored_csv': 0,
            'errors': 0
        }
        
    def extract_ticker_from_company(self, company_name: str, news_text: str = "") -> Optional[str]:
        """Extract ticker symbol from company name using OpenAI."""
        try:
            if not company_name:
                return None
                
            # Combine company name with news text for better context
            text = f"Company: {company_name}"
            if news_text:
                text += f" News: {news_text[:500]}"  # Limit text length
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract publicly traded companies from news text. Return JSON with 'tickers' and 'companies' arrays. Example: {\"tickers\": [\"AAPL\", \"MSFT\"], \"companies\": [\"Apple Inc\", \"Microsoft Corporation\"]}. If no companies found, return {\"tickers\": [], \"companies\": []}."
                    },
                    {
                        "role": "user",
                        "content": f"Extract company tickers and names from this text: {text}"
                    }
                ],
                max_tokens=100,
                temperature=0
            )
            
            result = response.choices[0].message.content
            try:
                parsed = json.loads(result)
                tickers = parsed.get('tickers', [])
                return tickers[0] if tickers else None
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse OpenAI response: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting ticker for {company_name}: {e}")
            return None
    
    def get_price_data(self, ticker: str, published_date: datetime) -> Optional[Dict]:
        """Get price data for ticker around the published date."""
        try:
            # Handle timezone-aware datetime
            if published_date.tzinfo is None:
                # Assume UTC if no timezone info
                published_date = published_date.replace(tzinfo=None)
            
            pub_time = published_date.time()
            pub_date = published_date.date()
            
            # Determine market timing based on publication time
            if time(9, 30) <= pub_time < time(16, 0):
                market = 'regular_market'
            elif time(16, 0) <= pub_time:
                market = 'after_market'
            elif time(0, 0) <= pub_time < time(9, 30):
                market = 'pre_market'
            else:
                market = 'regular_market'
            
            # Get trading days
            previous_trading_day = get_previous_trading_day(pub_date)
            next_trading_day = get_next_trading_day(pub_date)
            
            # Format dates for yfinance
            yf_prev_date = previous_trading_day.strftime('%Y-%m-%d')
            yf_today_date = pub_date.strftime('%Y-%m-%d')
            yf_next_date = next_trading_day.strftime('%Y-%m-%d')
            
            logger.debug(f"Getting price data for {ticker} on {yf_today_date}, market: {market}")
            
            # Download price data
            data = yf.download(ticker, start=yf_prev_date, end=yf_next_date, interval='1d')
            index_data = yf.download(self.index_symbol, start=yf_prev_date, end=yf_next_date, interval='1d')
            
            if data.empty or index_data.empty:
                logger.warning(f"No price data available for {ticker}")
                return None
            
            # Extract prices based on market timing
            if market == 'pre_market':
                if yf_prev_date not in data.index or yf_today_date not in data.index:
                    return None
                begin_price = float(data.loc[yf_prev_date, 'Close'].iloc[0])
                end_price = float(data.loc[yf_today_date, 'Open'].iloc[0])
                index_begin_price = float(index_data.loc[yf_prev_date, 'Close'].iloc[0])
                index_end_price = float(index_data.loc[yf_today_date, 'Open'].iloc[0])
            elif market == 'regular_market':
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
            
            volume = float(data.loc[yf_today_date, 'Volume'].iloc[0]) if yf_today_date in data.index else 0
            
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
                'market': market
            }
            
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {e}")
            return None
    
    def process_news_with_ticker(self, news_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Process news items that already have tickers and calculate price moves."""
        logger.info(f"Processing {len(news_df)} news items with existing tickers...")
        
        price_moves_data = []
        processed_count = 0
        
        for idx, row in news_df.iterrows():
            try:
                ticker = row['yf_ticker'] or row['ticker']
                if not ticker:
                    continue
                    
                published_date = row['published_date']
                if pd.isna(published_date):
                    continue
                
                # Get price data
                price_data = self.get_price_data(ticker, published_date)
                if price_data:
                    # Combine news data with price data
                    combined_data = {
                        'news_id': row['id'],
                        'title': row.get('title', ''),
                        'description': row.get('content', ''),
                        'link': row.get('link', ''),
                        'company': row.get('company', ''),
                        'ticker': ticker,
                        'published_date': published_date,
                        'publisher': row.get('publisher', ''),
                        'language': row.get('language', ''),
                        **price_data
                    }
                    price_moves_data.append(combined_data)
                    processed_count += 1
                    
                    # Store to database
                    price_move_obj = PriceMove(
                        news_id=row['id'],
                        ticker=ticker,
                        published_date=published_date,
                        begin_price=price_data['begin_price'],
                        end_price=price_data['end_price'],
                        index_begin_price=price_data['index_begin_price'],
                        index_end_price=price_data['index_end_price'],
                        volume=price_data['volume'],
                        market=price_data['market'],
                        price_change=price_data['price_change'],
                        price_change_percentage=price_data['price_change_percentage'],
                        index_price_change=price_data['index_price_change'],
                        index_price_change_percentage=price_data['index_price_change_percentage'],
                        daily_alpha=price_data['daily_alpha'],
                        actual_side=price_data['actual_side'],
                        predicted_side=row.get('predicted_side'),
                        predicted_move=row.get('predicted_move'),
                        price_source='yfinance'
                    )
                    
                    if store_price_move(price_move_obj):
                        self.stats['price_moves_stored_db'] += 1
                    else:
                        logger.warning(f"Failed to store price move for news_id {row['id']}")
                
                # Add small delay to avoid rate limiting
                time_module.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing news item {row.get('id', 'unknown')}: {e}")
                self.stats['errors'] += 1
        
        self.stats['price_moves_calculated'] = processed_count
        logger.info(f"Successfully processed {processed_count} price moves for news with tickers")
        
        return pd.DataFrame(price_moves_data), price_moves_data
    
    def process_news_without_ticker(self, news_df: pd.DataFrame) -> List[Dict]:
        """Process news items without tickers and extract tickers using OpenAI."""
        logger.info(f"Processing {len(news_df)} news items without tickers...")
        
        extracted_tickers = []
        extracted_count = 0
        
        for idx, row in news_df.iterrows():
            try:
                company_name = row.get('company', '')
                if not company_name:
                    continue
                
                # Extract ticker using OpenAI
                news_text = f"{row.get('title', '')} {row.get('content', '')}"
                ticker = self.extract_ticker_from_company(company_name, news_text)
                
                if ticker:
                    extracted_tickers.append({
                        'news_id': row['id'],
                        'ticker': ticker,
                        'yf_ticker': ticker,
                        'instrument_id': None,
                        'ticker_url': f"https://finance.yahoo.com/quote/{ticker}"
                    })
                    extracted_count += 1
                    logger.info(f"Extracted ticker {ticker} for company {company_name}")
                
                # Add delay to avoid rate limiting
                time_module.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error extracting ticker for news item {row.get('id', 'unknown')}: {e}")
                self.stats['errors'] += 1
        
        self.stats['ticker_extracted'] = extracted_count
        logger.info(f"Successfully extracted {extracted_count} tickers")
        
        # Update database with extracted tickers
        if extracted_tickers:
            update_news_tickers(extracted_tickers)
            logger.info(f"Updated database with {len(extracted_tickers)} extracted tickers")
        
        return extracted_tickers
    
    def save_results_to_csv(self, price_moves_df: pd.DataFrame, output_dir: str = 'data'):
        """Save price moves results to CSV file with timestamp."""
        if price_moves_df.empty:
            logger.warning("No price moves data to save to CSV")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'price_moves_backfill_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        
        price_moves_df.to_csv(filepath, index=False)
        self.stats['price_moves_stored_csv'] = len(price_moves_df)
        logger.info(f"Saved {len(price_moves_df)} price moves to {filepath}")
        
        return filepath
    
    def print_statistics(self):
        """Print comprehensive statistics about the backfill process."""
        logger.info("=" * 60)
        logger.info("BACKFILL PROCESS STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total news items processed: {self.stats['total_news']}")
        logger.info(f"News with existing tickers: {self.stats['with_ticker']}")
        logger.info(f"News without tickers: {self.stats['without_ticker']}")
        logger.info(f"Tickers extracted using OpenAI: {self.stats['ticker_extracted']}")
        logger.info(f"Price moves calculated: {self.stats['price_moves_calculated']}")
        logger.info(f"Price moves stored to database: {self.stats['price_moves_stored_db']}")
        logger.info(f"Price moves stored to CSV: {self.stats['price_moves_stored_csv']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info("=" * 60)
    
    def run_backfill(self) -> pd.DataFrame:
        """Main backfill process."""
        logger.info("Starting database backfill process...")
        
        # Get all news from database
        logger.info("Fetching all news from database...")
        news_df = get_news_df()
        self.stats['total_news'] = len(news_df)
        
        if news_df.empty:
            logger.warning("No news found in database")
            return pd.DataFrame()
        
        logger.info(f"Retrieved {len(news_df)} news items from database")
        
        # Split news into with/without tickers
        news_with_ticker = news_df[news_df['yf_ticker'].notna() | news_df['ticker'].notna()].copy()
        news_without_ticker = news_df[(news_df['yf_ticker'].isna()) & (news_df['ticker'].isna())].copy()
        
        self.stats['with_ticker'] = len(news_with_ticker)
        self.stats['without_ticker'] = len(news_without_ticker)
        
        logger.info(f"News with tickers: {len(news_with_ticker)}")
        logger.info(f"News without tickers: {len(news_without_ticker)}")
        
        # Process news with tickers
        price_moves_df = pd.DataFrame()
        if not news_with_ticker.empty:
            price_moves_df, _ = self.process_news_with_ticker(news_with_ticker)
        
        # Process news without tickers
        if not news_without_ticker.empty:
            extracted_tickers = self.process_news_without_ticker(news_without_ticker)
            
            # If we extracted tickers, try to calculate price moves for them
            if extracted_tickers:
                # Get the news items that now have tickers
                news_ids_with_tickers = [item['news_id'] for item in extracted_tickers]
                news_with_new_tickers = news_without_ticker[news_without_ticker['id'].isin(news_ids_with_tickers)].copy()
                
                # Update the ticker columns
                for item in extracted_tickers:
                    mask = news_with_new_tickers['id'] == item['news_id']
                    news_with_new_tickers.loc[mask, 'ticker'] = item['ticker']
                    news_with_new_tickers.loc[mask, 'yf_ticker'] = item['yf_ticker']
                
                # Calculate price moves for newly extracted tickers
                if not news_with_new_tickers.empty:
                    new_price_moves_df, _ = self.process_news_with_ticker(news_with_new_tickers)
                    if not new_price_moves_df.empty:
                        price_moves_df = pd.concat([price_moves_df, new_price_moves_df], ignore_index=True)
        
        # Save results to CSV
        if not price_moves_df.empty:
            csv_filepath = self.save_results_to_csv(price_moves_df)
            logger.info(f"Backfill results saved to: {csv_filepath}")
        
        # Print statistics
        self.print_statistics()
        
        logger.info("Database backfill process completed!")
        return price_moves_df

def main():
    """Main function to run the backfill process."""
    processor = DatabaseBackfillProcessor()
    
    try:
        # Run the backfill process
        price_moves_df = processor.run_backfill()
        
        if not price_moves_df.empty:
            logger.info("\nSample results:")
            sample_cols = ['company', 'ticker', 'published_date', 'price_change_percentage', 'actual_side']
            print(price_moves_df[sample_cols].head())
        else:
            logger.warning("No price moves were calculated")
            
    except Exception as e:
        logger.error(f"Error in backfill process: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main() 
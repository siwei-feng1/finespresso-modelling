#!/usr/bin/env python3
"""
Backfill script for processing RSS feeds and calculating price moves.

This script:
1. Fetches RSS feeds from GlobeNewswire
2. Extracts all fields including company from contributor field
3. Uses OpenAI to extract ticker symbols from company names
4. Calculates price moves based on publication time relative to market hours
5. Saves results to CSV files with timestamps
"""

import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta
import logging
import os
import sys
from typing import List, Dict, Optional
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv
import json
import time as time_module

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai.openai_util import client
from utils.date.date_adjuster import get_previous_trading_day, get_next_trading_day

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backfill.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RSSBackfillProcessor:
    def __init__(self):
        self.openai_client = client
        self.index_symbol = 'SPY'  # S&P 500 ETF as market index
        
    def fetch_rss_feed(self, rss_url: str) -> Optional[str]:
        """Fetch RSS feed content from URL."""
        try:
            logger.info(f"Fetching RSS feed from: {rss_url}")
            response = requests.get(rss_url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching RSS feed: {e}")
            return None
    
    def parse_rss_feed(self, rss_content: str) -> List[Dict]:
        """Parse RSS content and extract all fields including company from contributor."""
        try:
            # Parse XML
            root = ET.fromstring(rss_content)
            
            # Define namespaces
            namespaces = {
                'dc': 'http://dublincore.org/documents/dcmi-namespace/',
                'media': 'http://search.yahoo.com/mrss/'
            }
            
            items = []
            
            # Find all item elements
            for item in root.findall('.//item'):
                item_data = {}
                
                # Extract basic RSS fields
                for field in ['guid', 'link', 'title', 'description', 'pubDate']:
                    element = item.find(field)
                    if element is not None:
                        item_data[field] = element.text
                
                # Extract categories
                categories = []
                for category in item.findall('category'):
                    domain = category.get('domain', '')
                    category_text = category.text
                    if domain:
                        categories.append(f"{domain}:{category_text}")
                    else:
                        categories.append(category_text)
                item_data['categories'] = categories
                
                # Extract Dublin Core fields
                for dc_field in ['identifier', 'language', 'publisher', 'contributor', 'modified', 'subject']:
                    element = item.find(f'dc:{dc_field}', namespaces)
                    if element is not None:
                        item_data[dc_field] = element.text
                
                # Extract company from contributor field
                if 'contributor' in item_data:
                    item_data['company'] = item_data['contributor']
                else:
                    item_data['company'] = None
                
                items.append(item_data)
            
            logger.info(f"Parsed {len(items)} items from RSS feed")
            return items
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed: {e}")
            return []
    
    def extract_ticker_from_company(self, company_name: str, news_text: str = "", categories: List[str] = None) -> Optional[str]:
        """Extract ticker symbol from company name using OpenAI and RSS categories."""
        try:
            if not company_name:
                return None
            
            # First, try to extract ticker from RSS categories (e.g., "stock:ASM")
            if categories:
                for category in categories:
                    if 'stock:' in category:
                        ticker = category.split('stock:')[-1]
                        if ticker and len(ticker) <= 10:  # Reasonable ticker length
                            logger.info(f"Found ticker {ticker} from RSS category for {company_name}")
                            return ticker
            
            # If no ticker found in categories, use OpenAI
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
            # Convert to US Eastern time if needed (assuming RSS is in GMT)
            # For simplicity, we'll work with the date as-is for now
            
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
            
            logger.info(f"Getting price data for {ticker} on {yf_today_date}, market: {market}")
            
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
    
    def process_rss_feed(self, rss_url: str) -> pd.DataFrame:
        """Process RSS feed and return DataFrame with price moves."""
        # Fetch RSS content
        rss_content = self.fetch_rss_feed(rss_url)
        if not rss_content:
            return pd.DataFrame()
        
        # Parse RSS items
        items = self.parse_rss_feed(rss_content)
        if not items:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Parse publication dates - handle multiple formats
        def parse_pub_date(date_str):
            if pd.isna(date_str):
                return None
            try:
                # Try the standard RSS format first
                return pd.to_datetime(date_str, format='%a, %d %b %Y %H:%M:%S %Z')
            except:
                try:
                    # Try without timezone
                    return pd.to_datetime(date_str, format='%a, %d %b %Y %H:%M:%S')
                except:
                    try:
                        # Try just the date
                        return pd.to_datetime(date_str)
                    except:
                        logger.warning(f"Could not parse date: {date_str}")
                        return None
        
        df['pubDate'] = df['pubDate'].apply(parse_pub_date)
        
        # Extract tickers
        logger.info("Extracting ticker symbols...")
        df['ticker'] = df.apply(
            lambda row: self.extract_ticker_from_company(
                row['company'], 
                row.get('title', '') + ' ' + row.get('description', ''),
                row.get('categories', [])
            ), 
            axis=1
        )
        
        # Filter out items without tickers
        df_with_tickers = df[df['ticker'].notna()].copy()
        logger.info(f"Found tickers for {len(df_with_tickers)} out of {len(df)} items")
        
        # Get price data for each item
        logger.info("Calculating price moves...")
        price_data_list = []
        
        for idx, row in df_with_tickers.iterrows():
            ticker = row['ticker']
            pub_date = row['pubDate']
            
            if pd.isna(pub_date):
                continue
                
            price_data = self.get_price_data(ticker, pub_date)
            if price_data:
                # Combine RSS data with price data
                combined_data = {
                    'news_id': row.get('identifier'),
                    'title': row.get('title'),
                    'description': row.get('description'),
                    'link': row.get('link'),
                    'company': row.get('company'),
                    'ticker': ticker,
                    'published_date': pub_date,
                    'categories': row.get('categories', []),
                    'publisher': row.get('publisher'),
                    'language': row.get('language'),
                    **price_data
                }
                price_data_list.append(combined_data)
            
            # Add small delay to avoid rate limiting
            time_module.sleep(0.1)
        
        result_df = pd.DataFrame(price_data_list)
        logger.info(f"Successfully processed {len(result_df)} price moves")
        
        return result_df
    
    def save_results(self, df: pd.DataFrame, output_dir: str = 'data'):
        """Save results to CSV file with timestamp."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'price_moves_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
        
        return filepath

def main():
    """Main function to run the backfill process."""
    # RSS feed URL
    rss_url = "https://www.globenewswire.com/RssFeed/industry/9576-Semiconductors/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Semiconductors"
    
    # Initialize processor
    processor = RSSBackfillProcessor()
    
    # Process RSS feed
    logger.info("Starting RSS backfill process...")
    df = processor.process_rss_feed(rss_url)
    
    if df.empty:
        logger.warning("No data to save")
        return
    
    # Save results
    filepath = processor.save_results(df)
    
    # Print summary
    logger.info("Backfill process completed!")
    logger.info(f"Total records processed: {len(df)}")
    logger.info(f"Results saved to: {filepath}")
    
    # Print sample of results
    if not df.empty:
        logger.info("\nSample results:")
        print(df[['company', 'ticker', 'published_date', 'price_change_percentage', 'actual_side']].head())

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
GlobeNewswire News Downloader

This script downloads news from GlobeNewswire RSS feeds, extracts tickers using OpenAI,
and stores the data in the database for further processing.
"""

import sys
import os
import yaml
import requests
import feedparser
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy.exc import ProgrammingError

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai.openai_util import extract_ticker, extract_issuer
from utils.db.news_db_util import News, add_news_items, map_to_db
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

class GlobeNewswireNewsDownloader:
    """Downloads and processes news from GlobeNewswire RSS feeds."""
    
    def __init__(self, rss_yaml_path: str = "data/globenewswire_rss.yaml"):
        self.rss_yaml_path = rss_yaml_path
        self.session = requests.Session()
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/rss+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
    def load_rss_feeds(self) -> List[Dict[str, Any]]:
        """Load RSS feeds from YAML file."""
        try:
            with open(self.rss_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            feeds = data.get('feeds', [])
            logger.info(f"Loaded {len(feeds)} RSS feeds from {self.rss_yaml_path}")
            return feeds
            
        except Exception as e:
            logger.error(f"Failed to load RSS feeds from {self.rss_yaml_path}: {e}")
            raise
    
    def fetch_rss_feed(self, rss_url: str) -> Optional[str]:
        """Fetch RSS feed content."""
        try:
            logger.debug(f"Fetching RSS feed: {rss_url}")
            response = self.session.get(rss_url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch RSS feed {rss_url}: {e}")
            return None
    
    def parse_rss_feed(self, rss_content: str, feed_title: str) -> List[Dict[str, Any]]:
        """Parse RSS feed content and extract news items."""
        try:
            feed = feedparser.parse(rss_content)
            items = []
            
            for entry in feed.entries:
                # Extract basic information
                title = entry.get('title', '')
                link = entry.get('link', '')
                published = entry.get('published', '')
                summary = entry.get('summary', '')
                content = entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''
                
                # Use content if available, otherwise use summary
                full_content = content if content else summary
                
                # Parse published date
                published_date = None
                if published:
                    try:
                        # Try to parse the date using feedparser's date parsing
                        parsed_date = feedparser.parse(f"<rss><item><pubDate>{published}</pubDate></item></rss>")
                        if parsed_date.entries and parsed_date.entries[0].get('published_parsed'):
                            # Convert time.struct_time to timestamp
                            import time
                            timestamp = time.mktime(parsed_date.entries[0].published_parsed)
                            published_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    except Exception as e:
                        logger.warning(f"Failed to parse date '{published}': {e}")
                
                # Extract company name from title or content
                company = self.extract_company_from_text(title + " " + full_content)
                
                # Extract ticker using OpenAI
                ticker = None
                if company:
                    try:
                        ticker = extract_ticker(company)
                        logger.debug(f"Extracted ticker '{ticker}' for company '{company}'")
                    except Exception as e:
                        logger.warning(f"Failed to extract ticker for company '{company}': {e}")
                
                # Create news item
                news_item = {
                    'title': title,
                    'link': link,
                    'company': company,
                    'published_date': published_date,
                    'content': full_content,
                    'reason': '',  # Will be enriched later
                    'industry': '',  # Will be enriched later
                    'publisher_topic': feed_title,
                    'event': '',  # Will be enriched later
                    'publisher': 'globenewswire',
                    'status': 'raw',
                    'instrument_id': None,  # Will be enriched later
                    'yf_ticker': ticker,  # Using extracted ticker as yf_ticker
                    'ticker': ticker,
                    'published_date_gmt': published_date,
                    'timezone': 'UTC',
                    'publisher_summary': summary,
                    'ticker_url': '',
                    'predicted_side': None,  # Will be enriched later
                    'predicted_move': None,  # Will be enriched later
                    'language': 'en',  # Assuming English for GlobeNewswire
                    'title_en': title,
                    'content_en': full_content
                }
                
                items.append(news_item)
            
            logger.info(f"Parsed {len(items)} news items from feed: {feed_title}")
            return items
            
        except Exception as e:
            logger.error(f"Failed to parse RSS feed for {feed_title}: {e}")
            return []
    
    def extract_company_from_text(self, text: str) -> Optional[str]:
        """Extract company name from text using OpenAI."""
        try:
            company = extract_issuer(text)
            return company if company and company != "N/A" else None
        except Exception as e:
            logger.warning(f"Failed to extract company from text: {e}")
            return None
    
    def download_news_from_feeds(self, max_feeds: Optional[int] = None, start_position: int = 0) -> List[Dict[str, Any]]:
        """Download news from all RSS feeds."""
        feeds = self.load_rss_feeds()
        
        # Apply start position
        if start_position > 0:
            feeds = feeds[start_position:]
            logger.info(f"Starting from position {start_position}")
        
        if max_feeds:
            feeds = feeds[:max_feeds]
            logger.info(f"Limiting to {max_feeds} feeds for testing")
        
        all_news_items = []
        successful_feeds = 0
        failed_feeds = 0
        
        for i, feed_info in enumerate(feeds, 1):
            feed_title = feed_info.get('title', f'Feed {i}')
            rss_url = feed_info.get('feeds', {}).get('RSS', {}).get('url', '')
            
            if not rss_url:
                logger.warning(f"No RSS URL found for feed: {feed_title}")
                failed_feeds += 1
                continue
            
            logger.info(f"Processing feed {i}/{len(feeds)}: {feed_title}")
            
            # Fetch RSS content
            rss_content = self.fetch_rss_feed(rss_url)
            if not rss_content:
                failed_feeds += 1
                continue
            
            # Parse RSS content
            news_items = self.parse_rss_feed(rss_content, feed_title)
            all_news_items.extend(news_items)
            successful_feeds += 1
            
            # Add a small delay to be respectful to the server
            import time
            time.sleep(0.5)
        
        logger.info(f"Downloaded {len(all_news_items)} news items from {successful_feeds} feeds ({failed_feeds} failed)")
        return all_news_items
    
    def save_to_database(self, news_items: List[Dict[str, Any]]) -> int:
        """Save news items to database."""
        if not news_items:
            logger.warning("No news items to save")
            return 0
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(news_items)
            
            # Map to database objects
            db_objects = map_to_db(df, 'globenewswire')
            
            # Add to database
            added_count, duplicate_count = add_news_items(db_objects, check_uniqueness=True)
            
            logger.info(f"Successfully saved {added_count} news items to database ({duplicate_count} duplicates skipped)")
            return added_count
            
        except Exception as e:
            logger.error(f"Failed to save news items to database: {e}")
            raise
    
    def run_download(self, max_feeds: Optional[int] = None, start_position: int = 0) -> int:
        """Main method to run the news download process."""
        try:
            logger.info("Starting GlobeNewswire news download process")
            
            # Download news from feeds
            news_items = self.download_news_from_feeds(max_feeds, start_position)
            
            if not news_items:
                logger.warning("No news items downloaded")
                return 0
            
            # Save to database
            saved_count = self.save_to_database(news_items)
            
            logger.info(f"News download process completed. Saved {saved_count} items to database")
            return saved_count
            
        except Exception as e:
            logger.error(f"News download process failed: {e}")
            raise


def main():
    """Main function to run the news downloader."""
    try:
        # Create downloader instance
        downloader = GlobeNewswireNewsDownloader()
        
        # Run download for all 257 feeds
        saved_count = downloader.run_download(max_feeds=None, start_position=0)
        
        print(f"\n✅ Successfully downloaded and saved {saved_count} news items")
        print("📁 Data saved to database for further processing")
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
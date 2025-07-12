#!/usr/bin/env python3
"""
Test script to download from a single RSS feed URL and validate results.
"""

import sys
import os
import requests
import feedparser
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai.openai_util import extract_ticker, extract_issuer
from utils.db.news_db_util import News, add_news_items, map_to_db, get_news_df
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

def test_single_rss_feed():
    """Test downloading from a single RSS feed URL."""
    
    # Test URL - Semiconductors industry feed
    test_url = "https://www.globenewswire.com/RssFeed/industry/9576-Semiconductors/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Semiconductors"
    
    print(f"🔗 Testing RSS feed: {test_url}")
    print("=" * 80)
    
    # Create session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/rss+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    
    # Fetch RSS content
    print("📥 Fetching RSS feed...")
    try:
        response = session.get(test_url, timeout=30)
        response.raise_for_status()
        rss_content = response.text
        print(f"✅ Successfully fetched RSS content ({len(rss_content)} characters)")
    except Exception as e:
        print(f"❌ Failed to fetch RSS feed: {e}")
        return
    
    # Parse RSS content
    print("\n📋 Parsing RSS feed...")
    try:
        feed = feedparser.parse(rss_content)
        print(f"✅ Parsed {len(feed.entries)} entries")
        
        # Process first 5 entries for testing
        test_entries = feed.entries[:5]
        news_items = []
        
        for i, entry in enumerate(test_entries, 1):
            print(f"\n--- Processing Entry {i} ---")
            
            # Extract basic information
            title = entry.get('title', '')
            link = entry.get('link', '')
            published = entry.get('published', '')
            summary = entry.get('summary', '')
            content = entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''
            
            print(f"Title: {title[:100]}...")
            print(f"Link: {link}")
            print(f"Published: {published}")
            
            # Use content if available, otherwise use summary
            full_content = content if content else summary
            
            # Parse published date
            published_date = None
            if published:
                try:
                    parsed_date = feedparser.parse(f"<rss><item><pubDate>{published}</pubDate></item></rss>")
                    if parsed_date.entries and parsed_date.entries[0].get('published_parsed'):
                        timestamp = time.mktime(parsed_date.entries[0].published_parsed)
                        published_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        print(f"Parsed date: {published_date}")
                except Exception as e:
                    print(f"Failed to parse date: {e}")
            
            # Extract company name from title or content
            print("🔍 Extracting company name...")
            company = None
            try:
                company = extract_issuer(title + " " + full_content)
                if company and company != "N/A":
                    print(f"✅ Extracted company: {company}")
                else:
                    print("❌ No company extracted")
            except Exception as e:
                print(f"❌ Failed to extract company: {e}")
            
            # Extract ticker using OpenAI
            ticker = None
            if company:
                print("🔍 Extracting ticker...")
                try:
                    ticker = extract_ticker(company)
                    if ticker:
                        print(f"✅ Extracted ticker: {ticker}")
                    else:
                        print("❌ No ticker extracted")
                except Exception as e:
                    print(f"❌ Failed to extract ticker: {e}")
            
            # Create news item
            news_item = {
                'title': title,
                'link': link,
                'company': company,
                'published_date': published_date,
                'content': full_content,
                'reason': '',  # Will be enriched later
                'industry': '',  # Will be enriched later
                'publisher_topic': 'Semiconductors (Test)',
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
            
            news_items.append(news_item)
            print(f"✅ Created news item {i}")
            
            # Add delay to be respectful to OpenAI API
            time.sleep(1)
        
        print(f"\n📊 Created {len(news_items)} news items for testing")
        
        # Save to database
        print("\n💾 Saving to database...")
        try:
            # Convert to DataFrame
            df = pd.DataFrame(news_items)
            
            # Map to database objects
            db_objects = map_to_db(df, 'globenewswire')
            
            # Add to database
            added_count, duplicate_count = add_news_items(db_objects, check_uniqueness=True)
            
            print(f"✅ Successfully saved {added_count} news items to database ({duplicate_count} duplicates skipped)")
            
            return added_count
            
        except Exception as e:
            print(f"❌ Failed to save to database: {e}")
            return 0
            
    except Exception as e:
        print(f"❌ Failed to parse RSS feed: {e}")
        return 0

def show_database_results():
    """Query and display the latest news items from the database."""
    print("\n" + "=" * 80)
    print("📊 DATABASE RESULTS - Latest News Items")
    print("=" * 80)
    
    try:
        # Get latest news items
        df = get_news_df(publisher='globenewswire')
        
        if df.empty:
            print("❌ No news items found in database")
            return
        
        # Sort by published_date (newest first) and show latest 10
        df_sorted = df.sort_values('published_date', ascending=False).head(10)
        
        print(f"📈 Found {len(df)} total news items for globenewswire")
        print(f"📋 Showing latest {len(df_sorted)} items:\n")
        
        for idx, row in df_sorted.iterrows():
            print(f"🔸 News ID: {row['news_id']}")
            print(f"   Title: {row['title'][:100]}...")
            print(f"   Company: {row['company'] or 'N/A'}")
            print(f"   Ticker: {row['ticker'] or 'N/A'}")
            print(f"   Published: {row['published_date']}")
            print(f"   Publisher Topic: {row['publisher_topic']}")
            print(f"   Status: {row['status']}")
            print(f"   Link: {row['link']}")
            print("-" * 60)
        
        # Show summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        print(f"   Total items: {len(df)}")
        print(f"   Items with company: {df['company'].notna().sum()}")
        print(f"   Items with ticker: {df['ticker'].notna().sum()}")
        print(f"   Items with content: {df['content'].notna().sum()}")
        print(f"   Publisher topics: {df['publisher_topic'].nunique()}")
        
    except Exception as e:
        print(f"❌ Failed to query database: {e}")

def main():
    """Main function to run the test."""
    print("🧪 Testing Single RSS Feed Download and Database Storage")
    print("=" * 80)
    
    # Test single feed download
    added_count = test_single_rss_feed()
    
    if added_count > 0:
        # Show database results
        show_database_results()
    else:
        print("\n❌ No items were added to database, skipping results display")

if __name__ == "__main__":
    main() 
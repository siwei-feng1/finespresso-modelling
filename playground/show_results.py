#!/usr/bin/env python3
"""
Simple script to show database results for the test.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db.news_db_util import get_news_df

def show_results():
    """Show the latest news items from the database."""
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
            print(f"   Link: {row['link']}")
            print("-" * 60)
        
        # Show summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        print(f"   Total items: {len(df)}")
        print(f"   Items with company: {df['company'].notna().sum()}")
        print(f"   Items with ticker: {df['ticker'].notna().sum()}")
        print(f"   Items with content: {df['content'].notna().sum()}")
        print(f"   Publisher topics: {df['publisher_topic'].nunique()}")
        
        # Show unique publisher topics
        print(f"\n📋 Publisher Topics:")
        topics = df['publisher_topic'].value_counts()
        for topic, count in topics.head(10).items():
            print(f"   {topic}: {count} items")
        
    except Exception as e:
        print(f"❌ Failed to query database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_results() 
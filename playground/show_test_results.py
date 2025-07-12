#!/usr/bin/env python3
"""
Show the specific test results from the Semiconductors feed.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db.news_db_util import get_news_df

def show_test_results():
    """Show the specific test results from the Semiconductors feed."""
    print("🧪 TEST RESULTS - Semiconductors Feed Items")
    print("=" * 80)
    
    try:
        # Get news items
        df = get_news_df(publisher='globenewswire')
        
        if df.empty:
            print("❌ No news items found in database")
            return
        
        # Filter for our test items (Semiconductors topic)
        test_items = df[df['publisher_topic'] == 'Semiconductors (Test)']
        
        if test_items.empty:
            print("❌ No test items found (Semiconductors (Test) topic)")
            return
        
        print(f"📊 Found {len(test_items)} test items from Semiconductors feed")
        print(f"📋 Showing all test items:\n")
        
        for idx, row in test_items.iterrows():
            print(f"🔸 News ID: {row['news_id']}")
            print(f"   Title: {row['title']}")
            print(f"   Company: {row['company'] or 'N/A'}")
            print(f"   Ticker: {row['ticker'] or 'N/A'}")
            print(f"   Published: {row['published_date']}")
            print(f"   Publisher Topic: {row['publisher_topic']}")
            print(f"   Link: {row['link']}")
            print(f"   Content Length: {len(row['content']) if row['content'] else 0} characters")
            print("-" * 60)
        
        # Show summary of test results
        print("\n📊 TEST SUMMARY:")
        print(f"   Total test items: {len(test_items)}")
        print(f"   Items with company: {test_items['company'].notna().sum()}")
        print(f"   Items with ticker: {test_items['ticker'].notna().sum()}")
        
        # Show companies and tickers extracted
        print(f"\n🏢 COMPANIES EXTRACTED:")
        for idx, row in test_items.iterrows():
            if row['company']:
                print(f"   {row['company']} -> {row['ticker'] or 'No ticker'}")
        
    except Exception as e:
        print(f"❌ Failed to query database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_test_results() 
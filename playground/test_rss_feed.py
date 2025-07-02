#!/usr/bin/env python3
"""
Test script to verify RSS feed functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.backfill_price_moves import RSSBackfillProcessor

def test_rss_feed():
    """Test RSS feed fetching and parsing."""
    processor = RSSBackfillProcessor()
    
    # Test RSS URL
    rss_url = "https://www.globenewswire.com/RssFeed/industry/9576-Semiconductors/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Semiconductors"
    
    print("Testing RSS feed functionality...")
    
    # Test fetching
    print("1. Fetching RSS feed...")
    rss_content = processor.fetch_rss_feed(rss_url)
    if rss_content:
        print("✓ RSS feed fetched successfully")
        print(f"Content length: {len(rss_content)} characters")
    else:
        print("✗ Failed to fetch RSS feed")
        return
    
    # Test parsing
    print("\n2. Parsing RSS feed...")
    items = processor.parse_rss_feed(rss_content)
    if items:
        print(f"✓ Parsed {len(items)} items successfully")
        
        # Show sample item
        if items:
            sample = items[0]
            print("\nSample item:")
            for key, value in sample.items():
                if key != 'categories':  # Skip categories for brevity
                    print(f"  {key}: {value}")
            
            # Show categories separately
            if 'categories' in sample:
                print(f"  categories: {sample['categories']}")
    else:
        print("✗ Failed to parse RSS feed")
        return
    
    print("\n✓ RSS feed test completed successfully!")

if __name__ == "__main__":
    test_rss_feed() 
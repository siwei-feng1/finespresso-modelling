#!/usr/bin/env python3
"""
Example script showing how to use the RSS backfill processor with different feeds.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.backfill_price_moves import RSSBackfillProcessor

def main():
    """Example usage of the RSS backfill processor."""
    
    # Initialize processor
    processor = RSSBackfillProcessor()
    
    # Example RSS feeds (you can add more)
    rss_feeds = {
        "Semiconductors": "https://www.globenewswire.com/RssFeed/industry/9576-Semiconductors/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Semiconductors",
        "Technology": "https://www.globenewswire.com/RssFeed/industry/9577-Technology/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Technology",
        "Healthcare": "https://www.globenewswire.com/RssFeed/industry/9578-Healthcare/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Healthcare"
    }
    
    print("RSS Backfill Processor Example")
    print("=" * 40)
    
    # Process each feed
    for industry, rss_url in rss_feeds.items():
        print(f"\nProcessing {industry} RSS feed...")
        print("-" * 30)
        
        try:
            # Process the RSS feed
            df = processor.process_rss_feed(rss_url)
            
            if df.empty:
                print(f"No data found for {industry}")
                continue
            
            # Save results
            filepath = processor.save_results(df, output_dir=f'data/{industry.lower()}')
            
            # Print summary
            print(f"✓ Processed {len(df)} items for {industry}")
            print(f"✓ Saved to: {filepath}")
            
            # Show sample results
            if not df.empty:
                print("\nSample results:")
                sample_df = df[['company', 'ticker', 'published_date', 'price_change_percentage', 'actual_side']].head(3)
                print(sample_df.to_string(index=False))
            
        except Exception as e:
            print(f"✗ Error processing {industry}: {e}")
    
    print("\n" + "=" * 40)
    print("Example completed!")

if __name__ == "__main__":
    main() 
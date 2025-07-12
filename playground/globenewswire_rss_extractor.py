#!/usr/bin/env python3
"""
GlobeNewswire RSS Feed Extractor

This script scrapes the GlobeNewswire RSS feeds page and extracts all available
RSS feed information, saving it to a YAML file for further processing.
"""

import requests
from bs4 import BeautifulSoup
import yaml
import os
from typing import List, Dict, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlobeNewswireRSSExtractor:
    """Extracts RSS feed information from GlobeNewswire RSS feeds page."""
    
    def __init__(self, base_url: str = "https://www.globenewswire.com/rss/list"):
        self.base_url = base_url
        self.session = requests.Session()
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def fetch_page(self) -> str:
        """Fetch the RSS feeds page content."""
        try:
            logger.info(f"Fetching RSS feeds from: {self.base_url}")
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch page: {e}")
            raise
    
    def parse_rss_feeds(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse the HTML content and extract RSS feed information."""
        soup = BeautifulSoup(html_content, 'html.parser')
        feeds = []
        
        # Find all RSS feed list items
        feed_items = soup.find_all('li')
        
        for item in feed_items:
            # Look for items with rss-title class
            title_div = item.find('div', class_='rss-title')
            if not title_div:
                continue
            
            title = title_div.get_text(strip=True)
            if not title:
                continue
            
            # Extract feed links
            links_div = item.find('div', class_='rss-links')
            if not links_div:
                continue
            
            feed_links = {}
            for link in links_div.find_all('a', class_='feed-list-button'):
                href = link.get('href', '')
                link_title = link.get('title', '')
                
                # Only extract RSS feeds (skip ATOM and JavaScript Widget)
                if 'rss-rss' in link.get('class', []) and href:
                    # Make relative URLs absolute
                    if href.startswith('/'):
                        href = f"https://www.globenewswire.com{href}"
                    
                    feed_links['RSS'] = {
                        'url': href,
                        'title': link_title
                    }
            
            if feed_links:
                # Only include feeds that are categorized by industry or subject code
                # Skip geographic feeds (countries and states/provinces)
                rss_url = feed_links.get('RSS', {}).get('url', '')
                if rss_url and ('/industry/' in rss_url or '/subjectcode/' in rss_url or '/orgclass/' in rss_url):
                    feeds.append({
                        'title': title,
                        'feeds': feed_links,
                        'extracted_at': datetime.now().isoformat()
                    })
        
        logger.info(f"Extracted {len(feeds)} RSS feed categories")
        return feeds
    
    def save_to_yaml(self, feeds: List[Dict[str, Any]], output_path: str) -> None:
        """Save the extracted feeds to a YAML file."""
        try:
            # Ensure the data directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            data = {
                'source': 'GlobeNewswire RSS Feeds',
                'base_url': self.base_url,
                'extracted_at': datetime.now().isoformat(),
                'total_categories': len(feeds),
                'feeds': feeds
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2, allow_unicode=True)
            
            logger.info(f"Successfully saved {len(feeds)} feed categories to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save YAML file: {e}")
            raise
    
    def extract_and_save(self, output_path: str = "data/globenewswire_rss.yaml") -> List[Dict[str, Any]]:
        """Main method to extract RSS feeds and save to YAML."""
        try:
            # Fetch the page
            html_content = self.fetch_page()
            
            # Parse the feeds
            feeds = self.parse_rss_feeds(html_content)
            
            # Save to YAML
            self.save_to_yaml(feeds, output_path)
            
            return feeds
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise


def main():
    """Main function to run the RSS extractor."""
    try:
        extractor = GlobeNewswireRSSExtractor()
        feeds = extractor.extract_and_save()
        
        print(f"\n✅ Successfully extracted {len(feeds)} RSS feed categories")
        print("📁 Data saved to: data/globenewswire_rss.yaml")
        
        # Print a summary of extracted feeds
        print("\n📋 Extracted RSS Feed Categories:")
        for i, feed in enumerate(feeds, 1):
            print(f"  {i}. {feed['title']}")
            if 'RSS' in feed['feeds']:
                print(f"     - RSS: {feed['feeds']['RSS']['url']}")
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

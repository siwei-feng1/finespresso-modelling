# GlobeNewswire RSS Feed Extractor

A BeautifulSoup-based RSS feed extractor that scrapes GlobeNewswire's RSS feeds page and extracts industry and subject-specific news feeds for financial analysis and market monitoring.

## Overview

This tool extracts RSS feeds from [GlobeNewswire's RSS feeds page](https://www.globenewswire.com/rss/list) and saves them in a structured YAML format. It filters feeds to include only those categorized by industry or subject code, excluding geographic feeds.

## Features

- **RSS-Only Extraction**: Only extracts RSS feeds (no ATOM or JavaScript Widget feeds)
- **Industry & Subject Filtering**: Focuses on business-relevant feeds categorized by:
  - Industry sectors (`/industry/` URLs)
  - Subject codes (`/subjectcode/` URLs) 
  - Organization classes (`/orgclass/` URLs)
- **Structured Output**: Saves data in well-organized YAML format
- **Comprehensive Coverage**: Extracts 257 RSS feed categories
- **Error Handling**: Robust error handling and logging

## Requirements

- Python 3.8+
- BeautifulSoup4
- Requests
- PyYAML

All dependencies are already included in the project's `requirements.txt`.

## Usage

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the extractor
python playground/globenewswire_rss_extractor.py
```

### Output

The script generates:
- **Console output**: Summary of extracted feeds with categories and URLs
- **YAML file**: `data/globenewswire_rss.yaml` with structured feed data

## Feed Categories

### Subject Code Feeds (81 categories)

**Business Events:**
- Earnings Releases and Operating Results
- Mergers and Acquisitions
- Clinical Study
- Initial Public Offerings
- Management Changes

**Financial Topics:**
- Dividend Reports and Estimates
- Analyst Recommendations
- Bond Market News
- Equity Market Information
- Trading Information

**Corporate Actions:**
- Company Regulatory Filings
- Directors And Officers
- Proxy Statements And Analysis
- Major Shareholder Announcements
- Changes In Company's Own Shares

**Market Information:**
- Stock Market News
- Technical Analysis
- Economic Research And Reports
- Market Research Reports

### Industry Feeds (176 categories)

**Basic Materials:**
- Mining (Gold, Copper, General Mining)
- Chemicals (Commodity, Specialty, Diversified)
- Metals (Iron & Steel, Aluminum, Nonferrous Metals)
- Forestry and Paper

**Consumer Goods:**
- Automobiles and Auto Parts
- Food Products and Beverages
- Clothing & Accessories
- Consumer Electronics
- Personal Products

**Consumer Services:**
- Airlines and Travel
- Retail (Apparel, Food, Specialty)
- Hotels and Lodging
- Media and Entertainment
- Restaurants & Bars

**Energy:**
- Oil & Gas (Exploration, Integrated, Equipment)
- Alternative Fuels
- Renewable Energy Equipment
- Pipelines
- Coal

**Financials:**
- Banks and Financial Services
- Insurance (Life, Property & Casualty, Reinsurance)
- Investment Services
- Asset Managers & Custodians
- REITs (various types)

**Healthcare:**
- Biotechnology
- Pharmaceuticals
- Medical Equipment and Supplies
- Health Care Providers
- Cannabis Producers

**Industrials:**
- Aerospace and Defense
- Construction and Building Materials
- Machinery (Agricultural, Industrial, Specialty)
- Transportation Services
- Business Support Services

**Real Estate:**
- Various REIT types (Residential, Commercial, Healthcare, etc.)
- Real Estate Services
- Real Estate Holding & Development

**Technology:**
- Software and Computer Services
- Computer Hardware
- Semiconductors
- Internet and Telecommunications
- Electronic Equipment

**Utilities:**
- Conventional and Alternative Electricity
- Gas Distribution
- Water
- Multiutilities

## Output Format

The YAML file contains:

```yaml
source: GlobeNewswire RSS Feeds
base_url: https://www.globenewswire.com/rss/list
extracted_at: '2025-07-12T13:37:37.120000'
total_categories: 257
feeds:
  - title: "Public Companies"
    extracted_at: '2025-07-12T13:37:36.794000'
    feeds:
      RSS:
        title: RSS Feed
        url: https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire - News about Public Companies
  - title: "Earnings Releases and Operating Results"
    extracted_at: '2025-07-12T13:37:36.794000'
    feeds:
      RSS:
        title: RSS Feed
        url: https://www.globenewswire.com/RssFeed/subjectcode/13-Earnings Releases and Operating Results/feedTitle/GlobeNewswire - Earnings Releases and Operating Results
  # ... more feeds
```

## Use Cases

### Financial Analysis
- Monitor earnings releases and financial results
- Track mergers and acquisitions activity
- Follow analyst recommendations and market research

### Industry Monitoring
- Sector-specific news monitoring
- Competitive intelligence
- Market trend analysis

### News Aggregation
- Automated news collection
- Real-time market updates
- Custom news feeds for trading systems

### Research and Reporting
- Market research data collection
- Industry analysis reports
- Regulatory compliance monitoring

## Technical Details

### URL Structure
The extracted RSS feeds follow these URL patterns:
- Industry: `/RssFeed/industry/{code}-{name}/feedTitle/GlobeNewswire - Industry News on {name}`
- Subject: `/RssFeed/subjectcode/{code}-{name}/feedTitle/GlobeNewswire - {name}`
- Organization: `/RssFeed/orgclass/{code}/feedTitle/GlobeNewswire - News about {name}`

### Filtering Logic
The script filters feeds based on URL patterns:
```python
if rss_url and ('/industry/' in rss_url or '/subjectcode/' in rss_url or '/orgclass/' in rss_url):
    # Include feed
```

### Error Handling
- Network timeout handling (30 seconds)
- HTTP error handling
- YAML write error handling
- Comprehensive logging

## Maintenance

### Updating Feeds
Run the script periodically to get the latest feed URLs:
```bash
python playground/globenewswire_rss_extractor.py
```

### Monitoring Changes
The script logs extraction details and can be used to track changes in available feeds over time.

## Limitations

- Only extracts RSS feeds (no ATOM or JavaScript Widget feeds)
- Excludes geographic feeds (countries and states/provinces)
- Depends on GlobeNewswire's website structure
- Requires internet connection to fetch feeds

## Future Enhancements

Potential improvements:
- Add support for ATOM feeds
- Include geographic filtering options
- Add feed validation
- Implement incremental updates
- Add feed content parsing
- Create feed monitoring dashboard

## License

This tool is part of the FineSpresso Modelling project and follows the project's licensing terms.

## References

- [GlobeNewswire RSS Feeds](https://www.globenewswire.com/rss/list)
- [RSS 2.0 Specification](http://www.rssboard.org/rss-specification)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)
- [PyYAML Documentation](https://pyyaml.org/) 
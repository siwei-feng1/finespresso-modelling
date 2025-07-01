# RSS Backfill Script

This script processes RSS feeds from GlobeNewswire, extracts company information, gets ticker symbols, and calculates price moves based on publication time relative to market hours.

## Features

1. **RSS Feed Processing**: Fetches and parses RSS feeds from GlobeNewswire
2. **Company Extraction**: Extracts company names from the `dc:contributor` field
3. **Ticker Symbol Extraction**: Uses both RSS categories and OpenAI to extract ticker symbols
4. **Price Move Calculation**: Calculates price moves based on publication time:
   - **Market Hours (9:30 AM ≤ published_date < 4:00 PM)**: `price_move = price(t, close) - price(t, open)`
   - **Pre-Market (published_date < 9:30 AM)**: `price_move = price(t, open) - price(t-1, close)`
   - **After Hours (published_date > 4:00 PM)**: `price_move = price(t, close) - price(t+1, open)`
5. **Data Export**: Saves results to timestamped CSV files

## Usage

### Basic Usage

```bash
python tasks/backfill_price_moves.py
```

### Test RSS Feed

```bash
python tasks/test_rss_feed.py
```

## Configuration

### Environment Variables

Make sure you have the following environment variables set in your `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### RSS Feed URL

The default RSS feed URL is:
```
https://www.globenewswire.com/RssFeed/industry/9576-Semiconductors/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Semiconductors
```

You can modify this in the `main()` function of `backfill_price_moves.py`.

## Output

The script generates a CSV file in the `data/` directory with the following columns:

- `news_id`: Unique identifier for the news item
- `title`: News title
- `description`: News description
- `link`: Link to the full news article
- `company`: Company name extracted from contributor field
- `ticker`: Ticker symbol
- `published_date`: Publication date and time
- `categories`: RSS categories (including stock information)
- `publisher`: News publisher
- `language`: News language
- `begin_price`: Starting price for the period
- `end_price`: Ending price for the period
- `index_begin_price`: Starting price of market index (SPY)
- `index_end_price`: Ending price of market index (SPY)
- `price_change`: Absolute price change
- `price_change_percentage`: Percentage price change
- `index_price_change`: Absolute index price change
- `index_price_change_percentage`: Percentage index price change
- `daily_alpha`: Alpha (excess return vs index)
- `actual_side`: Direction of price move (UP/DOWN)
- `volume`: Trading volume
- `market`: Market timing (regular_market/pre_market/after_market)

## Price Move Calculation Rules

The script calculates price movements based on the publication time relative to market hours:

### Market Hours (9:30 AM ≤ published_date < 4:00 PM)
- **Calculation**: `price_move = price(t, close) - price(t, open)`
- **Measures**: Intraday price movement from market open to close on the same trading day

### Pre-Market (published_date < 9:30 AM)
- **Calculation**: `price_move = price(t, open) - price(t-1, close)`
- **Measures**: Overnight gap from previous day's close to current day's open

### After Hours (published_date > 4:00 PM)
- **Calculation**: `price_move = price(t, close) - price(t+1, open)`
- **Measures**: Overnight gap from current day's close to next day's open

## Dependencies

- `pandas`: Data manipulation
- `requests`: HTTP requests for RSS feeds
- `yfinance`: Yahoo Finance data
- `openai`: OpenAI API for ticker extraction
- `python-dotenv`: Environment variable management

## Error Handling

The script includes comprehensive error handling for:
- RSS feed fetching failures
- XML parsing errors
- Missing price data
- Invalid ticker symbols
- API rate limiting

## Logging

Logs are written to:
- Console output
- `logs/backfill.log` file

## Example Output

```
2025-07-01 17:32:46 - backfill_price_moves - INFO - Starting RSS backfill process...
2025-07-01 17:32:46 - backfill_price_moves - INFO - Fetching RSS feed from: https://www.globenewswire.com/RssFeed/industry/9576-Semiconductors/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Semiconductors
2025-07-01 17:32:47 - backfill_price_moves - INFO - Parsed 20 items from RSS feed
2025-07-01 17:32:47 - backfill_price_moves - INFO - Extracting ticker symbols...
2025-07-01 17:32:47 - backfill_price_moves - INFO - Found ticker ASM from RSS category for ASM International NV
2025-07-01 17:32:47 - backfill_price_moves - INFO - Found tickers for 5 out of 20 items
2025-07-01 17:32:47 - backfill_price_moves - INFO - Calculating price moves...
2025-07-01 17:32:47 - backfill_price_moves - INFO - Getting price data for ASM on 2025-06-30, market: regular_market
2025-07-01 17:32:48 - backfill_price_moves - INFO - Successfully processed 5 price moves
2025-07-01 17:32:48 - backfill_price_moves - INFO - Saved 5 records to data/price_moves_20250701_173257.csv
2025-07-01 17:32:48 - backfill_price_moves - INFO - Backfill process completed!
```

## Troubleshooting

### Common Issues

1. **No price data available**: Some tickers may be delisted or have no data
2. **OpenAI API errors**: Check your API key and rate limits
3. **RSS feed errors**: Verify the RSS URL is accessible
4. **Date parsing errors**: The script handles multiple date formats automatically

### Rate Limiting

The script includes a 0.1-second delay between API calls to avoid rate limiting. You can adjust this in the `process_rss_feed` method. 
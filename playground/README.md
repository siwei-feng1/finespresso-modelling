# Database Backfill Script

This script processes existing news from the database, extracts ticker symbols, and calculates price moves based on publication time relative to market hours.

## Features

1. **Database Integration**: Gets all existing news from the database using `news_db_util`
2. **Ticker Processing**: 
   - For news with existing `yf_ticker`: calculates price moves directly
   - For news without tickers: extracts ticker symbols using OpenAI and updates database
3. **Batch Processing**: Processes news items in configurable batches (default: 100) for memory efficiency
4. **Price Move Calculation**: Calculates price moves based on publication time:
   - **Market Hours (9:30 AM ≤ published_date < 4:00 PM)**: `price_move = price(t, close) - price(t, open)`
   - **Pre-Market (published_date < 9:30 AM)**: `price_move = price(t, open) - price(t-1, close)`
   - **After Hours (published_date > 4:00 PM)**: `price_move = price(t, close) - price(t+1, open)`
5. **Dual Storage**: Saves results to both database and timestamped CSV files
6. **Progress Tracking**: Comprehensive logging with batch progress and statistics

## Usage

### Basic Usage

```bash
python playground/backfill_price_moves.py
```

### Configuration

You can modify the batch size in the `main()` function:

```python
# Change batch size for processing
batch_size = 100  # Default: 100 items per batch
processor = DatabaseBackfillProcessor(batch_size=batch_size)
```

### Batch Processing

The script processes news items in batches to:
- Reduce memory usage
- Provide regular progress updates
- Write to database after each batch
- Enable error recovery (if one batch fails, others continue)

### Logging

The script provides detailed logging including:
- Batch progress (e.g., "Processing batch 1/5 (items 1-100)")
- Database write confirmations
- Statistics after each batch
- Final summary with all metrics

## Configuration

### Environment Variables

Make sure you have the following environment variables set in your `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
DATABASE_URL=your_database_connection_string
```

### Database Requirements

The script requires:
- Access to the `news` table with existing news data
- Access to the `price_moves` table for storing results
- Proper database permissions for read/write operations

### Market Index

The script uses **SPY** (S&P 500 ETF) as the market index for calculating alpha and relative performance.

## Output

The script generates results in two formats:

### 1. Database Storage
Price moves are stored in the `price_moves` table with the following structure:
- `news_id`: Links to the original news item
- `ticker`: Ticker symbol
- `published_date`: Publication date and time
- `begin_price`, `end_price`: Price range for the period
- `index_begin_price`, `index_end_price`: SPY index prices
- `price_change_percentage`: Percentage price change
- `daily_alpha`: Excess return vs market index
- `actual_side`: Direction of price move (UP/DOWN)
- `market`: Market timing (regular_market/pre_market/after_market)
- `volume`: Trading volume
- `price_source`: Source of price data (yfinance)

### 2. CSV Export
A timestamped CSV file is generated in the `data/` directory with columns:
- `news_id`: Unique identifier for the news item
- `title`: News title
- `description`: News content
- `link`: Link to the full news article
- `company`: Company name
- `ticker`: Ticker symbol
- `published_date`: Publication date and time
- `publisher`: News publisher
- `language`: News language
- `begin_price`, `end_price`: Price range for the period
- `index_begin_price`, `index_end_price`: SPY index prices
- `price_change_percentage`: Percentage price change
- `daily_alpha`: Alpha (excess return vs index)
- `actual_side`: Direction of price move (UP/DOWN)
- `volume`: Trading volume
- `market`: Market timing

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
- `yfinance`: Yahoo Finance data
- `openai`: OpenAI API for ticker extraction
- `python-dotenv`: Environment variable management
- `sqlalchemy`: Database operations
- `psycopg2`: PostgreSQL database adapter

## Error Handling

The script includes comprehensive error handling for:
- Database connection failures
- Missing price data from yfinance
- Invalid ticker symbols
- OpenAI API rate limiting and errors
- Batch processing failures (continues with next batch)
- Memory management for large datasets

## Logging

Logs are written to:
- Console output with real-time progress
- `logs/backfill_db.log` file for detailed logging

### Log Levels
- **INFO**: Batch progress, statistics, completion status
- **WARNING**: Missing data, failed operations
- **ERROR**: Critical failures, exceptions
- **DEBUG**: Detailed processing information

## Example Output

```
2025-07-01 17:32:46 - DatabaseBackfillProcessor - INFO - Starting database backfill process...
2025-07-01 17:32:46 - DatabaseBackfillProcessor - INFO - Fetching all news from database...
2025-07-01 17:32:47 - DatabaseBackfillProcessor - INFO - Retrieved 1500 news items from database
2025-07-01 17:32:47 - DatabaseBackfillProcessor - INFO - News with tickers: 800
2025-07-01 17:32:47 - DatabaseBackfillProcessor - INFO - News without tickers: 700
2025-07-01 17:32:47 - DatabaseBackfillProcessor - INFO - Processing 800 news items with existing tickers in batches of 100...
2025-07-01 17:32:47 - DatabaseBackfillProcessor - INFO - Processing batch 1/8 (items 1-100)
2025-07-01 17:32:48 - DatabaseBackfillProcessor - INFO - Batch 1/8 completed: 85 price moves processed
2025-07-01 17:32:48 - DatabaseBackfillProcessor - INFO - Processing batch 2/8 (items 101-200)
2025-07-01 17:32:49 - DatabaseBackfillProcessor - INFO - Batch 2/8 completed: 92 price moves processed
...
2025-07-01 17:33:15 - DatabaseBackfillProcessor - INFO - Processing 700 news items without tickers in batches of 100...
2025-07-01 17:33:15 - DatabaseBackfillProcessor - INFO - Processing batch 1/7 (items 1-100)
2025-07-01 17:33:16 - DatabaseBackfillProcessor - INFO - Extracted ticker AAPL for company Apple Inc
2025-07-01 17:33:16 - DatabaseBackfillProcessor - INFO - Updated database with 45 extracted tickers from batch 1
2025-07-01 17:33:16 - DatabaseBackfillProcessor - INFO - Batch 1/7 completed: 45 tickers extracted
...
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - ============================================================
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - BACKFILL PROCESS STATISTICS
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - ============================================================
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - Total news items processed: 1500
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - News with existing tickers: 800
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - News without tickers: 700
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - Tickers extracted using OpenAI: 320
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - Price moves calculated: 1120
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - Price moves stored to database: 1120
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - Price moves stored to CSV: 1120
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - Errors encountered: 5
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - ============================================================
2025-07-01 17:33:45 - DatabaseBackfillProcessor - INFO - Database backfill process completed!
```

## Troubleshooting

### Common Issues

1. **Database connection errors**: Check your `DATABASE_URL` and database permissions
2. **No price data available**: Some tickers may be delisted or have no data from yfinance
3. **OpenAI API errors**: Check your API key and rate limits
4. **Memory issues**: Reduce batch size for large datasets
5. **Missing tickers**: News without company names cannot have tickers extracted

### Performance Optimization

- **Batch size**: Adjust based on available memory and processing speed
- **Rate limiting**: Built-in delays prevent API throttling
- **Error recovery**: Failed batches don't stop the entire process
- **Database writes**: Each batch is committed immediately

### Monitoring

- Check `logs/backfill_db.log` for detailed error information
- Monitor database connection pool usage
- Track API rate limits for OpenAI calls
- Verify CSV file generation in `data/` directory 
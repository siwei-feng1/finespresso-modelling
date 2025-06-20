# Tests Directory

This directory contains test scripts and utilities for the FineSpresso Modelling project.

## Scripts

### download_data.py

A comprehensive data download script that extracts all news, instrument, and price move data from the database and saves them as CSV files in the `data/` directory.

#### Features

- Downloads all news data (`all_news.csv`)
- Downloads latest 1000 news items (`latest_news.csv`)
- Downloads all instrument data (`all_instruments.csv`)
- Downloads all price move data (`all_price_moves.csv`)
- Downloads recent price moves (last 30 days) (`recent_price_moves_30days.csv`)
- Creates summary statistics in JSON format
- Generates a comprehensive download report

#### Usage

```bash
# Run from the project root directory
python tests/download_data.py

# Or make it executable and run directly
chmod +x tests/download_data.py
./tests/download_data.py
```

#### Output Files

The script creates the following files in the `data/` directory:

**CSV Files:**
- `all_news.csv` - Complete news dataset
- `latest_news.csv` - Latest 1000 news items
- `all_instruments.csv` - Complete instrument dataset
- `all_price_moves.csv` - Complete price move dataset
- `recent_price_moves_30days.csv` - Recent price moves (last 30 days)

**JSON Summary Files:**
- `news_summary.json` - News data statistics
- `instrument_summary.json` - Instrument data statistics
- `price_move_summary.json` - Price move data statistics
- `download_report.json` - Comprehensive download report

**Log File:**
- `download_data.log` - Detailed execution log

#### Summary Statistics

The script generates summary statistics including:
- Total record counts
- Date ranges
- Publisher distributions
- Event type distributions
- Ticker distributions
- Price change statistics (for price moves)
- Asset class, exchange, country, and sector distributions (for instruments)

#### Requirements

- Database connection (configured via environment variables)
- All required Python packages from `requirements.txt`
- Write permissions to the `data/` directory

#### Error Handling

The script includes comprehensive error handling:
- Logs all operations to both file and console
- Continues processing even if one data type fails
- Provides detailed error messages and stack traces
- Creates a summary report even if some operations fail 
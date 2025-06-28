#!/usr/bin/env python3
"""
Command Line Backtester for FineSpresso Modelling

This utility runs backtests on news data using CSV files from the data directory.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import sys
from pathlib import Path
from utils.backtesting.backtest_util import run_backtest
from utils.logging.log_util import get_logger
import yaml
import uuid

logger = get_logger(__name__)

# Available publishers
AVAILABLE_PUBLISHERS = [
    "globenewswire_country_fi",
    "globenewswire_country_dk",
    "globenewswire_biotech",
    "globenewswire_country_no",
    "globenewswire_country_lt",
    "globenewswire_country_lv",
    "globenewswire_country_is",
    "baltics",
    "globenewswire_country_se",
    "globenewswire_country_ee",
    "omx",
    "euronext"
]

# Available events with accuracy scores
AVAILABLE_EVENTS = {
    "changes_in_companys_own_shares": 88.89,
    "business_contracts": 83.33,
    "patents": 83.33,
    "shares_issue": 81.82,
    "corporate_action": 81.82,
    "licensing_agreements": 80.00,
    "major_shareholder_announcements": 75.00,
    "financial_results": 73.08,
    "financing_agreements": 71.43,
    "clinical_study": 69.49,
    "dividend_reports_and_estimates": 66.67,
    "management_changes": 65.00,
    "partnerships": 63.64,
    "earnings_releases_and_operating_result": 61.54,
    "regulatory_filings": 61.54,
    "product_services_announcement": 60.00
}

def load_config(config_path="config/backtester_config.yaml"):
    """Load backtest configuration from YAML file"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config

def load_news_data(data_dir="data"):
    """Load news data from CSV files in the data directory"""
    news_file = Path(data_dir) / "all_news.csv"
    
    if not news_file.exists():
        logger.error(f"News file not found: {news_file}")
        return None
    
    try:
        logger.info(f"Loading news data from {news_file}")
        news_df = pd.read_csv(news_file)
        
        # Convert published_date to datetime
        if 'published_date' in news_df.columns:
            news_df['published_date'] = pd.to_datetime(news_df['published_date'])
        
        logger.info(f"Loaded {len(news_df)} news records")
        return news_df
        
    except Exception as e:
        logger.error(f"Error loading news data: {e}")
        return None

def filter_news_data(news_df, start_date=None, end_date=None, publishers=None, events=None):
    """Filter news data based on date range, publishers, and events"""
    filtered_df = news_df.copy()
    
    # Filter by date range
    if start_date:
        filtered_df = filtered_df[filtered_df['published_date'] >= start_date]
    
    if end_date:
        filtered_df = filtered_df[filtered_df['published_date'] <= end_date]
    
    # Filter by publishers
    if publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(publishers)]
    
    # Filter by events
    if events:
        filtered_df = filtered_df[filtered_df['event'].isin(events)]
    
    logger.info(f"Filtered to {len(filtered_df)} news records")
    return filtered_df

def print_metrics(metrics):
    """Print backtest metrics in a formatted way"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return:        {metrics['total_return']:.1f}%")
    print(f"Annualized Return:   {metrics['annualized_return']:.1f}%")
    print(f"Total PnL:           ${metrics['total_pnl']:,.0f}")
    print(f"Total Trades:        {metrics['total_trades']}")
    print(f"Win Rate:            {metrics['win_rate']:.1f}%")
    print(f"Average Trade PnL:   ${metrics.get('avg_trade_pnl', 0):,.0f}")
    print(f"Max Drawdown:        {metrics.get('max_drawdown', 0):.1f}%")
    print("="*60)

def save_results(trades_df, output_dir="data", timestamp=None):
    """Save backtest results to CSV file"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = Path(output_dir) / f"trades_{timestamp}.csv"
    output_path.parent.mkdir(exist_ok=True)
    
    trades_df.to_csv(output_path, index=False)
    print(f"Trade history saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Run backtests on news data using CSV files or DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtester.py --data-source db
  python backtester.py --data-source csv --data-dir data
  python backtester.py --config back_test_config.yaml
        """
    )
    
    # Date range arguments
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backtest (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date for backtest (YYYY-MM-DD format)"
    )
    
    # Trading parameters
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000,
        help="Initial capital in dollars (default: 10000)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=10,
        help="Position size as percentage of capital (default: 10)"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=1.0,
        help="Take profit percentage (default: 1.0)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.5,
        help="Stop loss percentage (default: 0.5)"
    )
    
    # Data filtering
    parser.add_argument(
        "--publisher",
        type=str,
        choices=AVAILABLE_PUBLISHERS,
        help="Filter by specific publisher"
    )
    parser.add_argument(
        "--events",
        nargs="+",
        choices=list(AVAILABLE_EVENTS.keys()),
        help="Filter by specific events"
    )
    
    # Data source
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing news data (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save results (default: reports)"
    )
    
    # Utility options
    parser.add_argument(
        "--list-publishers",
        action="store_true",
        help="List available publishers and exit"
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List available events with accuracy scores and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--save-trades",
        action="store_true",
        default=True,
        help="Save trade history to CSV (default: True)"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["db", "csv"],
        default="db",
        help="Source of news data: db (default) or csv"
    )
    # Always use config/backtester_config.yaml as config file
    config_path = "config/backtester_config.yaml"
    
    args = parser.parse_args()
    
    # Handle list options
    if args.list_publishers:
        print("Available publishers:")
        for publisher in AVAILABLE_PUBLISHERS:
            print(f"  - {publisher}")
        return
    
    if args.list_events:
        print("Available events (with accuracy scores):")
        for event, accuracy in sorted(AVAILABLE_EVENTS.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {event}: {accuracy}%")
        return
    
    # Load config (always from config/backtester_config.yaml)
    config = load_config(config_path)

    # Use config values unless overridden by CLI
    def get_param(param, default=None):
        return getattr(args, param) if getattr(args, param, None) is not None else config.get(param, default)

    start_date = get_param('start_date')
    end_date = get_param('end_date')
    initial_capital = get_param('initial_capital', 10000)
    position_size = get_param('position_size', 10)
    take_profit = get_param('take_profit', 1.0)
    stop_loss = get_param('stop_loss', 0.5)
    publisher = get_param('publisher')
    events = get_param('events')
    data_dir = get_param('data_dir', 'data')
    output_dir = get_param('output_dir', 'reports')
    data_source = get_param('data_source', 'db')

    # Parse dates if string
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if (end_date - start_date).days > 31:
            end_date = start_date + pd.Timedelta(days=31)
    elif start_date:
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(days=31)
    elif end_date:
        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.Timedelta(days=31)
    else:
        # Default to last month
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=31)
    print(f"Backtest date range: {start_date.date()} to {end_date.date()}")

    # Load news data
    if data_source == "csv":
        from utils.backtesting.backtest_price_util import load_news_data_csv, load_price_moves_csv
        news_df = load_news_data_csv(data_dir)
        price_moves_df = load_price_moves_csv(data_dir)
    else:
        from utils.backtesting.backtest_price_util import load_news_data_db, load_price_moves_db
        from utils.db.news_db_util import get_news_df_date_range
        publishers = [publisher] if publisher else None
        if not publishers:
            logger.warning("No publisher specified for DB source; loading all publishers.")
        news_df = get_news_df_date_range(
            publishers=publishers,
            start_date=start_date,
            end_date=end_date
        )
        if events:
            news_df = news_df[news_df['event'].isin(events)]
        price_moves_df = load_price_moves_db()

    # Ensure start_date and end_date are timezone-consistent with published_date
    if news_df is not None and 'published_date' in news_df.columns:
        if hasattr(news_df['published_date'], 'dt') and news_df['published_date'].dt.tz is not None:
            start_date = pd.to_datetime(start_date).tz_localize('UTC') if start_date.tzinfo is None else start_date
            end_date = pd.to_datetime(end_date).tz_localize('UTC') if end_date.tzinfo is None else end_date
        else:
            start_date = pd.to_datetime(start_date).tz_localize(None)
            end_date = pd.to_datetime(end_date).tz_localize(None)

    if news_df is None:
        print("No news data loaded.")
        return 1

    publishers = [publisher] if publisher else None
    filtered_news_df = filter_news_data(
        news_df=news_df,
        start_date=start_date,
        end_date=end_date,
        publishers=publishers,
        events=events
    )
    if filtered_news_df.empty:
        print("No news data found for the specified filters.")
        return 1

    print(f"Running backtest with {len(filtered_news_df)} news events...")
    print(f"Parameters: Capital=${initial_capital:,.0f}, Position={position_size}%, TP={take_profit}%, SL={stop_loss}%")

    from utils.backtesting.backtest_util import run_backtest
    results = run_backtest(
        news_df=filtered_news_df,
        initial_capital=initial_capital,
        position_size=position_size/100,
        take_profit=take_profit/100,
        stop_loss=stop_loss/100
    )
    if results is None:
        print("No trades were generated during the backtest period.")
        return 1
    trades_df, metrics = results
    print_metrics(metrics)
    # Add run_id and rundate columns to trades_df
    run_id = str(uuid.uuid4())[:8]
    rundate = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trades_df['runid'] = run_id
    trades_df['rundate'] = rundate
    # Save trades report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = Path(output_dir) / "backtest" / f"backtest_trades_{timestamp}.csv"
    trades_path.parent.mkdir(exist_ok=True)
    trades_df.to_csv(trades_path, index=False)
    print(f"Trade history saved to: {trades_path}")

    # Save summary report
    summary_cols = [
        'runid', 'rundate',
        'total_return', 'annualized_return', 'total_pnl', 'total_trades', 'win_rate', 'max_drawdown'
    ]
    summary_row = {
        'runid': run_id,
        'rundate': rundate,
        'total_return': metrics.get('total_return'),
        'annualized_return': metrics.get('annualized_return'),
        'total_pnl': metrics.get('total_pnl'),
        'total_trades': metrics.get('total_trades'),
        'win_rate': metrics.get('win_rate'),
        'max_drawdown': metrics.get('max_drawdown', 0)
    }
    summary_df = pd.DataFrame([summary_row], columns=summary_cols)
    summary_path = Path(output_dir) / "backtest" / f"backtest_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
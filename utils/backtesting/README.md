# FineSpresso Backtester Usage Guide

This backtester allows you to run event-driven trading backtests using news data from either CSV files or a database. Results are saved to the `reports/` directory.

## 1. Configuration

All main parameters can be set in a YAML config file (default: `config/backtester_config.yaml`). Example config:

```yaml
start_date: '2025/05/28'
end_date: '2025/06/27'
initial_capital: 10000
position_size: 10
# Take profit and stop loss in percent
# (as in UI: 1.00 and 0.50)
take_profit: 1.0
stop_loss: 0.5
data_source: db  # or 'csv'
publisher: globenewswire_biotech  # single publisher (recommended)
events:
  - changes_in_companys_own_shares
  - business_contracts
  # ... (list as needed)
data_dir: data  # only used for CSV
output_dir: reports
```

- **publisher**: Use a single publisher string (e.g., `globenewswire_biotech`).
- **events**: List of event types to include (optional).
- **data_source**: `db` (database) or `csv` (local files).

## 2. Running the Backtester

### Using the Config File (Recommended)

```bash
python tasks/backtesting/backtester.py
```

This will use `config/backtester_config.yaml` by default.

### Overriding Config with CLI Arguments

You can override any config value via CLI:

```bash
python tasks/backtesting/backtester.py --publisher globenewswire_biotech --start-date 2025-05-28 --end-date 2025-06-27 --data-source db
```

### Using a Custom Config File

```bash
python tasks/backtesting/backtester.py --config path/to/your_config.yaml
```

### Using CSV Data

Set `data_source: csv` in your config, or use `--data-source csv` on the CLI. Make sure your CSV files are in the directory specified by `data_dir` (default: `data`).

## 3. Output

- Results are saved to `reports/backtest_<timestamp>.csv`.
- The output file contains all trade details and can be used for further analysis.

## 4. Notes

- Only single publisher is supported for DB mode (set `publisher` to a string, not a list).
- For DB mode, ensure your environment is configured with the correct `DATABASE_URL`.
- For CSV mode, ensure your data files are up to date and in the correct format.

## 5. Example

```bash
python tasks/backtesting/backtester.py --publisher globenewswire_biotech --data-source db
```

This will run a backtest for the specified publisher using the database as the news source.

## 6. Sample Backtest Results Output

When you run the backtester, you will see output like this in your terminal:

```
============================================================
BACKTEST RESULTS
============================================================
Total Return:        1.0%
Annualized Return:   12.3%
Total PnL:           $98
Total Trades:        198
Win Rate:            36.9%
Average Trade PnL:   $0
Max Drawdown:        0.0%
============================================================
Trade history saved to: data/backtest_20250627_222145.csv
```

This summary is also saved as a CSV file in your output directory for further analysis.

# Features Engineering Module

This directory contains a comprehensive feature engineering system for financial news impact prediction data. The module provides tools for data enrichment, feature creation, and feature selection to prepare data for machine learning models.

## Overview

The features engineering module is designed to transform cleaned financial news data into rich feature sets that capture market dynamics, temporal patterns, company characteristics, and sentiment information. It includes external data enrichment, time-based features, company-specific features, and intelligent feature selection.

## Files Description

### 1. `features_engineering_pipeline.py` - Complete Pipeline Orchestrator
**Purpose**: Orchestrates the entire feature engineering pipeline from data loading to final feature selection.

**Key Features**:
- Sequential execution of all feature engineering steps
- Comprehensive logging and error handling
- Intermediate data saving for debugging
- Integration of all feature engineering components
- Automated pipeline execution

**Main Function**: `run_feature_engineering_pipeline()`

**Pipeline Steps**:
1. **Yahoo Finance Enrichment**: Add market data and company information
2. **Time Features**: Add temporal and market timing features
3. **Company Features**: Add company-specific characteristics
4. **Feature Selection**: Select optimal feature subset

**Usage**:
```bash
python tasks/features_engineering/features_engineering_pipeline.py
```

### 2. `fetch_yfinance_data.py` - External Data Enrichment
**Purpose**: Enriches the dataset with Yahoo Finance market data and company information.

**Key Features**:
- Fetches real-time market data using yfinance API
- Implements intelligent caching to reduce API calls
- Retry mechanism with exponential backoff
- Ticker validation and error handling
- Sector performance calculation

**Main Function**: `enrich_with_yfinance()`

**Added Features**:
- `market_cap`: Company market capitalization
- `float_shares`: Number of shares available for trading
- `exchange`: Stock exchange (NYSE, NASDAQ, etc.)
- `sector`: Company sector classification
- `industry`: Company industry classification
- `avg_volume`: Average trading volume
- `beta`: Stock volatility relative to market
- `recent_volume`: Most recent trading volume
- `float_ratio`: Ratio of float shares to market cap
- `sector_performance`: Sector index performance

**Key Methods**:
- `fetch_yf_data()`: Fetch data with retry mechanism
- `validate_ticker()`: Validate ticker symbols
- `load_cache()` / `save_cache()`: Cache management
- `enrich_with_yfinance()`: Main enrichment function

**Usage**:
```bash
python tasks/features_engineering/fetch_yfinance_data.py
```

### 3. `time_features.py` - Temporal Feature Engineering
**Purpose**: Creates time-based features that capture market timing and temporal patterns.

**Key Features**:
- Day of week and hour extraction
- Weekend and market hours identification
- Earnings season detection
- Quarter-end identification
- Days since event calculation

**Main Function**: `add_time_features()`

**Added Features**:
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `hour`: Hour of day (0-23)
- `is_weekend`: Binary flag for weekends
- `is_market_hours`: Binary flag for market hours (9:30 AM - 4:00 PM EST)
- `is_earnings_season`: Binary flag for earnings season
- `is_quarter_end`: Binary flag for quarter-end months
- `days_since_event`: Days elapsed since news publication

**Market Timing Logic**:
- **Market Hours**: 9:30 AM - 4:00 PM EST (US markets)
- **Earnings Season**: Months following quarter ends (Jan, Apr, Jul, Oct)
- **Quarter End**: March, June, September, December

**Usage**:
```bash
python tasks/features_engineering/time_features.py
```

### 4. `company_features.py` - Company-Specific Features
**Purpose**: Creates company-specific features that capture firm characteristics and historical patterns.

**Key Features**:
- Market cap categorization (Small, Mid, Large)
- Volatility calculation using historical price data
- Sector-relative volatility comparison
- Previous news sentiment analysis
- TextBlob sentiment analysis for news content

**Main Function**: `add_company_features()`

**Added Features**:
- `market_cap_category`: Categorical market cap (Small < $2B, Mid $2B-$10B, Large > $10B)
- `volatility`: Annualized stock volatility (252-day rolling)
- `sector_relative_volatility`: Stock volatility relative to sector index
- `prev_news_sentiment`: Average sentiment of previous news for the company
- `combined_sentiment`: Sentiment score from title + content (if not present)

**Volatility Calculation**:
- Uses 1-year historical price data
- Annualized using √252 (trading days)
- Sector comparison using sector indices (NASDAQ for Tech, Banking for Finance, etc.)

**Usage**:
```bash
python tasks/features_engineering/company_features.py
```

### 5. `features_selection.py` - Intelligent Feature Selection
**Purpose**: Performs feature selection using correlation analysis and Random Forest importance.

**Key Features**:
- Correlation-based feature elimination
- Random Forest importance-based selection
- Categorical feature encoding (One-Hot Encoding)
- Comprehensive feature validation
- Feature importance reporting

**Main Function**: `select_features()`

**Selection Methods**:
1. **Correlation Filter**: Removes highly correlated features (>0.8 threshold)
2. **Random Forest Importance**: Keeps features with importance > threshold
3. **Combined Approach**: Applies both filters sequentially

**Parameters**:
- `correlation_threshold`: Maximum correlation between features (default: 0.8)
- `importance_threshold`: Minimum Random Forest importance (default: 0.01)
- `method`: Selection method ('correlation', 'rf', 'correlation_and_rf')

**Key Methods**:
- `validate_categorical_columns()`: Validate and log categorical features
- `select_features()`: Main feature selection function

**Preserved Columns**:
- `event`: Event type (always preserved)
- `content`: News content
- `title`: News title
- `actual_side`: Target variable
- `price_change_percentage`: Target variable

**Usage**:
```bash
python tasks/features_engineering/features_selection.py
```

## Running the Complete Pipeline

To run the entire feature engineering pipeline:

```bash
cd /path/to/finespresso-modelling
python tasks/features_engineering/features_engineering_pipeline.py
```

This will:
1. Load cleaned data from `data/clean/clean_price_moves.csv`
2. Enrich with Yahoo Finance data
3. Add time-based features
4. Add company-specific features
5. Perform feature selection
6. Save final enriched data to `data/feature_engineering/final_enriched_data.csv`

## Output Structure

The pipeline generates the following outputs:

```
data/
├── feature_engineering/
│   ├── yfinance_enriched_data.csv      # Data with Yahoo Finance features
│   ├── time_features_data.csv          # Data with time features
│   ├── company_features_data.csv       # Data with company features
│   ├── selected_features_data.csv      # Data with selected features
│   └── final_enriched_data.csv         # Final enriched dataset
├── cache/
│   └── yfinance_cache.pkl              # Yahoo Finance data cache
├── quality_metrics/
│   └── yfinance_metrics_YYYYMMDD_HHMMSS.csv  # Yahoo Finance fetch metrics
└── reports/
    ├── time_feature_stats_YYYYMMDD_HHMMSS.csv    # Time feature statistics
    ├── company_features_stats_YYYYMMDD_HHMMSS.csv # Company feature statistics
    └── feature_importance_YYYYMMDD_HHMMSS.csv     # Feature importance rankings
```

## Feature Categories

### Market Data Features (Yahoo Finance)
- **Company Information**: Market cap, sector, industry, exchange
- **Trading Metrics**: Volume, beta, float shares
- **Sector Performance**: Relative sector index performance

### Temporal Features
- **Time Components**: Day of week, hour, days since event
- **Market Timing**: Weekend flags, market hours, earnings season
- **Quarterly Patterns**: Quarter-end identification

### Company-Specific Features
- **Size Classification**: Market cap categories
- **Risk Metrics**: Volatility, sector-relative volatility
- **Sentiment History**: Previous news sentiment patterns

### Text Features
- **Sentiment Analysis**: TextBlob sentiment scores
- **Content Features**: Title and content text (preserved for NLP)

## Configuration

Default parameters can be modified in each script:

**Feature Selection Parameters**:
```python
params = {
    'method': 'correlation_and_rf',
    'correlation_threshold': 0.8,
    'importance_threshold': 0.01
}
```

**Market Cap Categories**:
- Small Cap: < $2 billion
- Mid Cap: $2-10 billion
- Large Cap: > $10 billion

**Sector Indices**:
- Technology: ^IXIC (NASDAQ)
- Finance: ^IXBK (Banking)
- Healthcare: ^IXHC (Healthcare)
- Default: ^GSPC (S&P 500)

## Dependencies

Required Python packages (see `requirements.txt`):
- pandas
- numpy
- yfinance
- scikit-learn
- textblob
- retrying
- logging

## Logging

All scripts generate comprehensive logs stored in:
- `logs/features_engineering/features_engineering_pipeline.log`
- `logs/features_engineering/yfinance.log`
- `logs/features_engineering/time_features.log`
- `logs/features_engineering/company_features.log`
- `logs/features_engineering/features_selection.log`

## Caching

The Yahoo Finance module implements intelligent caching:
- **Cache Location**: `data/cache/yfinance_cache.pkl`
- **Cache Key**: `{ticker}_{YYYYMMDD}`
- **Cache Benefits**: Reduces API calls, improves performance
- **Cache Management**: Automatic loading/saving with error handling

## Error Handling

The pipeline includes robust error handling:
- **API Failures**: Retry mechanism with exponential backoff
- **Invalid Tickers**: Graceful handling with logging
- **Missing Data**: Intelligent imputation and validation
- **Feature Dependencies**: Validation of required columns

## Performance Considerations

- **API Rate Limiting**: Built-in delays and retry logic
- **Memory Management**: Efficient DataFrame operations
- **Caching**: Reduces redundant API calls
- **Parallel Processing**: Sequential execution for data consistency

# SQL Folder

This folder contains database schema definitions and related SQL utilities for the FineSpresso Modelling project.

## Contents

### Database Schema

#### `create-tables.sql`
The main database schema file that defines all tables used in the project:

- **`news`** - Stores news articles with metadata including titles, content, AI summaries, and topic classifications
- **`company`** - Maps company tickers between different data sources (Yahoo Finance, MarketWatch)
- **`price_moves`** - Tracks price movements associated with news events, including actual vs predicted outcomes
- **`eq_model_results_binary`** - Stores binary classification model performance metrics
- **`eq_model_results_regression`** - Stores regression model performance metrics

### Utilities

#### `extract_ddl.py`
Python script for extracting Data Definition Language (DDL) statements from existing databases.

#### `create-tables.ddl`
Alternative DDL file format containing table creation statements.

#### `create-tables.json`
JSON representation of the database schema, likely used for programmatic schema management.

## Database Design Overview

The database is designed to support a news-driven trading model with the following key components:

1. **News Processing Pipeline**: Raw news articles are stored with AI-enhanced metadata
2. **Company Mapping**: Cross-reference tickers across different financial data providers
3. **Price Movement Tracking**: Capture actual market reactions to news events
4. **Model Performance Tracking**: Monitor and compare model accuracy across different event types

## Usage

To set up the database schema:

```bash
# Using the SQL file directly
psql -d your_database -f create-tables.sql

# Or using the DDL file
psql -d your_database -f create-tables.ddl
```

## Schema Relationships

- `news` → `price_moves` (via `news_id`)
- `company` → `price_moves` (via `ticker`)
- Model results tables track performance across different event types and model types

## Notes

- The schema uses PostgreSQL-specific features like `SERIAL` for auto-incrementing IDs
- UUID generation is used for run tracking in model results tables
- Timestamps include timezone information for accurate event tracking
- The schema supports both binary classification and regression model evaluation 
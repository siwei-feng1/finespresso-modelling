# Data Cleaning Module

This directory contains a comprehensive data cleaning and quality management system for financial news impact prediction data. The module provides tools for data analysis, cleaning, validation, metrics monitoring, and versioning.

## Overview

The data cleaning module is designed to handle financial news data with event-specific cleaning strategies, ensuring data quality while preserving important patterns for machine learning models.

## Files Description

### 1. `data_cleaner.py` - Main Data Cleaning Engine
**Purpose**: Core data cleaning functionality with event-specific handling for financial news data.

**Key Features**:
- Event-specific outlier detection with different thresholds for different event types
- Missing value imputation using event-specific medians
- Text cleaning with financial-specific noise removal
- Datetime standardization with flexible parsing
- Relaxed winsorizing (5% instead of 1%) to preserve variance
- Comprehensive logging and metrics tracking

**Main Class**: `DataCleaner`

**Key Methods**:
- `load_data()`: Load CSV data with error handling
- `drop_high_missing_columns()`: Remove columns with >90% missing values
- `impute_missing_values()`: Event-specific imputation for numerical columns
- `detect_outliers_iqr()`: Event-specific outlier detection
- `handle_outliers()`: Winsorizing with relaxed limits
- `clean_datetime()`: Standardize datetime columns
- `clean_text_data()`: Clean text with financial noise removal
- `clean()`: Run complete cleaning pipeline

**Usage**:
```bash
python tasks/data_cleaning/data_cleaner.py
```

### 2. `analyze_data_quality.py` - Data Quality Analysis
**Purpose**: Comprehensive data quality assessment with visualizations and reports.

**Key Features**:
- Missing values analysis with percentage calculations
- Duplicate record detection
- Price movement distribution analysis with outlier detection
- Event type distribution analysis
- Date range validation
- Text quality assessment (length, empty values)
- Automated report generation in CSV format
- Distribution plots for numerical columns

**Main Class**: `DataQualityAnalyzer`

**Outputs**:
- Missing values report
- Duplicate analysis report
- Outlier analysis report
- Event distribution report
- Date analysis report
- Text quality report
- Distribution plots

**Usage**:
```bash
python tasks/data_cleaning/analyze_data_quality.py
```

### 3. `data_metrics.py` - Data Quality Metrics Monitoring
**Purpose**: Continuous monitoring of data quality metrics with baseline comparison.

**Key Features**:
- Completeness metrics calculation
- Consistency metrics (unique value counts)
- Validity metrics (invalid value detection)
- Outlier detection using IQR method
- Distribution plotting for numerical columns
- Baseline comparison functionality
- Automated metrics saving with timestamps

**Main Class**: `DataMetricsMonitor`

**Key Methods**:
- `calculate_completeness()`: Calculate data completeness
- `calculate_consistency()`: Calculate data consistency
- `calculate_validity()`: Calculate data validity
- `calculate_outliers()`: Detect outliers using IQR
- `plot_distributions()`: Generate distribution plots
- `compare_with_baseline()`: Compare with baseline metrics
- `monitor()`: Run complete metrics pipeline

**Usage**:
```bash
python tasks/data_cleaning/data_metrics.py
```

### 4. `data_validation.py` - Data Validation Engine
**Purpose**: Comprehensive data validation with business rule enforcement.

**Key Features**:
- Input structure validation
- Data type validation and conversion
- Categorical value balancing (class balancing for actual_side)
- Relaxed outlier detection (3.0 IQR multiplier)
- Price range validation
- Text quality validation
- Comprehensive error handling and logging

**Main Class**: `DataValidator`

**Key Methods**:
- `check_input_structure()`: Validate input data structure
- `validate_data_types()`: Validate and convert data types
- `validate_categorical_values()`: Balance classes if needed
- `validate_outliers()`: Detect and handle outliers
- `validate_price_ranges()`: Validate price change ranges
- `validate_text_quality()`: Validate text quality
- `validate()`: Run complete validation pipeline

**Usage**:
```bash
python tasks/data_cleaning/data_validation.py
```

### 5. `data_versioning.py` - Data Versioning and Lineage
**Purpose**: Manage dataset versions and track data lineage for reproducibility.

**Key Features**:
- Automatic version numbering (semantic versioning)
- SHA256 hash computation for data integrity
- Lineage tracking with processing steps and parameters
- Version manifest management
- Comprehensive audit trail

**Main Class**: `DataVersioning`

**Key Methods**:
- `compute_hash()`: Compute SHA256 hash of DataFrame
- `get_next_version()`: Determine next version number
- `save_version()`: Save versioned data
- `save_lineage()`: Save lineage information
- `update_manifest()`: Update versions manifest
- `version_and_track()`: Complete versioning and tracking

**Usage**:
```bash
python tasks/data_cleaning/data_versioning.py
```

### 6. `data_quality_pipeline.py` - Complete Pipeline Orchestrator
**Purpose**: Orchestrate the complete data quality pipeline.

**Key Features**:
- Sequential execution of all data quality steps
- Comprehensive logging throughout the pipeline
- Integration of all cleaning, validation, and monitoring components
- Automated versioning and lineage tracking
- Centralized configuration management

**Main Function**: `run_data_quality_pipeline()`

**Pipeline Steps**:
1. Data Quality Analysis
2. Data Cleaning
3. Data Validation
4. Metrics Monitoring
5. Data Versioning

**Usage**:
```bash
python tasks/data_cleaning/data_quality_pipeline.py
```

## Running the Complete Pipeline

To run the entire data quality pipeline:

```bash
cd /path/to/finespresso-modelling
python tasks/data_cleaning/data_quality_pipeline.py
```

This will:
1. Analyze the quality of `data/all_price_moves.csv`
2. Clean the data using event-specific strategies
3. Validate the cleaned data
4. Monitor data quality metrics
5. Version the data and track lineage
6. Save all outputs to appropriate directories

## Output Structure

The pipeline generates the following outputs:

```
data/
├── clean/
│   └── clean_price_moves.csv          # Cleaned data
├── quality_metrics/
│   ├── cleaning_metrics.csv           # Cleaning metrics
│   ├── validation_metrics.csv         # Validation metrics
│   ├── metrics_YYYYMMDD_HHMMSS.csv    # Timestamped metrics
│   └── plots/                         # Distribution plots
├── quality_reports/
│   ├── missing_values_report.csv      # Missing values analysis
│   ├── duplicate_report.csv           # Duplicate analysis
│   ├── outlier_report.csv             # Outlier analysis
│   ├── event_distribution.csv         # Event distribution
│   ├── date_analysis_report.csv       # Date analysis
│   └── text_quality_report.csv        # Text quality analysis
├── versions/
│   ├── v1.0.0/                        # Versioned data
│   ├── v1.0.1/
│   └── versions.csv                    # Version manifest
└── lineage/
    └── lineage_v1.0.0_YYYYMMDD_HHMMSS.json  # Lineage tracking
```

## Configuration

The pipeline uses default paths but can be customized by modifying the parameters in each script:

- **Input data**: `data/all_price_moves.csv`
- **Output data**: `data/clean/clean_price_moves.csv`
- **Metrics directory**: `data/quality_metrics/`
- **Versions directory**: `data/versions/`
- **Lineage directory**: `data/lineage/`

## Dependencies

Required Python packages (see `requirements.txt`):
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- logging

## Logging

All scripts generate comprehensive logs stored in:
- `logs/cleaning.log`
- `logs/metrics.log`
- `logs/validation.log`
- `logs/versioning.log`
- `logs/pipeline.log`

## Event-Specific Handling

The cleaning system uses different strategies for different event types:

- **Partnerships**: Less strict outlier detection (1.5 IQR)
- **Earnings**: Stricter outlier detection (2.0 IQR)
- **Corporate Action**: Moderate outlier detection (1.8 IQR)
- **Default**: Standard outlier detection (1.5 IQR)

This ensures that different types of financial events are handled appropriately based on their inherent characteristics.

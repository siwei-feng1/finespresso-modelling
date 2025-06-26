# Finespresso Modelling - Financial News Impact Prediction

A machine learning system that predicts the impact of financial news on stock prices using natural language processing and machine learning techniques.

## 🎯 Project Overview

This system analyzes financial news articles and predicts:

1. **Direction Classification**: Whether a stock will go UP or DOWN after news release
2. **Price Movement Regression**: The percentage change in stock price

The models are trained on real financial news data with corresponding price movements, covering various event types like earnings releases, clinical studies, mergers & acquisitions, and more.

## 📊 Current Performance

### Classification Models

- **Average Accuracy**: ~84.49% across all events (post-feature engineering, based on `model_comparison_binary.csv`)
- **Best Performing Events**:
  - Capital Investment: 100.0% accuracy
  - Fund Data Announcement: 100.0% accuracy
  - Geographic Expansion: 100.0% accuracy
  - Interim Information: 100.0% accuracy
  - Mergers & Acquisitions: 100.0% accuracy
  - Partnerships: 100.0% accuracy
  - Patents: 100.0% accuracy
  - Product Services Announcement: 100.0% accuracy
  - Voting Rights: 100.0% accuracy
- **Events Above 70% Accuracy**: 24 events (out of 29)
- **Key Insight**: Feature engineering (e.g., sentiment, market context, time-based features) improved average accuracy from ~78.43% (post-cleaning) to ~84.49% (+7.73%). Significant gains in events like `voting_rights` (+400.0%), `business_contracts` (+50.0%), and `earnings_releases_and_operating_results` (+22.22%). Minor regression in `conference_call_webinar` (-4.08%) suggests event-specific sensitivity or small sample sizes.

### Regression Models

- **Average R² Score**: ~38.34% across all events (post-feature engineering, based on `model_comparison_regression.csv`)
- **Best Performing Events**:
  - Geographic Expansion: R² = 0.9873 (98.73%)
  - Business Contracts: R² = 0.9554 (95.54%)
  - Fund Data Announcement: R² = 0.9461 (94.61%)
  - Changes in Company's Own Shares: R² = 0.8397 (83.97%)
- **Events Above R² 0.5**: 11 events (out of 33)
- **Key Insight**: Feature engineering improved R² from 24.77% to 38.34% for `all_events` (+54.79%). Significant gains in `business_contracts` (+4271.31%), `patents` (+12395.66%), and `prospectus_announcement` (+124.28%). Persistent negative R² in events like `company_regulatory_filings` (-378.01%) and `major_shareholder_announcements` (-599.3%) indicates challenges with complex price dynamics or small samples.

### Model Comparison (Post-Cleaning vs. Post-Feature Engineering)

#### Classification Comparison

| Event | Previous Accuracy (%) | Current Accuracy (%) | Accuracy Improvement (%) | Above 70% |
|-------|-----------------------|-----------------------|--------------------------|-----------|
| all_events | 78.43 | 84.49 | 7.73 | True |
| annual_general_meeting | 66.67 | 75.0 | 12.50 | True |
| business_contracts | 50.0 | 75.0 | 50.0 | True |
| capital_investment | 100.0 | 100.0 | 0.0 | True |
| changes_in_companys_own_shares | 72.73 | 81.82 | 12.50 | True |
| clinical_study | 80.23 | 85.47 | 6.52 | True |
| company_regulatory_filings | 80.0 | 80.0 | 0.0 | True |
| conference_call_webinar | 73.13 | 70.15 | -4.08 | True |
| corporate_action | 54.55 | 54.55 | 0.0 | False |
| earnings_releases_and_operating_results | 56.25 | 68.75 | 22.22 | False |
| exchange_announcement | 83.33 | 83.33 | 0.0 | True |
| financial_results | 74.19 | 83.87 | 13.04 | True |
| financing_agreements | 70.0 | 85.0 | 21.43 | True |
| fund_data_announcement | 100.0 | 100.0 | 0.0 | True |
| geographic_expansion | 100.0 | 100.0 | 0.0 | True |
| interim_information | 100.0 | 100.0 | 0.0 | True |
| licensing_agreements | 80.0 | 80.0 | 0.0 | True |
| management_changes | 81.48 | 85.19 | 4.55 | True |
| mergers_acquisitions | 100.0 | 100.0 | 0.0 | True |
| partnerships | 100.0 | 100.0 | 0.0 | True |
| patents | 100.0 | 100.0 | 0.0 | True |
| press_releases | 72.73 | 81.82 | 12.50 | True |
| product_services_announcement | 100.0 | 100.0 | 0.0 | True |
| prospectus_announcement | 66.67 | 66.67 | 0.0 | False |
| regulatory_filings | 66.67 | 73.33 | 10.0 | True |
| share_capital_increase | 66.67 | 66.67 | 0.0 | False |
| shares_issue | 50.0 | 56.25 | 12.5 | False |
| trade_show | 83.33 | 83.33 | 0.0 | True |
| voting_rights | 20.0 | 100.0 | 400.0 | True |

#### Regression Comparison

| Event | Previous R² (%) | Current R² (%) | R² Improvement (%) | Above 0.5 R² |
|-------|------------------|----------------|---------------------|--------------|
| all_events | 24.77 | 38.34 | 54.79 | False |
| annual_general_meeting | -489.26 | -193.31 | -60.49 | False |
| bond_fixing | -17.91 | 18.76 | -204.76 | False |
| business_contracts | 2.19 | 95.54 | 4271.31 | True |
| capital_investment | 68.11 | 50.81 | -25.40 | True |
| changes_in_companys_own_shares | 52.74 | 83.97 | 59.21 | True |
| clinical_study | 28.52 | 38.47 | 34.89 | False |
| company_regulatory_filings | -5100.54 | -378.01 | -92.59 | False |
| conference_call_webinar | 19.55 | 20.56 | 5.18 | False |
| corporate_action | -31.55 | 18.06 | -157.24 | False |
| dividend_reports_and_estimates | -2565.63 | -768.35 | -70.05 | False |
| earnings_releases_and_operating_results | 22.11 | 41.33 | 86.96 | False |
| environmental_social_governance | 0.0 | 0.0 | 0.0 | False |
| exchange_announcement | 42.99 | 54.2 | 26.09 | True |
| financial_results | 23.43 | 45.37 | 93.65 | False |
| financing_agreements | 36.19 | 49.3 | 36.24 | False |
| fund_data_announcement | 92.0 | 94.61 | 2.83 | True |
| geographic_expansion | 69.22 | 98.73 | 42.64 | True |
| interim_information | 78.27 | 27.5 | -64.87 | False |
| licensing_agreements | 33.26 | 8.44 | -74.63 | False |
| major_shareholder_announcements | -1546.18 | -599.3 | -61.24 | False |
| management_changes | 74.15 | 84.65 | 14.16 | True |
| mergers_acquisitions | 30.69 | 31.97 | 4.17 | False |
| partnerships | 69.15 | 73.11 | 5.74 | True |
| patents | -0.51 | 62.94 | -12395.66 | True |
| press_releases | -15.85 | -4.39 | -72.33 | False |
| product_services_announcement | 54.3 | 55.31 | 1.86 | True |
| prospectus_announcement | 29.06 | 65.18 | 124.28 | True |
| regulatory_filings | -0.25 | 11.2 | -4586.24 | False |
| share_capital_increase | -584.4 | -857.28 | 46.70 | False |
| shares_issue | -3.5 | 10.61 | -402.95 | False |
| trade_show | 84.7 | 81.81 | -3.42 | True |
| voting_rights | -47.51 | 78.75 | -265.77 | True |

## 🏗️ System Architecture

### Data Pipeline

```
Database → CSV Export → Data Quality Pipeline → Feature Engineering Pipeline → Model Training → Results & Models
```

### Model Types

- **Random Forest Classifier**: For UP/DOWN prediction
- **Random Forest Regressor**: For price percentage prediction
- **TF-IDF Vectorization**: Text feature extraction
- **spaCy Preprocessing**: Text cleaning and lemmatization
- **New Features**: Sentiment analysis (`TextBlob`), market context (`yfinance`), time-based features, company-specific features, feature selection


### File Organization

```
finespresso-modelling/
├── data/                    # Raw, cleaned, and feature-engineered data CSV files
│   ├── clean/              # Cleaned data from Step 1
│   ├── feature_engineering/ # Feature-engineered data from Step 2
│   ├── quality_metrics/    # Data quality metrics
│   ├── versions/          # Versioned datasets
│   ├── lineage/           # Data lineage logs
├── models/                  # Trained model files (.joblib)
├── reports/                 # Training results and metrics
├── tasks/ai/               # Training and comparison scripts
├── tasks/data_cleaning/    # Data quality and preprocessing scripts
├── tasks/feature_engineering/ # Feature engineering scripts
├── tests/                  # Data download utilities
├── logs/                   # Log files for data processing
└── requirements.txt        # Project dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <repository>
cd finespresso-modelling
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install sentence-transformers textblob scikit-learn scipy yfinance

# Set up database connection
echo "DATABASE_URL='your_database_url'" > .env
```

### 2. Download Data

```bash
python tests/download_data.py
```

### 3. Run Data Quality Pipeline (Step 1)

```bash
python tasks/data_cleaning/data_quality_pipeline.py
```

### 4. Run Feature Engineering Pipeline (Step 2)

```bash
python tasks/features_engineering/features_engineering_pipeline.py
```

### 5. Train Models

```bash
# Train classification models (uses feature-engineered data)
python tasks/ai/train_classifier_enhanced.py

# Train regression models (uses feature-engineered data)
python tasks/ai/train_regression_enhanced.py
```

> **Note:**  
> - If you run the **full feature engineering pipeline**, models will use the combined enriched file:  
>   `data/feature_engineering/final_enriched_data.csv`
> - If you run feature scripts **step by step** (file by file), use the output of each step as input for the next.  
>   For company features, the output is:  
>   `data/feature_engineering/company_features_data.csv`


### 6. Compare Model Results

```bash
# Compare results between cleaned and featured data
python tasks/ai/compare_results.py
```

### 7. View Results

- **Model Performance**: `reports/model_results_binary_after_features_eng.csv`, `reports/model_results_regression_after_features_eng.csv`
- **Model Comparisons**: `reports/model_comparison_binary.csv`, `reports/model_comparison_regression.csv`
- **Trained Models**: `models/` directory
- **Raw and Cleaned Data**: `data/` directory
- **Data Quality Metrics**: `data/quality_metrics/`
- **Versioned Data**: `data/versions/`
- **Data Lineage**: `data/lineage/`
- **Logs**: `data/logs/` directory

### Model Training Process

**Current Approach**

1. **Data Loading**: Load feature-engineered data from `data/feature_engineering/selected_features_data.csv`
2. **Text Processing**:
   - Priority: `content` → `title`
   - spaCy preprocessing (lemmatization, stop word removal)
3. **Feature Extraction**:
   - TF-IDF vectorization (1000 features max)
   - Sentiment scores (`TextBlob`: `combined_sentiment`, `prev_news_sentiment`)
   - Market context (`yfinance`: `market_cap`, `beta`, `sector_performance`)
   - Time-based features (`is_earnings_season`, `is_market_hours`)
   - Company-specific features (`volatility`, `float_ratio`, `market_cap_category`)
4. **Feature Selection**:
   - Correlation threshold (<0.8)
   - Random Forest importance (>0.01)
5. **Data Preprocessing**:
   - Class balancing for `actual_side`
   - Relaxed outlier clipping (5% winsorizing, 3x IQR)
6. **Model Training**:
   - Individual models per event (min 5 samples)
   - All-events fallback model
7. **Evaluation**: 80/20 train-test split with cross-validation metrics
8. **Comparison**: Compare performance between cleaned (`data/clean/clean_price_moves.csv`) and feature-engineered data

**Current Limitations**

- Insufficient samples for some events (e.g., `major_shareholder_announcements`, `voting_rights`)
- Negative R² in regression for some events due to complex price dynamics
- Basic sentiment analysis (`TextBlob`) limits text understanding
- Feature selection may drop relevant predictors

## 🎯 Take-Home Challenge: Model Improvement

You are tasked with improving the financial news impact prediction system. The current models show promise but have significant room for improvement. Your challenge is to enhance the system across multiple dimensions.

### Challenge Tasks

#### 1. 📊 Data Quality Enhancement (Priority: High)

**Current State**: Completed with improved classification and regression performance

**Tasks Completed**:

- [x] Analyzed data quality issues in `data/all_price_moves.csv`

- [x] Implemented data cleaning for outliers and anomalies

- [x] Added data validation checks for price movements

- [x] Created data quality metrics and monitoring

- [x] Handled missing values and text preprocessing edge cases

- [x] Implemented data versioning and lineage tracking

- [x] Balanced `actual_side` classes for classification

- [x] Relaxed outlier clipping to preserve data variance

**Status**: Completed in branch `feature/data-quality-enhancement`

#### 2. 🏢 Feature Engineering & Market Context (Priority: High)

**Current State**: Basic text and sentiment features

**Your Tasks**:

- [x] Integrated Yahoo Finance API (`yfinance`) for market context features (`market_cap`, `beta`, `sector_performance`, etc.)

- [x] Added time-based features (`is_earnings_season`, `is_market_hours`, etc.)

- [x] Created company-specific features (`volatility`, `float_ratio`, `market_cap_category`, etc.)

- [x] Implemented feature selection (`features_selection.py`) with correlation threshold (<0.8) and Random Forest importance (>0.01)

- [x] Generated feature-engineered dataset (`data/feature_engineering/selected_features_data.csv`)

- [x] Updated training scripts (`train_classifier_enhanced.py`, `train_regression_enhanced.py`) to use combined TF-IDF and new features

**Status:** Completed in branch `feature/feature-engineering`

**Expected Impact**: 10-20% improvement in model accuracy

**Performance Impact:**

- **Classifier:** Improved average accuracy from 78.43% to 84.49% (+7.73%), with 24 events above 70% accuracy (e.g., `voting_rights`: +400.0%, `business_contracts`: +50.0%)
- **Regression:** Improved average R² from 24.77% to 38.34% (+54.79%), with 11 events above 0.5 R² (e.g., `business_contracts`: +4271.31%, `geographic_expansion`: +42.64%)
- **Key Features:** Sentiment (`combined_sentiment`, `prev_news_sentiment`), market context (`market_cap`, `sector_performance`), and volatility features drove significant gains


#### 3. 🤖 Model Architecture Improvements (Priority: Medium)

**Current State**: Basic Random Forest models with grid search

**Your Tasks**:

- [ ] Experiment with different model architectures

- [x] Implement hyperparameter optimization (GridSearchCV)

- [ ] Add model interpretability

- [ ] Create model comparison framework

- [ ] Implement cross-validation strategies

**Expected Impact**: 5-15% improvement in model accuracy

#### 4. 📝 Advanced Text Processing (Priority: Medium)

**Current State**: TF-IDF with spaCy preprocessing, sentiment features

**Your Tasks**:

- [x] Implement advanced text vectorization (BERT via `sentence-transformers`)

- [ ] Add domain-specific financial vocabulary

- [x] Implement sentiment analysis features (`TextBlob`)

- [ ] Create text augmentation techniques

- [ ] Add multilingual support

**Expected Impact**: 8-15% improvement in model accuracy

#### 5. 🧠 LLM Integration (Bonus Challenge)

**Current State**: Traditional ML only

**Your Tasks**:

- [ ] Implement few-shot classification using LLMs

- [ ] Create prompt engineering for financial news

- [ ] Implement LLM-based feature extraction

- [ ] Add LLM ensemble with traditional models

- [ ] Create cost-effective LLM usage patterns

**Expected Impact**: 15-25% improvement in model accuracy

#### 6. 📊 Experiment Tracking & MLOps (Bonus Challenge)

**Current State**: No experiment tracking

**Your Tasks**:

- [ ] Integrate MLflow for experiment tracking

- [ ] Implement model serving pipeline

- [ ] Add automated retraining workflows

- [ ] Create model monitoring and alerting

- [ ] Implement A/B testing framework

**Expected Impact**: Better model management and reproducibility

### 🎯 Success Metrics

**Primary Goals:**

- Achieve >70% accuracy for classification models (achieved: 24 events above 70%)
- Achieve positive R² scores for regression models (achieved: 11 events above 0.5)
- Reduce prediction variance across different events

**Secondary Goals:**

- Improve model interpretability
- Reduce training time
- Create reproducible experiments
- Build scalable inference pipeline

---

### 📋 Deliverables

- **Enhanced Training Scripts:** Improved versions of `train_classifier_enhanced.py` and `train_regression_enhanced.py`
- **Feature Engineering Pipeline:** Scripts to extract and integrate new features
- **Model Comparison Report:** Analysis of different approaches and their performance
- **Documentation:** Updated README with your improvements
- **Code Quality:** Clean, well-documented, and tested code

### 🛠️ Technical Requirements

- Python 3.8+
- Familiarity with ML libraries (scikit-learn, pandas, numpy)
- Experience with text processing (spaCy, transformers, sentence-transformers)
- Knowledge of financial markets (bonus)
- Experience with MLOps tools (bonus)

## Feature Branch: `feature/data-quality-enhancement`

This branch implements **Step 1: Data Quality Enhancement** of the take-home challenge, with partial progress on **Feature Engineering** and **Model Architecture Improvements**.

### Work Completed

- **Data Quality Analysis** (`tasks/data_cleaning/analyze_data_quality.py`):
  - Analyzed missing values, duplicates, outliers, event distributions, and text quality.
  - Generated reports in `data/quality_metrics/`.
- **Data Cleaning** (`tasks/data_cleaning/data_cleaner.py`):
  - Implemented event-specific cleaning, relaxed winsorizing (5% for `price_change_percentage`, `daily_alpha`), and text preprocessing.
  - Added class balancing for `actual_side`.
  - Output cleaned data to `data/clean/cleaned_price_moves_YYYYMMDD.csv`.
- **Data Metrics Monitoring** (`tasks/data_cleaning/data_metrics.py`):
  - Monitored completeness, consistency, validity, and outliers.
  - Generated metrics in `data/quality_metrics/`.
- **Data Validation** (`tasks/data_cleaning/data_validation.py`):
  - Validated data types, categorical values, price ranges, and text quality.
  - Added class balancing for `actual_side` and relaxed outlier clipping (3x IQR, capping instead of imputing).
  - Saved validated data to `data/clean/clean_price_moves.csv`.
- **Data Versioning and Lineage Tracking** (`tasks/data_cleaning/data_versioning.py`):
  - Implemented dataset versioning and lineage tracking.
  - Saved to `data/versions/` and `data/lineage/`.
- **Data Quality Pipeline** (`tasks/data_cleaning/data_quality_pipeline.py`):
  - Integrated all components into a unified pipeline.
- **Model Training Updates**:
  - Updated `tasks/ai/train_classifier_enhanced.py` and `tasks/ai/train_regression_enhanced.py`:
    - Reduced sample threshold to 5.
  - Updated `tasks/ai/compare_results.py`:
    - Added sample size tracking, fixed `spacy` subprocess issues.
- **Performance Results**:
  - **Classifier**: Improved from 64.61% to 78.43% average accuracy, with 15 events above 70% (e.g., `trade_show`: +233.33%, `mergers_acquisitions`: +75.0%). Regressions in 7 events (e.g., `business_contracts`: -33.33%).
  - **Regression**: Improved from -55.53% to 24.77% average R², with 9 events above 0.5 (e.g., `trade_show`: +1425.53%, `fund_data_announcement`: +92.0%). Negative R² persists in 25 events (e.g., `company_regulatory_filings`: -5100.54%).
- **Performance Impact**:
  - **Classifier**: Class balancing improved minority class prediction, achieving goal of &gt;70% accuracy for 15 events.
  - **Regression**: Relaxed clipping preserved variance, enabling positive R² for 9 events, but outliers remain a challenge.

### Key Improvements

- Added class balancing for `actual_side` in `data_cleaner.py` and `data_validation.py`.
- Relaxed outlier clipping in `data_cleaner.py` (5% winsorizing) and `data_validation.py` (3x IQR, capping).
- Improved logging and metrics tracking.

### Running the Data Quality Pipeline

```bash
python tasks/data_cleaning/data_quality_pipeline.py
```

### Running Model Training and Comparison

```bash
python tasks/ai/train_classifier_enhanced.py
python tasks/ai/train_regression_enhanced.py
python tasks/ai/compare_results.py
```

### Output Files

- **Cleaned Data**: `data/clean/clean_price_moves.csv`
- **Quality Metrics**: `data/quality_metrics/` (e.g., `validation_metrics.csv`)
- **Versioned Data**: `data/versions/`
- **Lineage Logs**: `data/lineage/`
- **Model Results**: `reports/model_results_binary_after_cleaning.csv`, `reports/model_results_regression.csv`
- **Model Comparisons**: `reports/model_comparison_binary.csv`, `reports/model_comparison_regression.csv`
- **Logs**: `data/logs/`

## Feature Branch: `feature/feature-engineering`

This branch implements **Step 2: Feature Engineering & Market Context** of the take-home challenge.

### Work Completed

- **Yahoo Finance Integration** (`tasks/feature_engineering/fetch_yfinance_data.py`):
  - Added market context features: `market_cap`, `beta`, `sector_performance`, `float_ratio`, `sector`, `industry`.
  - Fetched real-time financial data for companies in the dataset.
- **Time-Based Features** (`tasks/feature_engineering/time_features.py`):
  - Added `is_earnings_season`, `is_market_hours`, `day_of_week`, `quarter` to capture temporal effects.
- **Company-Specific Features** (`tasks/feature_engineering/company_features.py`):
  - Added `volatility`, `avg_volume`, `market_cap_category`, `combined_sentiment`, `prev_news_sentiment` to capture company and news-specific signals.
- **Feature Selection** (`tasks/feature_engineering/features_selection.py`):
  - Implemented correlation-based filtering (<0.8) and Random Forest feature importance (>0.01) to select high-impact features.
  - Generated `reports/feature_importance_<timestamp>.csv`.
- **Feature Engineering Pipeline** (`tasks/feature_engineering/feature_engineering_pipeline.py`):
  - Integrated all feature extraction scripts, outputting `data/feature_engineering/selected_features_data.csv`.
- **Model Training Updates**:
  - Updated `tasks/ai/train_classifier_enhanced.py` and `tasks/ai/train_regression_enhanced.py`:
    - Combined TF-IDF features with new numerical, binary, and categorical features using `scipy.sparse.hstack`.
    - Simplified text selection to content or title (dropped `content_en`, `title_en`).
    - Added `random_state=42` for reproducibility.
    - Improved logging with dedicated `setup_logger` and `.env` support.
  - Updated `tasks/ai/compare_results.py`:
    - Compared performance between cleaned and feature-engineered data.

### Performance Results

- **Classifier:** Improved average accuracy from 78.43% to 84.49% (+7.73%), with 24 events above 70% accuracy (e.g., `voting_rights`: +400.0%, `business_contracts`: +50.0%).
- **Regression:** Improved average R² from 24.77% to 38.34% (+54.79%), with 11 events above 0.5 R² (e.g., `business_contracts`: +4271.31%, `geographic_expansion`: +42.64%).

#### Performance Impact

- **Classifier:** Sentiment (`combined_sentiment`, `prev_news_sentiment`), market context (`market_cap`, `sector_performance`), and volatility features improved prediction for events like `voting_rights` (100%) and `business_contracts` (75%).
- **Regression:** Market context (`volatility`, `float_ratio`) and sentiment features drove significant R² gains for `business_contracts` (95.54%) and `geographic_expansion` (98.73%).
- **Challenges:** Negative R² persists for events like `company_regulatory_filings` (-378.01%) due to complex price dynamics or small samples.

### Key Improvements

- Added 22 new features, enhancing model context and predictive power.
- Implemented feature selection to reduce noise and overfitting.
- Simplified text selection to align with Step 1’s cleaning.
- Enhanced logging and reproducibility with `.env` support and `random_state`.

### Running the Feature Engineering Pipeline

```bash
python tasks/feature_engineering/feature_engineering_pipeline.py
```

### Running Model Training and Comparison

```bash
python tasks/ai/train_classifier_enhanced.py
python tasks/ai/train_regression_enhanced.py
python tasks/ai/compare_results.py
```

### Output Files

- **Feature-Engineered Data:** `data/feature_engineering/selected_features_data.csv`
- **Feature Importance:** `reports/feature_importance_<timestamp>.csv`
- **Model Results:** `reports/model_results_binary_after_features_eng.csv`, `reports/model_results_regression_after_features_eng.csv`
- **Model Comparisons:** `reports/model_comparison_binary.csv`, `reports/model_comparison_regression.csv`
- **Logs:** `logs/classification.log`, `logs/regression.log`

## 🤝 Contributing

### For Take-Home Challenge Participants

#### 1. Fork the Repository

```bash
git clone https://github.com/YOUR_USERNAME/finespresso-modelling.git
cd finespresso-modelling
```

#### 2. Create a Feature Branch

```bash
git checkout -b feature/your-improvement-name
```

#### 3. Implement Your Improvements

- Follow the challenge tasks
- Keep commits atomic and well-described
- Add tests for new functionality
- Update documentation

#### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: describe your improvement"
```

#### 5. Push and Create Pull Request

```bash
git push origin feature/your-improvement-name
# Create a Pull Request on GitHub
```

#### 6. Pull Request Template

```markdown
## 🎯 Challenge Task(s) Addressed
- [ ] Data Quality Enhancement
- [ ] Feature Engineering & Market Context
- [ ] Model Architecture Improvements
- [ ] Advanced Text Processing
- [ ] LLM Integration
- [ ] Experiment Tracking & MLOps

## 📊 Performance Improvements
- **Before**: [Baseline metrics]
- **After**: [Your improved metrics]
- **Improvement**: [Percentage/absolute improvement]

## 🛠️ Technical Changes
- [List of major changes made]
- [New dependencies]
- [Files modified/added]

## 📋 Testing
- [ ] Unit tests added
- [ ] Integration tests
- [ ] Performance benchmarks

## 📚 Documentation
- [ ] README updated
- [ ] Code comments
- [ ] Setup instructions
```

### For General Contributors

- Open an issue for bugs or features
- Create a branch with `fix/` or `feature/` prefix
- Follow code style guidelines
- Add tests and update documentation
- Submit a PR with a clear description

### Code Style Guidelines

- **Python**: Follow PEP 8 standards
- **Documentation**: Use docstrings
- **Commits**: Use conventional commit messages
- **Tests**: Aim for &gt;80% coverage
- **Type Hints**: Use for functions and classes

### Review Process

- **Automated Checks**: Tests and linting
- **Code Review**: One or more maintainers review
- **Performance Review**: Ensure clear performance impact
- **Documentation Review**: Ensure clear documentation
- **Good luck!** We're excited to see your innovative approaches to improving financial news impact prediction! 🚗🚖

```
```
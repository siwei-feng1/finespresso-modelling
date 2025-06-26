# Finespresso Modelling - Financial News Impact Prediction

A machine learning system that predicts the impact of financial news on stock prices using natural language processing and machine learning techniques.

## 🎯 Project Overview

This system analyzes financial news articles and predicts:

1. **Direction Classification**: Whether a stock will go UP or DOWN after news release
2. **Price Movement Regression**: The percentage change in stock price

The models are trained on real financial news data with corresponding price movements, covering various event types like earnings releases, clinical studies, mergers & acquisitions, and more.

## 📊 Current Performance

### Classification Models

- **Average Accuracy**: \~78.43% across all events (post-cleaning, based on `model_comparison_binary.csv`)
- **Best Performing Events**:
  - Capital Investment: 100.0% accuracy
  - Fund Data Announcement: 100.0% accuracy
  - Geographic Expansion: 100.0% accuracy
  - Interim Information: 100.0% accuracy
  - Mergers & Acquisitions: 100.0% accuracy
  - Partnerships: 100.0% accuracy
  - Patents: 100.0% accuracy
  - Product Services Announcement: 100.0% accuracy
- **Events Above 70% Accuracy**: 15 events (out of 30)
- **Key Insight**: Class balancing and relaxed outlier clipping improved average accuracy from \~64.61% to \~78.43% (+21.39%). Significant gains in events like `trade_show` (+233.33%) and `clinical_study` (+34.88%), but regressions in `business_contracts` (-33.33%) and `voting_rights` (-70.0%) suggest event-specific sensitivity.

### Regression Models

- **Average R² Score**: \~24.77% across all events (post-cleaning, based on `model_comparison_regression.csv`)
- **Best Performing Events**:
  - Fund Data Announcement: R² = 0.9200 (92.00%)
  - Trade Show: R² = 0.8470 (84.70%)
  - Interim Information: R² = 0.7827 (78.27%)
  - Management Changes: R² = 0.7415 (74.15%)
- **Events Above R² 0.5**: 9 events (out of 34)
- **Key Insight**: Relaxed outlier clipping improved R² from -55.53% to 24.77% for `all_events`, with 9 events achieving positive R² &gt; 0.5 (e.g., `trade_show`: +1425.53%, `management_changes`: +2161.12%). However, extreme negative R² in `company_regulatory_filings` (-5100.54%) and `dividend_reports_and_estimates` (-2565.63%) indicates persistent outlier issues.

## 🏗️ System Architecture

### Data Pipeline

```
Database → CSV Export → Data Quality Pipeline → Model Training → Results & Models
```

### Model Types

- **Random Forest Classifier**: For UP/DOWN prediction
- **Random Forest Regressor**: For price percentage prediction
- **TF-IDF Vectorization**: Text feature extraction
- **spaCy Preprocessing**: Text cleaning and lemmatization
- **New Features**: Sentiment analysis (`TextBlob`), robust scaling (`RobustScaler`), class balancing, relaxed outlier clipping

### File Organization

```
finespresso-modelling/
├── data/                    # Raw and cleaned data CSV files
├── models/                  # Trained model files (.joblib)
├── reports/                 # Training results and metrics
├── tasks/ai/               # Training and comparison scripts
├── tasks/data_cleaning/    # Data quality and preprocessing scripts
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
pip install sentence-transformers textblob scikit-learn scipy

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

### 4. Train Models

```bash
# Train classification models (uses cleaned data)
python tasks/ai/train_classifier_enhanced.py

# Train regression models (uses cleaned data)
python tasks/ai/train_regression_enhanced.py
```

### 5. Compare Model Results

```bash
# Compare results between original and cleaned data
python tasks/ai/compare_results.py
```

### 6. View Results

- **Model Performance**: `reports/model_results_binary_after_cleaning.csv`, `reports/model_results_regression.csv`
- **Model Comparisons**: `reports/model_comparison_binary.csv`, `reports/model_comparison_regression.csv`
- **Trained Models**: `models/` directory
- **Raw and Cleaned Data**: `data/` directory
- **Data Quality Metrics**: `data/quality_metrics/`
- **Versioned Data**: `data/versions/`
- **Data Lineage**: `data/lineage/`
- **Logs**: `data/logs/` directory

## 📈 Model Training Process

### Current Approach

1. **Data Loading**: Load cleaned data from `data/clean/clean_price_moves.csv`
2. **Text Processing**:
   - Priority: `content` → `title`
   - spaCy preprocessing (lemmatization, stop word removal)
3. **Feature Extraction**: TF-IDF vectorization (1000 features max), sentiment scores (`TextBlob`)
4. **Data Preprocessing**: Class balancing for `actual_side`, relaxed outlier clipping (5% winsorizing, 3x IQR)
5. **Model Training**:
   - Individual models per event (min 5 samples)
   - All-events fallback model
6. **Evaluation**: 80/20 train-test split with cross-validation metrics
7. **Comparison**: Compare performance between original (`data/all_price_moves.csv`) and cleaned data

### Current Limitations

- Insufficient samples for some events (e.g., `major_shareholder_announcements`, `voting_rights`)
- Negative R² in regression for some events due to outliers
- Basic TF-IDF features limit text understanding
- Limited market context features

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

- [ ] Integrate Yahoo Finance API (`yfinance`) for additional features

- [ ] Add time-based features

- [ ] Create company-specific features

- [ ] Implement feature selection and importance analysis

**Expected Impact**: 10-20% improvement in model accuracy

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

**Primary Goals**:

- Achieve &gt;70% accuracy for classification models (achieved: 15 events above 70%)
- Achieve positive R² scores for regression models (achieved: 9 events above 0.5)
- Reduce prediction variance across different events

**Secondary Goals**:

- Improve model interpretability
- Reduce training time
- Create reproducible experiments
- Build scalable inference pipeline

### 📋 Deliverables

1. **Enhanced Training Scripts**: Improved versions of `train_classifier_enhanced.py` and `train_regression_enhanced.py`
2. **Feature Engineering Pipeline**: Scripts to extract and integrate new features
3. **Model Comparison Report**: Analysis of different approaches and their performance
4. **Documentation**: Updated README with your improvements
5. **Code Quality**: Clean, well-documented, and tested code

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
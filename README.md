# Finespresso Modelling - Financial News Impact Prediction

A machine learning system that predicts the impact of financial news on stock prices using natural language processing and machine learning techniques.

## 🎯 Project Overview

This system analyzes financial news articles and predicts:
1. **Direction Classification**: Whether a stock will go UP or DOWN after news release
2. **Price Movement Regression**: The percentage change in stock price

The models are trained on real financial news data with corresponding price movements, covering various event types like earnings releases, clinical studies, mergers & acquisitions, and more.

## 📊 Current Performance

### Classification Models
- **Average Accuracy**: ~57.4% across all events (post-cleaning, based on `model_comparison_binary.csv`)
- **Best Performing Events**: 
  - Business Contracts: 83.33% accuracy
  - Annual General Meeting: 77.78% accuracy
  - Company Regulatory Filings: 75.0% accuracy
  - Voting Rights: 75.0% accuracy
  - Corporate Action: 72.73% accuracy
- **Events Above 70% Accuracy**: 5 events (out of 26)
- **Key Insight**: Cleaning improved accuracy for 10 events (e.g., `business_contracts`: +11.11%), but 12 events regressed (e.g., `partnerships`: -30.0%), indicating data quality or sample size issues.

### Regression Models
- **Average R² Score**: Negative (~-714.7%, skewed by outliers like `capital_investment`), indicating poor fit
- **Best Performing Event**: 
  - Bond Fixing: R² = 0.8019 (80.19%)
- **Events Above R² 0.5**: 1 event (out of 32)
- **Key Insight**: Cleaning improved R² for 16 events (e.g., `bond_fixing`: 0.0% → 80.19%), but extreme negative R² for others (e.g., `capital_investment`: -22004.21%) suggests outlier sensitivity.

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
- **New Features**: Sentiment analysis (`TextBlob`), robust scaling (`RobustScaler`), and planned BERT embeddings

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
pip install sentence-transformers textblob

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
python tasks/ai/train_regression.py
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
4. **Model Training**: 
   - Individual models per event (min 5 samples)
   - All-events fallback model
5. **Evaluation**: 80/20 train-test split with cross-validation metrics
6. **Comparison**: Compare performance between original (`data/all_price_moves.csv`) and cleaned data

### Current Limitations
- Insufficient samples for some events (e.g., `trade_show`, `capital_investment`)
- Negative R² in regression due to outliers
- Basic TF-IDF features limit text understanding
- Limited hyperparameter tuning
- No market context features

## 🎯 Take-Home Challenge: Model Improvement

You are tasked with improving the financial news impact prediction system. The current models show promise but have significant room for improvement. Your challenge is to enhance the system across multiple dimensions.

### Challenge Tasks

#### 1. 📊 Data Quality Enhancement (Priority: High)
**Current State**: Completed with cleaned data and validation

**Your Tasks**:
- [x] Analyze data quality issues in `data/all_price_moves.csv`
- [x] Implement data cleaning for outliers and anomalies
- [x] Add data validation checks for price movements
- [x] Create data quality metrics and monitoring
- [x] Handle missing values and text preprocessing edge cases
- [x] Implement data versioning and lineage tracking


**Status**: Mostly completed in branch `feature/data-quality-enhancement`

#### 2. 🏢 Feature Engineering & Market Context (Priority: High)
**Current State**: Basic text and sentiment features

**Your Tasks**:
- [ ] Integrate Yahoo Finance API (`yfinance`) for additional features:
  - Market capitalization
  - Stock float ratio
  - Exchange information
  - Sector/industry classification
  - Trading volume
  - Beta coefficient
- [ ] Add time-based features:
  - Market hours vs after-hours
  - Day of week effects
  - Earnings season indicators
- [ ] Create company-specific features:
  - Historical volatility
  - Previous news sentiment
  - Company size classification
- [ ] Implement feature selection and importance analysis

**Expected Impact**: 10-20% improvement in model accuracy

#### 3. 🤖 Model Architecture Improvements (Priority: Medium)
**Current State**: Basic Random Forest models with grid search

**Your Tasks**:
- [ ] Experiment with different model architectures:
  - Gradient Boosting (XGBoost, LightGBM)
  - Deep Learning (LSTM, Transformer-based models)
  - Ensemble methods
- [x] Implement hyperparameter optimization (GridSearchCV)
- [ ] Add model interpretability (SHAP, LIME)
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
- [ ] Implement few-shot classification using LLMs:
  - OpenAI GPT models
  - Local LLMs (Llama, Mistral)
  - Claude API integration
- [ ] Create prompt engineering for financial news
- [ ] Implement LLM-based feature extraction
- [ ] Add LLM ensemble with traditional models
- [ ] Create cost-effective LLM usage patterns

**Expected Impact**: 15-25% improvement in model accuracy

#### 6. 📊 Experiment Tracking & MLOps (Bonus Challenge)
**Current State**: No experiment tracking

**Your Tasks**:
- [ ] Integrate MLflow for experiment tracking:
  - Model versioning
  - Hyperparameter logging
  - Performance metrics tracking
  - Model comparison dashboards
- [ ] Implement model serving pipeline
- [ ] Add automated retraining workflows
- [ ] Create model monitoring and alerting
- [ ] Implement A/B testing framework

**Expected Impact**: Better model management and reproducibility

### 🎯 Success Metrics

**Primary Goals**:
- Achieve >70% accuracy for classification models (currently 5 events above 70%)
- Achieve positive R² scores for regression models (currently only `bond_fixing` positive)
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
  - Analyzed missing values, duplicates, outliers, event distributions, and text quality in `data/all_price_moves.csv`.
  - Generated reports (`missing_values_report.csv`, `duplicate_report.csv`) in `data/quality_metrics/`.
- **Data Cleaning** (`tasks/data_cleaning/data_cleaner.py`):
  - Implemented event-specific cleaning for missing values, outliers (winsorizing), datetime standardization, and text preprocessing.
  - Output cleaned data to `data/clean/cleaned_price_moves_YYYYMMDD.csv`.
  - Logged to `data/logs/cleaner.log`.
- **Data Metrics Monitoring** (`tasks/data_cleaning/data_metrics.py`):
  - Monitored completeness, consistency, validity, and outliers.
  - Generated metrics and plots in `data/quality_metrics/`.
  - Logged to `data/logs/metrics.log`.
- **Data Validation** (`tasks/data_cleaning/data_validation.py`):
  - Validated data types, categorical values (`event`, `actual_side`), price ranges, and text quality.
  - Fixed empty output issue by imputing invalid values (e.g., random UP/DOWN for `actual_side`, median for prices).
  - Saved validated data to `data/clean/clean_price_moves.csv` and metrics to `data/quality_metrics/validation_metrics.csv`.
  - Logged to `data/logs/validation.log`.
- **Data Versioning and Lineage Tracking** (`tasks/data_cleaning/data_versioning.py`):
  - Implemented dataset versioning with SHA256 hashes and lineage tracking.
  - Saved versioned datasets to `data/versions/` and lineage logs to `data/lineage/`.
  - Logged to `data/logs/versioning.log`.
- **Data Quality Pipeline** (`tasks/data_cleaning/data_quality_pipeline.py`):
  - Integrated all components into a unified pipeline.
  - Ensured separate logging to `data/logs/pipeline.log`.
- **Model Training Updates**:
  - Updated `tasks/ai/train_classifier_enhanced.py`:
    - Reduced sample threshold to 5.
  - Updated `tasks/ai/train_regression_enhanced.py`:
    - Reduced sample threshold to 5.

  - Updated `tasks/ai/compare_results.py`:
    - Added `total_sample_prev` and `total_sample_curr` to track sample size changes.
    - Fixed subprocess issues to avoid `spacy` errors.
- **Performance Results** (based on `model_comparison_binary.csv` and `model_comparison_regression.csv`):
  - **Classifier**:
    - Improved accuracy for 10 events (e.g., `business_contracts`: 75.0% → 83.33%, `annual_general_meeting`: 62.5% → 77.78%).
    - Regressed for 12 events (e.g., `partnerships`: 90.91% → 63.64%, `management_changes`: 64.71% → 40.0%).
    - 5 events above 70% accuracy.
    - Average accuracy: ~57.4%.
  - **Regression**:
    - Improved R² for 16 events (e.g., `bond_fixing`: 0.0% → 80.19%, `corporate_action`: -43.7% → 0.98%).
    - Extreme negative R² for some (e.g., `capital_investment`: -22004.21%).
    - Only 1 event (`bond_fixing`) above R² 0.5.
    - Average R²: ~-714.7% (skewed by outliers).
- **Performance Impact**:
  - **Classifier**: Cleaning improved accuracy for key events, but regressions highlight sample size and data quality issues.
  - **Regression**: Robust scaling helped some events, but outliers remain a challenge.
  - **Expected Further Improvement**: 10-20% accuracy boost and positive R² with additional data and features.

### Key Fixes
- Fixed empty output in `data_validation.py` by improving imputation logic.
- Removed `content_en` and `title_en` expectations, preserving original event names.
- Added class balancing and relaxed outlier clipping in `data_validation.py`.
- Fixed `compare_results.py` to avoid `spacy` subprocess errors.
- Enhanced logging across all scripts to `data/logs/`.

### Running the Data Quality Pipeline
```bash
python tasks/data_cleaning/data_quality_pipeline.py
```

### Running Model Training and Comparison
```bash
# Train models with cleaned data
python tasks/ai/train_classifier_enhanced.py
python tasks/ai/train_regression_enhanced.py

# Compare results between original and cleaned data
python tasks/ai/compare_results.py
```

### Output Files
- **Cleaned Data**: `data/clean/clean_price_moves.csv`
- **Quality Metrics**: `data/quality_metrics/` (e.g., `validation_metrics.csv`)
- **Versioned Data**: `data/versions/` (e.g., `v1.0.0/clean_price_moves.csv`)
- **Lineage Logs**: `data/lineage/` (e.g., `lineage_v1.0.0_YYYYMMDD_HHMMSS.json`)
- **Model Results**: `reports/model_results_binary_after_cleaning.csv`, `reports/model_results_regression.csv`
- **Model Comparisons**: `reports/model_comparison_binary.csv`, `reports/model_comparison_regression.csv`
- **Logs**: `data/logs/` (e.g., `validation.log`, `classification.log`)


## 🤝 Contributing

### For Take-Home Challenge Participants

If you're working on the take-home challenge, please follow this workflow:

#### 1. Fork the Repository
```bash
# Fork this repository on GitHub
# Then clone your fork locally
git clone https://github.com/YOUR_USERNAME/finespresso-modelling.git
cd finespresso-modelling
```

#### 2. Create a Feature Branch
```bash
# Create and switch to a new feature branch
git checkout -b feature/your-improvement-name
```

#### 3. Implement Your Improvements
- Follow the challenge tasks outlined above
- Keep your commits atomic and well-described
- Add tests for new functionality
- Update documentation as needed

#### 4. Commit Your Changes
```bash
# Add your changes
git add .

# Commit with descriptive messages
git commit -m "feat: integrate Yahoo Finance API for market features"
git commit -m "feat: implement BERT embeddings for text processing"
```

#### 5. Push and Create Pull Request
```bash
# Push your feature branch
git push origin feature/your-improvement-name

# Create a Pull Request on GitHub
```

#### 6. Pull Request Template
When creating your PR, please include:

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
- [New dependencies added]
- [Files modified/added]

## 📋 Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Performance benchmarks included

## 📚 Documentation
- [ ] README updated
- [ ] Code comments added
- [ ] Setup instructions included
```

### For General Contributors

1. **Open an Issue**: Describe the bug or feature request
2. **Create a Branch**: Use `fix/` or `feature/` prefix
3. **Follow Code Style**: Use consistent formatting and naming
4. **Add Tests**: Ensure new code is tested
5. **Update Docs**: Keep documentation current
6. **Submit PR**: Create a pull request with clear description

### Code Style Guidelines

- **Python**: Follow PEP 8 standards
- **Documentation**: Use docstrings for functions and classes
- **Commits**: Use conventional commit messages
- **Tests**: Aim for >80% code coverage
- **Type Hints**: Use type hints for function parameters and returns

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: At least one maintainer reviews the PR
3. **Performance Review**: For model changes, performance impact is assessed
4. **Documentation Review**: Ensure documentation is clear and complete
5. **Merge**: Once approved, PR is merged to main branch

---

**Good luck! We're excited to see your innovative approaches to improving financial news impact prediction! 🚀**
# FineSpresso Modelling - Financial News Impact Prediction

A machine learning system that predicts the impact of financial news on stock prices using natural language processing and machine learning techniques.

## 🎯 Project Overview

This system analyzes financial news articles and predicts:
1. **Direction Classification**: Whether a stock will go UP or DOWN after news release
2. **Price Movement Regression**: The percentage change in stock price

The models are trained on real financial news data with corresponding price movements, covering various event types like earnings releases, clinical studies, mergers & acquisitions, and more.

## 📊 Current Performance

### Classification Models
- **Average Accuracy**: ~62% across all events
- **Best Performing Events**: 
  - Partnerships: 90.9% accuracy
  - Annual General Meeting: 87.5% accuracy
  - Corporate Action: 70% accuracy

### Regression Models
- **Average R² Score**: Currently negative (indicating room for improvement)
- **Best Performing Events**: 
  - Management Changes: R² = 0.033
  - Business Contracts: R² = 0.044
  - Voting Rights: R² = 0.022

## 🏗️ System Architecture

### Data Pipeline
```
Database → CSV Export → Model Training → Results & Models
```

### Model Types
- **Random Forest Classifier**: For UP/DOWN prediction
- **Random Forest Regressor**: For price percentage prediction
- **TF-IDF Vectorization**: Text feature extraction
- **spaCy Preprocessing**: Text cleaning and lemmatization

### File Organization
```
finespresso-modelling/
├── data/                    # Raw data CSV files
├── models/                  # Trained model files (.joblib)
├── reports/                 # Training results and metrics
├── tasks/ai/               # Training and prediction scripts
└── tests/                  # Data download utilities
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and setup
git clone <repository>
cd finespresso-modelling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set up database connection
echo "DATABASE_URL='your_database_url'" > .env
```

### 2. Download Data
```bash
python tests/download_data.py
```

### 3. Train Models
```bash
# Train classification models
python tasks/ai/train_classifier.py

# Train regression models
python tasks/ai/train_regression.py
```

### 4. View Results
- **Model Performance**: `reports/model_results_binary.csv` and `reports/model_results_regression.csv`
- **Trained Models**: `models/` directory
- **Raw Data**: `data/` directory

## 📈 Model Training Process

### Current Approach
1. **Data Loading**: Load merged news and price data from CSV
2. **Text Processing**: 
   - Priority: `content_en` → `title_en` → `content` → `title`
   - spaCy preprocessing (lemmatization, stop word removal)
3. **Feature Extraction**: TF-IDF vectorization (1000 features max)
4. **Model Training**: 
   - Individual models per event (min 10 samples)
   - All-events fallback model
5. **Evaluation**: 80/20 train-test split with cross-validation metrics

### Current Limitations
- Basic text features only
- No market context features
- Simple TF-IDF vectorization
- Limited hyperparameter tuning
- No experiment tracking

## 🎯 Take-Home Challenge: Model Improvement

You are tasked with improving the financial news impact prediction system. The current models show promise but have significant room for improvement. Your challenge is to enhance the system across multiple dimensions.

### Challenge Tasks

#### 1. 📊 Data Quality Enhancement (Priority: High)
**Current State**: Basic text preprocessing with potential data quality issues

**Your Tasks**:
- [ ] Analyze data quality issues in `data/all_price_moves.csv`
- [ ] Implement data cleaning for outliers and anomalies
- [ ] Add data validation checks for price movements
- [ ] Create data quality metrics and monitoring
- [ ] Handle missing values and text preprocessing edge cases
- [ ] Implement data versioning and lineage tracking

**Expected Impact**: 5-10% improvement in model accuracy

#### 2. 🏢 Feature Engineering & Market Context (Priority: High)
**Current State**: Only text features used

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
**Current State**: Basic Random Forest models

**Your Tasks**:
- [ ] Experiment with different model architectures:
  - Gradient Boosting (XGBoost, LightGBM)
  - Deep Learning (LSTM, Transformer-based models)
  - Ensemble methods
- [ ] Implement hyperparameter optimization (Optuna, Hyperopt)
- [ ] Add model interpretability (SHAP, LIME)
- [ ] Create model comparison framework
- [ ] Implement cross-validation strategies

**Expected Impact**: 5-15% improvement in model accuracy

#### 4. 📝 Advanced Text Processing (Priority: Medium)
**Current State**: Basic TF-IDF with spaCy preprocessing

**Your Tasks**:
- [ ] Implement advanced text vectorization:
  - Word2Vec, GloVe, FastText embeddings
  - BERT/RoBERTa fine-tuning
  - Sentence transformers
- [ ] Add domain-specific financial vocabulary
- [ ] Implement sentiment analysis features
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
- Achieve >70% accuracy for classification models
- Achieve positive R² scores for regression models
- Reduce prediction variance across different events

**Secondary Goals**:
- Improve model interpretability
- Reduce training time
- Create reproducible experiments
- Build scalable inference pipeline

### 📋 Deliverables

1. **Enhanced Training Scripts**: Improved versions of `train_classifier.py` and `train_regression.py`
2. **Feature Engineering Pipeline**: Scripts to extract and integrate new features
3. **Model Comparison Report**: Analysis of different approaches and their performance
4. **Documentation**: Updated README with your improvements
5. **Code Quality**: Clean, well-documented, and tested code

### 🛠️ Technical Requirements

- Python 3.8+
- Familiarity with ML libraries (scikit-learn, pandas, numpy)
- Experience with text processing (spaCy, transformers)
- Knowledge of financial markets (bonus)
- Experience with MLOps tools (bonus)

### 🚀 Getting Started

1. **Fork the repository** and set up your development environment
2. **Run the baseline models** to understand current performance
3. **Choose your focus areas** from the challenge tasks
4. **Implement improvements** incrementally
5. **Document your approach** and results
6. **Submit your enhanced solution**

### 📚 Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [MLflow Documentation](https://mlflow.org/)
- [Financial NLP Papers](https://paperswithcode.com/task/financial-nlp)
- [Transformers Library](https://huggingface.co/transformers/)

---

**Good luck! We're excited to see your innovative approaches to improving financial news impact prediction! 🚀**


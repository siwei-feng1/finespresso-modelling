# AI Modeling System Documentation

## Overview

The system consists of two types of prediction models:
1. Direction Classification (Binary UP/DOWN prediction)
2. Price Move Regression (Predicting percentage price change)

Both model types are trained per event and also have an "all_events" model that combines data across events.

## Model Architecture

### Libraries Used
- scikit-learn: Core ML functionality (RandomForest models, TF-IDF vectorization)
- spaCy: Text preprocessing with English language model
- pandas: Data handling
- joblib: Model serialization
- numpy: Numerical operations

### Model Types

#### Direction Classification Model
- **Model**: RandomForestClassifier
- **Input**: News text (preprocessed with spaCy)
- **Output**: Binary classification (UP/DOWN)
- **Features**: TF-IDF vectors (max 1000 features)
- **Training File**: `train_classifier.py`

#### Price Move Regression Model
- **Model**: RandomForestRegressor
- **Input**: News text (preprocessed with spaCy)
- **Output**: Predicted price change percentage
- **Features**: TF-IDF vectors (max 1000 features)
- **Training File**: `train_regression.py`

## File Structure

### Data Storage
- **Raw Data**: `data/` directory contains CSV files downloaded from the database
  - `all_price_moves.csv`: Merged news and price data for training
  - `all_news.csv`: Complete news dataset
  - `all_instruments.csv`: Instrument data
  - Various summary and report files

### Model Storage
Models are saved in the `models/` directory with the following naming convention:

For event-specific models: 
```
models/{event_name}_classifier_binary.joblib          # Direction classifier
models/{event_name}_tfidf_vectorizer_binary.joblib    # Direction TF-IDF vectorizer
models/{event_name}_regression.joblib                 # Move regression model
models/{event_name}_tfidf_vectorizer_regression.joblib # Move TF-IDF vectorizer
```

For all_events models:
```
models/all_events_classifier_binary.joblib            # Direction classifier
models/all_events_tfidf_vectorizer_binary.joblib      # Direction TF-IDF vectorizer
models/all_events_regression.joblib                   # Move regression model
models/all_events_tfidf_vectorizer_regression.joblib  # Move TF-IDF vectorizer
```

### Results Storage
Training results are saved in the `reports/` directory:
```
reports/model_results_binary.csv      # Classification model performance metrics
reports/model_results_regression.csv  # Regression model performance metrics
```

## Training Process

### Data Preparation
Before training, download all data from the database using:
```bash
python tests/download_data.py
```

This creates the necessary CSV files in the `data/` directory.

### Training Files
1. `train_classifier.py`: Trains binary classification models
2. `train_regression.py`: Trains regression models for price movement prediction

Both training scripts now support a `--source` argument to select the data source:
- `--source db` (default): Loads data from the database (recommended for up-to-date data)
- `--source csv`: Loads data from the CSV file (`data/all_price_moves.csv`)

Example usage:
```bash
# Train classification models using the database
python tasks/ai/train_classifier.py --source db

# Train regression models using the database
python tasks/ai/train_regression.py --source db

# Train classification models using CSV
python tasks/ai/train_classifier.py --source csv

# Train regression models using CSV
python tasks/ai/train_regression.py --source csv
```

Both training scripts:
1. Load news and price data from CSV files (`data/all_price_moves.csv`)
2. Preprocess text using spaCy
3. Train individual models for each event
4. Train an all-events model
5. Save models and vectorizers to `models/` directory
6. Save performance metrics to `reports/` directory
7. Record performance metrics in database

### Running Training
```bash
# Train classification models
python tasks/ai/train_classifier.py

# Train regression models
python tasks/ai/train_regression.py
```


## Running Enhanced Training and Comparison Scripts

The system includes enhanced training scripts and a comparison script to evaluate model performance before and after data cleaning and feature engineering improvements (e.g., KNN imputation in `data_cleaner.py`).

### Enhanced Training Scripts
1. **`train_classifier_enhanced.py`**: Trains improved binary classification models with optimized hyperparameters and enhanced feature engineering (e.g., refined TF-IDF features or additional numerical features from cleaned data).
   - **Input**: Cleaned data from `data/all_price_moves.csv`
   - **Output**: Saves models to `models/` and metrics to `reports/model_results_binary.csv`
   - **Command**:
     ```bash
     python tasks/ai/train_classifier_enhanced.py
     ```

2. **`train_regression_enhanced.py`**: Trains improved regression models for price movement prediction with optimized hyperparameters and enhanced feature engineering.
   - **Input**: Cleaned data from `data/all_price_moves.csv`
   - **Output**: Saves models to `models/` and metrics to `reports/model_results_regression.csv`
   - **Command**:
     ```bash
     python tasks/ai/train_regression_enhanced.py
     ```

### Comparison Script
- **`compare_results.py`**: Compares model performance before and after enhancements by analyzing metrics from previous and current runs.
  - **Input Files**:
    - `reports/model_results_binary.csv`: Contains classification metrics (e.g., accuracy, precision) for current models.
    - `reports/model_results_regression.csv`: Contains regression metrics (e.g., R², MSE) for current models.
    - **Note**: To compare with previous results, ensure previous metrics files (e.g., `reports/model_results_binary_prev.csv` and `reports/model_results_regression_prev.csv`) are available. Rename or copy previous results files before running new training to avoid overwriting.
  - **Output**: Generates comparison reports (e.g., CSV or plots) in the `reports/` directory, detailing improvements in accuracy and R².
  - **Command**:
    ```bash
    python tasks/ai/compare_results.py
    ```

### Steps to Run Comparison
1. **Backup Previous Results**:
   - Before running enhanced training, copy the existing `model_results_binary.csv` and `model_results_regression.csv` to `model_results_binary_prev.csv` and `model_results_regression_prev.csv` in the `reports/` directory:
     ```bash
     cp reports/model_results_binary.csv reports/model_results_binary_prev.csv
     cp reports/model_results_regression.csv reports/model_results_regression_prev.csv
     ```
2. **Run Enhanced Training**:
   - Execute `train_classifier_enhanced.py` and `train_regression_enhanced.py` to generate new results.
3. **Run Comparison**:
   - Execute `compare_results.py` to compare old and new metrics.

## Performance Comparison: Before and After Data Cleaning and Feature Engineering

The enhancements to the data cleaning process, specifically replacing the dropping of high-missing columns with KNN imputation for numerical columns and mode imputation for categorical columns in `data_cleaner.py`, have significantly impacted model performance. Below is a comparison of classification (accuracy) and regression (R²) results before and after these changes, highlighting key findings and limitations.

### Classification Performance (Direction Prediction)
The classification results compare the previous and current accuracy for each event type, with the goal of achieving accuracy above 70%.

| Event                            | Prev Accuracy (%) | Curr Accuracy (%) | Accuracy Improvement (%) | Above 70% |
|----------------------------------|-------------------|-------------------|--------------------------|-----------|
| all_events                       | 65.15             | 82.35             | 26.41                    | True      |
| business_contracts               | 75.00             | 100.00            | 33.33                    | True      |
| corporate_action                 | 60.00             | 90.91             | 51.52                    | True      |
| financial_results                | 52.63             | 83.87             | 59.35                    | True      |
| regulatory_filings               | 63.64             | 93.33             | 46.67                    | True      |
| mergers_acquisitions             | 57.14             | 100.00            | 75.00                    | True      |
| annual_general_meeting           | 75.00             | 66.67             | -11.11                   | False     |
| company_regulatory_filings       | 66.67             | 60.00             | -10.00                   | False     |
| major_shareholder_announcements  | 66.67             | 0.00              | -100.00                  | False     |

**Key Findings**:
- **Improvements**: Significant accuracy improvements were observed for most events, with `all_events` increasing from 65.15% to 82.35% (+26.41%). Notable gains include `mergers_acquisitions` (+75.00%), `financial_results` (+59.35%), and `corporate_action` (+51.52%), all exceeding the 70% threshold.
- **High Performers**: Events like `business_contracts` and `mergers_acquisitions` achieved 100% accuracy, likely due to improved data quality from KNN imputation preserving critical numerical features.
- **Declines**: Some events, such as `annual_general_meeting` (-11.11%) and `major_shareholder_announcements` (-100.00%), saw declines, possibly due to insufficient data or event-specific noise introduced during imputation.

**Limitations to Address**:
- **Events Below 70%**: Events like `annual_general_meeting`, `company_regulatory_filings`, and `major_shareholder_announcements` have accuracies below 70%. These events may require more data or tailored feature engineering (e.g., event-specific text features or additional numerical indicators).
- **Data Sparsity**: Events with low sample sizes (e.g., `major_shareholder_announcements`) may suffer from overfitting or poor generalization post-imputation.
- **Imputation Artifacts**: KNN imputation may introduce biases in events with highly variable numerical data, requiring further tuning of imputation parameters (e.g., number of neighbors).

### Regression Performance (Price Move Prediction)
The regression results compare the previous and current R² values, with the goal of achieving positive R² (>0.5).

| Event                            | Prev R² (%) | Curr R² (%) | R² Improvement (%) | Above 0.5 R² |
|----------------------------------|-------------|-------------|--------------------|--------------|
| all_events                       | -55.53      | 33.43       | -160.19            | False        |
| bond_fixing                      | 0.00        | 82.22       | 0.00               | True         |
| exchange_announcement            | -1614.31    | 77.92       | -104.83            | True         |
| financing_agreements             | 6.05        | 55.73       | 821.82             | True         |
| fund_data_announcement           | -133557.63  | 81.12       | -100.06            | True         |
| product_services_announcement    | -79.32      | 53.26       | -167.15            | True         |
| prospectus_announcement          | 0.00        | 76.69       | 0.00               | True         |
| clinical_study                   | -64.70      | 35.13       | -154.30            | False        |
| major_shareholder_announcements  | -29.38      | -3909.90    | 13209.58           | False        |

**Key Findings**:
- **Improvements**: Significant R² improvements were seen in several events, with `bond_fixing` (0.00 to 82.22), `exchange_announcement` (-1614.31 to 77.92), and `financing_agreements` (6.05 to 55.73) achieving positive R² above 0.5. The `all_events` model improved from -55.53 to 33.43, though it remains below 0.5.
- **High Performers**: Events like `bond_fixing`, `exchange_announcement`, and `fund_data_announcement` show strong positive R², indicating that KNN imputation and preserved data variance improved regression predictions.
- **Declines**: Some events, such as `major_shareholder_announcements` (-29.38 to -3909.90), saw drastic declines, likely due to data sparsity or inappropriate imputation for highly volatile price movements.

**Limitations to Address**:
- **Negative R²**: Many events, including `all_events` and `clinical_study`, still have R² below 0.5 or negative, indicating poor predictive power. This suggests that the regression models struggle with capturing complex price movement patterns.
- **Feature Engineering Needs**: Additional features (e.g., market sentiment, historical price trends, or external economic indicators) may be needed to improve R², especially for events with high variability.
- **Data Quality**: Events with negative R² (e.g., `major_shareholder_announcements`, `share_capital_increase`) may require more robust data cleaning techniques, such as event-specific imputation strategies or outlier removal before imputation.
- **Sample Size**: Events with limited data (e.g., `prospectus_announcement`) may benefit from transfer learning or using the `all_events` model as a starting point.

## Directory Organization

```
finespresso-modelling/
├── data/                    # Raw data CSV files
│   ├── all_price_moves.csv
│   ├── all_news.csv
│   ├── all_instruments.csv
│   └── ...
├── models/                  # Trained model files (.joblib)
│   ├── *_classifier_binary.joblib
│   ├── *_tfidf_vectorizer_binary.joblib
│   ├── *_regression.joblib
│   └── *_tfidf_vectorizer_regression.joblib
├── reports/                 # Training results and metrics
│   ├── model_results_binary.csv
│   ├── model_results_regression.csv
│   ├── model_results_binary_prev.csv
│   ├── model_results_regression_prev.csv
│   └── ...
├── tasks/ai/               # Training and prediction scripts
│   ├── train_classifier.py
│   ├── train_regression.py
│   ├── train_classifier_enhanced.py
│   ├── train_regression_enhanced.py
│   ├── compare_results.py
│   └── predict.py
├── tasks/data_cleaning/    # Data cleaning scripts
│   └── data_cleaner.py
├── tests/                  # Data download utilities
│   └── download_data.py
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```


### Model Usage

The `predict.py` script handles predictions using the trained models:

1. Loads appropriate models based on event type
2. Falls back to all_events model if event-specific model unavailable
3. Makes two predictions for each news item:
   - Direction (UP/DOWN) using classifier models
   - Price move percentage using regression models

### Text Processing Priority

Both training and prediction use the following priority order for text selection:
1. content_en (English translated content)
2. title_en (English translated title)
3. content (Original content)
4. title (Original title)

## Performance Tracking

Training results are saved in:
- CSV files: `reports/model_results_binary.csv` and `reports/model_results_regression.csv`
- Database tables through model_db_util

Metrics tracked:
- Direction models: accuracy, precision, recall, F1 score, AUC-ROC, directional accuracy (UP/DOWN)
- Regression models: MSE, R², MAE, RMSE

## Directory Organization

```
finespresso-modelling/
├── data/                    # Raw data CSV files
│   ├── all_price_moves.csv
│   ├── all_news.csv
│   ├── all_instruments.csv
│   └── ...
├── models/                  # Trained model files (.joblib)
│   ├── *_classifier_binary.joblib
│   ├── *_tfidf_vectorizer_binary.joblib
│   ├── *_regression.joblib
│   └── *_tfidf_vectorizer_regression.joblib
├── reports/                 # Training results and metrics
│   ├── model_results_binary.csv
│   └── model_results_regression.csv
├── tasks/ai/               # Training and prediction scripts
│   ├── train_classifier.py
│   ├── train_regression.py
│   └── predict.py
└── tests/                  # Data download utilities
    └── download_data.py
```

## Usage Example

```python
from tasks.ai.predict import predict
import pandas as pd

# Create DataFrame with news items
news_df = pd.DataFrame({
    'event': ['fed_meeting'],
    'content_en': ['Federal Reserve raises interest rates by 25 basis points'],
    'title_en': ['Fed Hikes Rates']
})

# Get predictions
predictions_df = predict(news_df)
# Results include 'predicted_move' and 'predicted_side' columns
```

## Data Workflow

1. **Data Download**: Use `tests/download_data.py` to extract data from database to CSV files
2. **Model Training**: Run training scripts to create models from CSV data
3. **Results Analysis**: Review performance metrics in `reports/` directory
4. **Prediction**: Use trained models for new predictions

## Notes

- Models require at least 10 samples per event for training
- Each event gets its own model if sufficient data exists
- The all_events model serves as a fallback for new or rare events
- Models are automatically retrained periodically to incorporate new data
- All data is loaded from CSV files rather than direct database queries for better performance and reproducibility
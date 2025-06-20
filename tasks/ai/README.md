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
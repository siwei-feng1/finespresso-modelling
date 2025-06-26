import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import os
import logging
import math
from datetime import datetime
import sys
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.db.model_db_util import save_regression_results

def setup_logger(name: str) -> logging.Logger:
    """
    Configure logging for the regression training module.
    
    Args:
        name (str): Name of the logger.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'regression.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

# Load English spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Set up directories
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
reports_dir = os.path.join(base_dir, 'reports')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def preprocess(text: str) -> str:
    """
    Preprocess text using spaCy for lemmatization and cleaning.
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        str: Preprocessed text with lemmatized tokens, excluding stop words and punctuation.
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def load_data_from_csv() -> pd.DataFrame:
    """
    Load data from enriched CSV file produced by feature engineering pipeline.
    
    Returns:
        pd.DataFrame: Loaded DataFrame or empty DataFrame if loading fails.
    """
    logger.info("Loading data from enriched CSV file...")
    enriched_file = os.path.join(base_dir, 'data', 'feature_engineering', 'selected_features_data.csv')
    
    try:
        if os.path.exists(enriched_file):
            logger.info(f"Loading data from {enriched_file}")
            df = pd.read_csv(enriched_file)
            logger.info(f"Loaded {len(df)} records from CSV")
            return df
        else:
            logger.error(f"CSV file not found: {enriched_file}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from CSV: {str(e)}")
        logger.exception("Detailed traceback:")
        return pd.DataFrame()

def train_and_save_model_for_event(event: str, df: pd.DataFrame) -> dict:
    """
    Train and save a regression model for a specific event.
    
    Args:
        event (str): Event name to train the model for.
        df (pd.DataFrame): Input DataFrame with features and target.
    
    Returns:
        dict: Dictionary with model metrics and file paths, or None if training fails.
    """
    try:
        event_df = df[df['event'] == event].copy()
        logger.info(f"Processing event: {event}, Number of samples: {len(event_df)}")
        
        if len(event_df) < 10:
            logger.warning(f"Not enough data for event {event}. Skipping.")
            return None

        # Text selection logic
        event_df['text_to_process'] = event_df.apply(
            lambda row: (row['content'] if pd.notna(row['content']) and row['content'] != ''
                       else row['title'] if pd.notna(row['title']) and row['title'] != ''
                       else ''),
            axis=1
        )
        
        # Remove rows with empty text
        event_df = event_df[event_df['text_to_process'] != '']
        logger.info(f"Number of samples after removing empty text for event {event}: {len(event_df)}")
        
        if len(event_df) < 10:
            logger.warning(f"Not enough valid text data for event {event} after filtering. Skipping.")
            return None

        event_df['processed_content'] = event_df['text_to_process'].apply(preprocess)

        # Use price_change_percentage as the target variable
        y = event_df['price_change_percentage']

        # Define feature columns
        exclude_cols = ['event', 'content', 'title', 'actual_side', 'price_change_percentage', 'text_to_process', 'processed_content']
        feature_cols = [col for col in event_df.columns if col not in exclude_cols]
        
        if not feature_cols:
            logger.warning(f"No valid features available for event {event}. Skipping.")
            return None

        # Create and save TF-IDF vectorizer separately
        vectorizer = TfidfVectorizer(max_features=1000)
        X_text = vectorizer.fit_transform(event_df['processed_content'])
        vectorizer_filename = os.path.join(models_dir, f'{event.replace(" ", "_").lower()}_tfidf_vectorizer_regression.joblib')
        joblib.dump(vectorizer, vectorizer_filename)
        logger.info(f"Saved TF-IDF vectorizer to {vectorizer_filename}")

        # Combine numerical, binary, categorical, and TF-IDF features
        from scipy.sparse import hstack
        X = event_df[feature_cols]
        X_combined = hstack([X[feature_cols], X_text])

        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

        # Train regressor
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Save model
        model_filename = os.path.join(models_dir, f'{event.replace(" ", "_").lower()}_regression.joblib')
        joblib.dump(model, model_filename)
        logger.info(f"Saved model to {model_filename}")

        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mse)

        logger.info(f"Model trained successfully for event: {event}")
        logger.info(f"MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {
            'event': event,
            'language': 'en',
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'test_sample': len(y_test),
            'training_sample': len(y_train),
            'total_sample': len(event_df),
            'model_filename': model_filename,
            'vectorizer_filename': vectorizer_filename
        }

    except Exception as e:
        logger.error(f"Error processing event {event}: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def train_and_save_all_events_model(df: pd.DataFrame) -> dict:
    """
    Train and save a regression model for all events combined.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and target.
    
    Returns:
        dict: Dictionary with model metrics and file paths, or None if training fails.
    """
    try:
        logger.info(f"Processing all events model, Number of samples: {len(df)}")
        
        if len(df) < 10:
            logger.warning("Not enough data for all events model. Skipping.")
            return None

        # Text selection logic
        df['text_to_process'] = df.apply(
            lambda row: (row['content'] if pd.notna(row['content']) and row['content'] != ''
                       else row['title'] if pd.notna(row['title']) and row['title'] != ''
                       else ''),
            axis=1
        )
        
        # Remove rows with empty text
        df = df[df['text_to_process'] != '']
        logger.info(f"Number of samples after removing empty text: {len(df)}")
        
        if len(df) < 10:
            logger.warning("Not enough valid text data for all events model after filtering. Skipping.")
            return None

        df['processed_content'] = df['text_to_process'].apply(preprocess)

        # Use price_change_percentage as the target variable
        y = df['price_change_percentage']

        # Define feature columns
        exclude_cols = ['event', 'content', 'title', 'actual_side', 'price_change_percentage', 'text_to_process', 'processed_content']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No valid features available for all events model. Skipping.")
            return None

        # Create and save TF-IDF vectorizer separately
        vectorizer = TfidfVectorizer(max_features=1000)
        X_text = vectorizer.fit_transform(df['processed_content'])
        vectorizer_filename = os.path.join(models_dir, 'all_events_tfidf_vectorizer_regression.joblib')
        joblib.dump(vectorizer, vectorizer_filename)
        logger.info(f"Saved TF-IDF vectorizer to {vectorizer_filename}")

        # Combine numerical, binary, categorical, and TF-IDF features
        from scipy.sparse import hstack
        X = df[feature_cols]
        X_combined = hstack([X[feature_cols], X_text])

        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

        # Train regressor
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Save model
        model_filename = os.path.join(models_dir, 'all_events_regression.joblib')
        joblib.dump(model, model_filename)
        logger.info(f"Saved model to {model_filename}")

        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mse)

        logger.info("All events model trained successfully")
        logger.info(f"MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {
            'event': 'all_events',
            'language': 'en',
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'test_sample': len(y_test),
            'training_sample': len(y_train),
            'total_sample': len(df),
            'model_filename': model_filename,
            'vectorizer_filename': vectorizer_filename
        }

    except Exception as e:
        logger.error(f"Error processing all events model: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def train_models_per_event(df: pd.DataFrame) -> list[dict]:
    """
    Train regression models for each unique event in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and target.
    
    Returns:
        list[dict]: List of dictionaries containing model metrics and file paths.
    """
    results = []
    if 'event' not in df.columns:
        logger.error("Event column missing in dataset. Skipping per-event training.")
        return results
    for event in df['event'].unique():
        try:
            result = train_and_save_model_for_event(event, df)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"Error training/saving model for event '{event}': {e}")
            logger.exception("Detailed traceback:")
    return results

def process_results(results: list[dict], df: pd.DataFrame):
    """
    Process and save model results to CSV and database.
    
    Args:
        results (list[dict]): List of model result dictionaries.
        df (pd.DataFrame): Input DataFrame for calculating event counts.
    """
    try:
        valid_results = [r for r in results if r['mse'] is not None]
        
        results_df = pd.DataFrame(valid_results)
        
        # Calculate event counts
        event_counts = df.groupby('event').size().to_dict() if 'event' in df.columns else {'all_events': len(df)}
        results_df['total_sample'] = results_df.apply(
            lambda x: event_counts.get(x['event'], len(df)), 
            axis=1
        )
        
        results_df = results_df.sort_values(by='r2', ascending=False)
        
        # Ensure correct data types
        results_df['event'] = results_df['event'].astype(str)
        results_df['language'] = results_df['language'].astype(str)
        results_df['model_filename'] = results_df['model_filename'].astype(str)
        results_df['vectorizer_filename'] = results_df['vectorizer_filename'].astype(str)
        
        float_columns = ['mse', 'r2', 'mae', 'rmse']
        for col in float_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(float)
        
        int_columns = ['test_sample', 'training_sample', 'total_sample']
        for col in int_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)
        
        results_df = results_df.replace({np.nan: None})
        
        # Save to CSV
        results_csv = os.path.join(reports_dir, 'model_results_regression_after_features_eng.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f'Successfully wrote results to {results_csv}')
        
        # Save to database
        success = save_regression_results(results_df)
        if success:
            logger.info('Successfully wrote results to database')
        else:
            logger.error('Failed to write results to database')
        
        logger.info(f'Average R2 score: {results_df["r2"].mean():.4f}')
        
    except Exception as e:
        logger.error(f"Error processing/saving results: {e}")
        logger.exception("Detailed traceback:")

def main():
    """
    Main function to orchestrate regression training.
    """
    logger.info("Starting regression training")
    
    # Load data
    df = load_data_from_csv()
    
    logger.info(f"Shape of DataFrame: {df.shape}")
    logger.info(f"Columns in DataFrame: {df.columns.tolist()}")
    
    if df.empty:
        logger.error("No data loaded from CSV files. Please check the data files.")
        return
    
    # Log target distribution
    logger.info(f"Price change percentage statistics:\n{df['price_change_percentage'].describe()}")
    
    # Define required columns
    required_columns = ['event', 'content', 'title', 'price_change_percentage']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return
    
    logger.info("All required columns are present")
    
    # Remove rows with missing required columns
    df = df.dropna(subset=required_columns)
    logger.info(f"Shape after removing null values: {df.shape}")
    
    # Log data statistics
    logger.info(f"Number of unique events: {df['event'].nunique()}")
    logger.info(f"Event value counts:\n{df['event'].value_counts()}")
    logger.info(f"Price change percentage statistics:\n{df['price_change_percentage'].describe()}")
    
    # Train models
    logger.info("Starting to train models for each event")
    results = train_models_per_event(df)
    logger.info(f"Number of events processed: {len(results)}")

    logger.info("Training all events model")
    all_events_result = train_and_save_all_events_model(df)
    if all_events_result:
        results.append(all_events_result)
        logger.info("All events model added to results")
    else:
        logger.warning("Failed to train all events model")

    if not results:
        logger.warning("No models were trained. Check the data and event filtering.")
        return

    # Process and save results
    logger.info("Processing and saving results")
    process_results(results, df)

    logger.info("Regression training completed")

if __name__ == '__main__':
    main()

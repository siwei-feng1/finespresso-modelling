import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time
import numpy as np
import os
import math
import sys
import logging
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.db.model_db_util import save_regression_results
from utils.ai.language_util import spacy_model_mapping

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'regression.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger(__name__)

# Keep only the English model
nlp = spacy.load("en_core_web_sm")

# Ensure reports and models directories exist at the top-level
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports')
os.makedirs(reports_dir, exist_ok=True)
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
os.makedirs(models_dir, exist_ok=True)

def preprocess(text):
    """Preprocess text using spaCy for lemmatization and cleaning."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def load_data_from_csv():
    """Load data from cleaned CSV file produced by data quality pipeline."""
    logger.info("Loading data from cleaned CSV file...")
    
    try:
        # Load cleaned price moves data
        price_moves_file = 'data/clean/clean_price_moves.csv'
        if os.path.exists(price_moves_file):
            logger.info(f"Loading data from {price_moves_file}")
            df = pd.read_csv(price_moves_file)
            logger.info(f"Loaded {len(df)} records from CSV")
            return df
        else:
            logger.error(f"CSV file not found: {price_moves_file}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading data from CSV: {str(e)}")
        logger.exception("Detailed traceback:")
        return pd.DataFrame()

def train_and_save_model_for_event(event, df):
    try:
        event_df = df[df['event'] == event].copy()
        logger.info(f"Processing event: {event}, Number of samples: {len(event_df)}")
        
        if len(event_df) < 10:
            logger.warning(f"Not enough data for event {event}. Skipping.")
            return None

        # Update text selection logic to use available columns
        event_df['text_to_process'] = event_df.apply(
            lambda row: (row['content'] if pd.notna(row['content']) and row['content'] != ''
                       else row['title'] if pd.notna(row['title']) and row['title'] != ''
                       else ''),
            axis=1
        )
        
        # Remove rows with empty text
        event_df = event_df[event_df['text_to_process'] != '']
        
        if len(event_df) < 10:
            logger.warning(f"Not enough valid text data for event {event}. Skipping.")
            return None

        event_df['processed_content'] = event_df['text_to_process'].apply(preprocess)
        
        # Use price_change_percentage as target
        y = event_df['price_change_percentage']

        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(event_df['processed_content'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Save model and vectorizer
        model_filename = os.path.join(models_dir, f'{event.replace(" ", "_").lower()}_regression.joblib')
        vectorizer_filename = os.path.join(models_dir, f'{event.replace(" ", "_").lower()}_tfidf_vectorizer_regression.joblib')

        joblib.dump(model, model_filename)
        joblib.dump(tfidf, vectorizer_filename)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mse)

        logger.info(f"Model trained successfully for event: {event}")
        logger.info(f"MSE: {mse}, R2: {r2}, MAE: {mae}, RMSE: {rmse}")

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

def train_and_save_all_events_model(df):
    try:
        logger.info(f"Processing all events model, Number of samples: {len(df)}")
        
        if len(df) < 10:
            logger.warning("Not enough data for all events model. Skipping.")
            return None

        # Update text selection logic to use available columns
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

        y = df['price_change_percentage']

        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(df['processed_content'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Save model and vectorizer
        model_filename = os.path.join(models_dir, 'all_events_regression.joblib')
        vectorizer_filename = os.path.join(models_dir, 'all_events_tfidf_vectorizer_regression.joblib')

        joblib.dump(model, model_filename)
        joblib.dump(tfidf, vectorizer_filename)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mse)

        logger.info("All events model trained successfully")
        logger.info(f"MSE: {mse}, R2: {r2}, MAE: {mae}, RMSE: {rmse}")

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

def train_models_per_event(df):
    results = []
    if df['event'].nunique() == 1 and df['event'].iloc[0] == 'Other':
        logger.info("Only 'Other' event present, skipping per-event training")
        return results
    for event in df['event'].unique():
        try:
            result = train_and_save_model_for_event(event, df)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"Error training/saving model for event '{event}': {e}")
    return results

def process_results(results, df):
    try:
        valid_results = [r for r in results if r['mse'] is not None]
        
        results_df = pd.DataFrame(valid_results)
        event_counts = df['event'].value_counts().to_dict()
        results_df['total_sample'] = results_df['event'].map(event_counts)
        results_df = results_df.sort_values(by='r2', ascending=False)
        
        # Ensure correct data types with safeguards
        results_df['event'] = results_df['event'].astype(str)
        
        # Handle potential non-finite values for float columns
        float_columns = ['mse', 'r2', 'mae', 'rmse']
        for col in float_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(float)
        
        # Handle potential non-finite values for integer columns
        int_columns = ['test_sample', 'training_sample', 'total_sample']
        for col in int_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)
        
        # Replace NaN values with None for database compatibility
        results_df = results_df.replace({np.nan: None})
        
        # Save to CSV
        results_csv = os.path.join(reports_dir, 'model_results_regression_after_cleaning.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f'Successfully wrote results to {results_csv}')
        
        # Save results to the database
        success = save_regression_results(results_df)
        if success:
            logger.info('Successfully wrote results to database')
        else:
            logger.error('Failed to write results to database')
        
        logger.info(f'Average R2 score: {results_df["r2"].mean()}')
    except Exception as e:
        logger.error(f"Error processing/saving results: {e}")
        logger.exception("Detailed traceback:")

def main():
    logger.info("Starting main function")
    
    # Load data from CSV files instead of database
    logger.info("Loading data from CSV files")
    merged_df = load_data_from_csv()
    
    logger.info(f"Shape of merged_df: {merged_df.shape}")
    logger.info(f"Columns in merged_df: {merged_df.columns.tolist()}")
    
    if merged_df.empty:
        logger.error("No data loaded from CSV files. Please check the data files.")
        return
    
    # Ensure all required columns are present
    required_columns = ['id', 'content', 'title', 'event', 'price_change_percentage', 'daily_alpha']
    missing_columns = [col for col in required_columns if col not in merged_df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {merged_df.columns.tolist()}")
        return
    
    logger.info("All required columns are present")
    
    # Remove rows with null values in required columns
    merged_df = merged_df.dropna(subset=required_columns)
    logger.info(f"Shape after removing null values: {merged_df.shape}")
    
    # Print some statistics about the data
    logger.info(f"Daily alpha statistics:\n{merged_df['daily_alpha'].describe()}")
    logger.info(f"Price change percentage statistics:\n{merged_df['price_change_percentage']}\n")
    
    logger.info("Starting to train models for each event")
    # Train models for each event and save them
    results = train_models_per_event(merged_df)
    logger.info(f"Number of events processed: {len(results)}")

    logger.info("Training all events model")
    all_events_result = train_and_save_all_events_model(merged_df)
    if all_events_result:
        results.append(all_events_result)
        logger.info("All events model added to results")
    else:
        logger.warning("Failed to train all events model")

    if not results:
        logger.warning("No models were trained. Check the data and event filtering.")
        return None

    logger.info("Processing and saving results")
    # Process the results and save to a file and database
    process_results(results, merged_df)

    logger.info("Main function completed")

if __name__ == "__main__":
    main()
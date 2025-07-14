import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import time
import numpy as np
import os
import sys
import argparse

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.db.model_db_util import save_results, save_model_to_db, load_model_from_db
from utils.logging.log_util import get_logger
from utils.ai.language_util import spacy_model_mapping
logger = get_logger(__name__)

# Keep only the English model
nlp = spacy.load("en_core_web_sm")

# Ensure reports directory exists at the top-level
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports')
os.makedirs(reports_dir, exist_ok=True)

def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def get_language_code(news_item):
    """Get language code from news item, default to 'en' if not specified"""
    return news_item.get('language', 'en')

def load_data_from_database():
    """Load data from database using news and price moves tables"""
    logger.info("Loading data from database...")
    
    try:
        from utils.db.price_move_db_util import get_price_moves
        
        # Get price moves data (which already includes joined news data)
        logger.info("Loading price moves data from database...")
        df = get_price_moves()
        logger.info(f"Loaded {len(df)} records from database")
        
        if not df.empty:
            logger.info(f"Data columns: {df.columns.tolist()}")
            logger.info(f"Sample data shape: {df.shape}")
            return df
        else:
            logger.warning("No data returned from database")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        logger.exception("Detailed traceback:")
        return pd.DataFrame()

def load_data_from_csv():
    """Load data from CSV files"""
    logger.info("Loading data from CSV files...")
    
    try:
        # Load price moves data (contains the merged news and price data)
        price_moves_file = 'data/all_price_moves.csv'
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

def calculate_directional_metrics(y_true, y_pred):
    """Calculate separate metrics for UP and DOWN predictions"""
    up_mask = y_true == 1
    down_mask = y_true == 0
    
    # Total counts
    total_up = np.sum(up_mask)
    total_down = np.sum(down_mask)
    
    # Correct predictions for each direction
    correct_up = np.sum((y_true == y_pred) & up_mask)
    correct_down = np.sum((y_true == y_pred) & down_mask)
    
    # Calculate percentages
    up_accuracy = (correct_up / total_up * 100) if total_up > 0 else 0
    down_accuracy = (correct_down / total_down * 100) if total_down > 0 else 0
    
    # Calculate percentage distribution of predictions
    total_predictions = len(y_pred)
    up_predictions = np.sum(y_pred == 1)
    down_predictions = np.sum(y_pred == 0)
    
    up_pred_pct = (up_predictions / total_predictions * 100) if total_predictions > 0 else 0
    down_pred_pct = (down_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    return {
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy,
        'total_up': int(total_up),
        'total_down': int(total_down),
        'correct_up': int(correct_up),
        'correct_down': int(correct_down),
        'up_predictions_pct': up_pred_pct,
        'down_predictions_pct': down_pred_pct
    }

def train_and_save_model_for_event(event, df):
    try:
        event_df = df[df['event'] == event].copy()
        logger.info(f"Processing event: {event}, Number of samples: {len(event_df)}")
        
        # Filter out 'unknown' values
        event_df = event_df[event_df['actual_side'].isin(['UP', 'DOWN'])]
        
        if len(event_df) < 10:
            logger.warning(f"Not enough data for event {event} after filtering. Skipping.")
            return None

        # Update text selection logic with priority order
        event_df['text_to_process'] = event_df.apply(
            lambda row: (row['content_en'] if pd.notna(row['content_en']) and row['content_en'] != '' 
                       else row['title_en'] if pd.notna(row['title_en']) and row['title_en'] != ''
                       else row['content'] if pd.notna(row['content']) and row['content'] != ''
                       else row['title']),
            axis=1
        )
        
        event_df['processed_content'] = event_df['text_to_process'].apply(preprocess)
        
        # Use actual_side as the target variable
        y = event_df['actual_side'].map({'UP': 1, 'DOWN': 0})
        
        if len(y.unique()) < 2:
            logger.warning(f"Only one class present in the target variable for event {event}. Skipping.")
            return None

        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(event_df['processed_content'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Check if test set has only one class
        if len(np.unique(y_test)) < 2:
            logger.warning(f"Test set for event {event} has only one class. Skipping this event.")
            return None

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Save model and vectorizer to database
        event_name = event.replace(" ", "_").lower()
        
        # Save classifier model
        model_success, model_version = save_model_to_db(
            model, 
            f'{event_name}_classifier_binary', 
            event, 
            'classifier_binary'
        )
        
        # Save vectorizer
        vectorizer_success, vectorizer_version = save_model_to_db(
            tfidf, 
            f'{event_name}_tfidf_vectorizer_binary', 
            event, 
            'vectorizer'
        )

        if not model_success or not vectorizer_success:
            logger.error(f"Failed to save models to database for event {event}")
            return None

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0

        # Calculate directional metrics
        directional_metrics = calculate_directional_metrics(y_test, y_pred)
        
        logger.info(f"Model trained successfully for event: {event}")
        logger.info(f"Model version: {model_version}, Vectorizer version: {vectorizer_version}")
        logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC-ROC: {auc_roc}")
        logger.info(f"UP accuracy: {directional_metrics['up_accuracy']:.2f}%, "
                   f"DOWN accuracy: {directional_metrics['down_accuracy']:.2f}%")
        logger.info(f"Predictions distribution - UP: {directional_metrics['up_predictions_pct']:.2f}%, "
                   f"DOWN: {directional_metrics['down_predictions_pct']:.2f}%")

        return {
            'event': event,
            'language': 'en',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'test_sample': len(y_test),
            'training_sample': len(y_train),
            'total_sample': len(event_df),
            'model_version': model_version,
            'vectorizer_version': vectorizer_version,
            'up_accuracy': directional_metrics['up_accuracy'],
            'down_accuracy': directional_metrics['down_accuracy'],
            'total_up': directional_metrics['total_up'],
            'total_down': directional_metrics['total_down'],
            'correct_up': directional_metrics['correct_up'],
            'correct_down': directional_metrics['correct_down'],
            'up_predictions_pct': directional_metrics['up_predictions_pct'],
            'down_predictions_pct': directional_metrics['down_predictions_pct']
        }

    except Exception as e:
        logger.error(f"Error processing event {event}: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def train_and_save_all_events_model(df):
    try:
        logger.info(f"Processing all events model, Number of samples: {len(df)}")
        
        # Filter out 'unknown' values
        df = df[df['actual_side'].isin(['UP', 'DOWN'])].copy()
        
        if len(df) < 10:
            logger.warning("Not enough data for all events model after filtering. Skipping.")
            return None

        # Update text selection logic with priority order
        df['text_to_process'] = df.apply(
            lambda row: (row['content_en'] if pd.notna(row['content_en']) and row['content_en'] != '' 
                       else row['title_en'] if pd.notna(row['title_en']) and row['title_en'] != ''
                       else row['content'] if pd.notna(row['content']) and row['content'] != ''
                       else row['title']),
            axis=1
        )
        df['processed_content'] = df['text_to_process'].apply(preprocess)
        
        # Use actual_side as the target variable
        y = df['actual_side'].map({'UP': 1, 'DOWN': 0})
        
        if len(y.unique()) < 2:
            logger.warning("Only one class present in the target variable for all events. Skipping.")
            return None

        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(df['processed_content'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Save model and vectorizer to database
        # Save classifier model
        model_success, model_version = save_model_to_db(
            model, 
            'all_events_classifier_binary', 
            'all_events', 
            'classifier_binary'
        )
        
        # Save vectorizer
        vectorizer_success, vectorizer_version = save_model_to_db(
            tfidf, 
            'all_events_tfidf_vectorizer_binary', 
            'all_events', 
            'vectorizer'
        )

        if not model_success or not vectorizer_success:
            logger.error("Failed to save all events models to database")
            return None

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Calculate directional metrics
        directional_metrics = calculate_directional_metrics(y_test, y_pred)

        logger.info("All events model trained successfully")
        logger.info(f"Model version: {model_version}, Vectorizer version: {vectorizer_version}")
        logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC-ROC: {auc_roc}")
        logger.info(f"UP accuracy: {directional_metrics['up_accuracy']:.2f}%, "
                   f"DOWN accuracy: {directional_metrics['down_accuracy']:.2f}%")
        logger.info(f"Predictions distribution - UP: {directional_metrics['up_predictions_pct']:.2f}%, "
                   f"DOWN: {directional_metrics['down_predictions_pct']:.2f}%")

        result = {
            'event': 'all_events',
            'language': 'en',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'test_sample': len(y_test),
            'training_sample': len(y_train),
            'total_sample': len(df),
            'model_version': model_version,
            'vectorizer_version': vectorizer_version,
            'up_accuracy': directional_metrics['up_accuracy'],
            'down_accuracy': directional_metrics['down_accuracy'],
            'total_up': directional_metrics['total_up'],
            'total_down': directional_metrics['total_down'],
            'correct_up': directional_metrics['correct_up'],
            'correct_down': directional_metrics['correct_down'],
            'up_predictions_pct': directional_metrics['up_predictions_pct'],
            'down_predictions_pct': directional_metrics['down_predictions_pct']
        }
        
        return result

    except Exception as e:
        logger.error(f"Error processing all events model: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def train_models_per_event(df):
    results = []
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
        valid_results = [r for r in results if r['accuracy'] is not None]
        
        results_df = pd.DataFrame(valid_results)
        
        # Calculate event counts without language grouping
        event_counts = df.groupby('event').size().to_dict()
        results_df['total_sample'] = results_df.apply(
            lambda x: event_counts.get(x['event'], 0), 
            axis=1
        )
        
        results_df = results_df.sort_values(by='accuracy', ascending=False)
        
        # Ensure correct data types with safeguards
        results_df['event'] = results_df['event'].astype(str)
        
        # Handle potential non-finite values for float columns
        float_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        for col in float_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(float)
        
        # Handle potential non-finite values for integer columns
        int_columns = ['test_sample', 'training_sample', 'total_sample']
        for col in int_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)
        
        # Replace NaN values with None for database compatibility
        results_df = results_df.replace({np.nan: None})
        
        # Save to CSV
        results_csv = os.path.join(reports_dir, 'model_results_binary.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f'Successfully wrote results to {results_csv}')
        
        # Save results to the database
        success = save_results(results_df)
        if success:
            logger.info('Successfully wrote results to database')
        else:
            logger.error('Failed to write results to database')
        
        logger.info(f'Average accuracy score: {results_df["accuracy"].mean()}')
    except Exception as e:
        logger.error(f"Error processing/saving results: {e}")
        logger.exception("Detailed traceback:")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train classifier models')
    parser.add_argument('--source', choices=['db', 'csv'], default='db',
                       help='Data source: "db" for database or "csv" for CSV files (default: db)')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of samples per event type for few-shot learning (default: 5)')
    args = parser.parse_args()
    
    logger.info("Starting main function")
    logger.info(f"Using data source: {args.source}")
    
    # Load data based on source choice
    if args.source == 'db':
        logger.info("Loading data from database")
        merged_df = load_data_from_database()
    else:
        logger.info("Loading data from CSV files")
        merged_df = load_data_from_csv()
    
    logger.info(f"Shape of merged_df: {merged_df.shape}")
    logger.info(f"Columns in merged_df: {merged_df.columns.tolist()}")
    
    if merged_df.empty:
        logger.error("No data loaded from CSV files. Please check the data files.")
        return
    
    logger.info(f"Value counts of actual_side: {merged_df['actual_side'].value_counts(dropna=False).to_dict()}")
    logger.info(f"Number of non-null actual_side values: {merged_df['actual_side'].notnull().sum()}")
    logger.info(f"Number of null actual_side values: {merged_df['actual_side'].isnull().sum()}")
    
    # Print out counts for each unique value in actual_side
    for value in merged_df['actual_side'].unique():
        count = (merged_df['actual_side'] == value).sum()
        logger.info(f"Count of '{value}' in actual_side: {count}")
    
    # Ensure all required columns are present
    required_columns = ['id', 'content', 'actual_side']
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
    logger.info(f"Number of unique events: {merged_df['event'].nunique()}")
    logger.info(f"Event value counts:\n{merged_df['event'].value_counts()}")   
    logger.info(f"Actual side value counts:\n{merged_df['actual_side'].value_counts()}")
    logger.info(f"Number of rows with actual_side as 'UP' or 'DOWN': {merged_df['actual_side'].isin(['UP', 'DOWN']).sum()}")
    
    if merged_df['actual_side'].isin(['UP', 'DOWN']).sum() == 0:
        logger.error("No valid 'UP' or 'DOWN' values in actual_side column. Cannot train models.")
        return
    
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
        return

    logger.info("Processing and saving results")
    # Process the results and save to a file and database
    process_results(results, merged_df)

    logger.info("Main function completed")

if __name__ == '__main__':
    main()

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import numpy as np
import os
import sys
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sqlalchemy import Float, create_engine, Table, Column, Integer, String, MetaData, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text

# Load environment variables
load_dotenv()

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Ensure reports and models directories exist
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'reports_training_classification')
os.makedirs(reports_dir, exist_ok=True)
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
os.makedirs(models_dir, exist_ok=True)

from utils.db.model_db_util import save_results
from utils.logging.log_util import get_logger
from utils.ai.language_util import spacy_model_mapping
logger = get_logger(__name__)

# Load English spaCy model
nlp = spacy.load("en_core_web_sm")

# Set MLflow tracking URI from .env
mlflow_tracking_uri = os.getenv('DATABASE_URL')
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI set to {mlflow_tracking_uri}")
else:
    logger.error("DATABASE_URL not found in .env file, cannot proceed with MLflow tracking")
    sys.exit(1)

# Database setup
try:
    engine = create_engine(os.getenv('DATABASE_URL'))
except OperationalError as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    sys.exit(1)
metadata = MetaData()

# Custom model versions table to avoid conflict with MLflow
custom_model_versions_table = Table(
    'custom_model_versions', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('event_name', String),
    Column('language', String),
    Column('mlflow_run_id', String),
    Column('mlflow_model_uri', String),
    Column('model_filename', String),
    Column('vectorizer_filename', String),
    Column('accuracy', Float),
    Column('precision', Float),
    Column('recall', Float),
    Column('f1_score', Float),
    Column('auc_roc', Float)
)

# Predictions table
predictions_table = Table(
    'predictions', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('data_id', Integer),
    Column('event_name', String),
    Column('actual', String),
    Column('predicted', String),
    Column('text', String)
)

def initialize_database():
    """Initialize database schema, create tables if they don't exist"""
    try:
        with engine.connect() as connection:
            inspector = inspect(engine)
            # Check and create custom_model_versions table
            if not inspector.has_table(custom_model_versions_table.name):
                logger.info(f"Creating table {custom_model_versions_table.name}")
                custom_model_versions_table.create(bind=engine)
            else:
                logger.info(f"Table {custom_model_versions_table.name} already exists")
                # Verify required columns
                existing_columns = {col['name'] for col in inspector.get_columns(custom_model_versions_table.name)}
                required_columns = {c.name for c in custom_model_versions_table.columns}
                missing_columns = required_columns - existing_columns
                if missing_columns:
                    logger.info(f"Adding missing columns to {custom_model_versions_table.name}: {missing_columns}")
                    with connection.begin():
                        for col_name in missing_columns:
                            col_type = next(c.type for c in custom_model_versions_table.columns if c.name == col_name)
                            connection.execute(text(f"ALTER TABLE {custom_model_versions_table.name} ADD COLUMN {col_name} {col_type}"))

            # Check and create predictions table
            if not inspector.has_table(predictions_table.name):
                logger.info(f"Creating table {predictions_table.name}")
                predictions_table.create(bind=engine)
            else:
                logger.info(f"Table {predictions_table.name} already exists")
                # Verify required columns
                existing_columns = {col['name'] for col in inspector.get_columns(predictions_table.name)}
                required_columns = {c.name for c in predictions_table.columns}
                missing_columns = required_columns - existing_columns
                if missing_columns:
                    logger.info(f"Adding missing columns to {predictions_table.name}: {missing_columns}")
                    with connection.begin():
                        for col_name in missing_columns:
                            col_type = next(c.type for c in predictions_table.columns if c.name == col_name)
                            connection.execute(text(f"ALTER TABLE {predictions_table.name} ADD COLUMN {col_name} {col_type}"))
                # Verify column types for actual and predicted
                for col_name in ['actual', 'predicted']:
                    col_info = next((col for col in inspector.get_columns(predictions_table.name) if col['name'] == col_name), None)
                    if col_info and col_info['type'].__class__.__name__ != 'String':
                        logger.info(f"Converting {col_name} to TEXT in {predictions_table.name}")
                        with connection.begin():
                            connection.execute(text(f"ALTER TABLE {predictions_table.name} ALTER COLUMN {col_name} TYPE TEXT"))

            logger.info("Database tables initialized successfully")
    except OperationalError as e:
        logger.error(f"Failed to initialize database schema: {str(e)}")
        logger.exception("Detailed traceback:")
        sys.exit(1)

# Initialize database schema
initialize_database()

def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def load_data_from_csv():
    """Load data from CSV file"""
    logger.info("Loading data from CSV file...")
    
    try:
        price_moves_file = os.getenv('DATA_FILE_PATH', 'data/feature_engineering/selected_features_data.csv')
        if os.path.exists(price_moves_file):
            logger.info(f"Loading data from {price_moves_file}")
            df = pd.read_csv(price_moves_file)
            logger.info(f"Loaded {len(df)} records from CSV")
            logger.info(f"Columns in CSV: {df.columns.tolist()}")
            logger.info(f"Sample data:\n{df.head(2).to_string()}")
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
    
    total_up = np.sum(up_mask)
    total_down = np.sum(down_mask)
    
    correct_up = np.sum((y_true == y_pred) & up_mask)
    correct_down = np.sum((y_true == y_pred) & down_mask)
    
    up_accuracy = (correct_up / total_up * 100) if total_up > 0 else 0
    down_accuracy = (correct_down / total_down * 100) if total_down > 0 else 0
    
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
        logger.info(f"After filtering out 'unknown', samples for {event}: {len(event_df)}")
        
        if len(event_df) < 10:
            logger.warning(f"Not enough data for event {event} after filtering. Skipping.")
            return None, None

        # Text selection logic
        event_df['text_to_process'] = event_df.apply(
            lambda row: (
                row['content'] if pd.notna(row['content']) and row['content'] != ''
                else row['title'] if pd.notna(row['title']) and row['title'] != ''
                else ''
            ),
            axis=1
        )
        
        # Remove rows with empty text
        event_df = event_df[event_df['text_to_process'] != '']
        logger.info(f"After removing empty text, samples for {event}: {len(event_df)}")
        if len(event_df) < 10:
            logger.warning(f"Not enough valid text samples for event {event}. Skipping.")
            return None, None

        event_df['processed_content'] = event_df['text_to_process'].apply(preprocess)
        
        # Use actual_side as the target variable
        y = event_df['actual_side'].map({'UP': 1, 'DOWN': 0})
        
        if len(y.unique()) < 2:
            logger.warning(f"Only one class present in the target variable for event {event}. Skipping.")
            return None, None

        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(event_df['processed_content'])

        # Ensure indices are aligned
        event_df = event_df.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, event_df.index, test_size=0.2, random_state=42
        )
        
        # Check if test set has only one class
        if len(np.unique(y_test)) < 2:
            logger.warning(f"Test set for event {event} has only one class. Skipping this event.")
            return None, None

        # Debug shapes
        logger.info(f"y_test shape: {y_test.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"event_df shape after filtering: {event_df.shape}")
        logger.info(f"Test indices length: {len(test_idx)}")

        # Prepare input example for MLflow
        input_example = X_train[:1].toarray()

        # Start MLflow run
        with mlflow.start_run(run_name=f"{event}_classifier") as run:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Check for NaN in predictions
            if np.any(np.isnan(y_pred)):
                logger.warning(f"NaN values detected in predictions for event {event}. Replacing with 'DOWN' (0).")
                y_pred = np.where(np.isnan(y_pred), 0, y_pred)

            # Calculate standard metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0

            # Calculate directional metrics
            directional_metrics = calculate_directional_metrics(y_test, y_pred)
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metric("up_accuracy", directional_metrics['up_accuracy'])
            mlflow.log_metric("down_accuracy", directional_metrics['down_accuracy'])
            mlflow.log_metric("up_predictions_pct", directional_metrics['up_predictions_pct'])
            mlflow.log_metric("down_predictions_pct", directional_metrics['down_predictions_pct'])

            # Log model to MLflow with input example
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=f"{event.replace(' ', '_').lower()}_classifier",
                signature=signature,
                input_example=input_example
            )
            
            # Save model and vectorizer locally
            model_filename = os.path.join(models_dir, f'{event.replace(" ", "_").lower()}_classifier_binary.joblib')
            vectorizer_filename = os.path.join(models_dir, f'{event.replace(" ", "_").lower()}_tfidf_vectorizer_binary.joblib')
            joblib.dump(model, model_filename)
            joblib.dump(tfidf, vectorizer_filename)

            # Log model and vectorizer as artifacts
            mlflow.log_artifact(model_filename)
            mlflow.log_artifact(vectorizer_filename)

            # Create predictions DataFrame with aligned indices
            predictions_df = pd.DataFrame({
                'id': event_df.loc[test_idx, 'id'].values if 'id' in event_df.columns else test_idx,
                'event_name': event,
                'actual': y_test.map({1: 'UP', 0: 'DOWN'}),
                'predicted': pd.Series(y_pred).map({1: 'UP', 0: 'DOWN'}),
                'text': event_df.loc[test_idx, 'text_to_process'].values
            }, index=test_idx)

            predictions_csv = os.path.join(reports_dir, f'{event.replace(" ", "_").lower()}_predictions.csv')
            predictions_df.to_csv(predictions_csv, index=False)
            logger.info(f"Saved predictions to {predictions_csv}")
            mlflow.log_artifact(predictions_csv)

            logger.info(f"Model trained successfully for event: {event}")
            logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC-ROC: {auc_roc}")
            logger.info(f"UP accuracy: {directional_metrics['up_accuracy']:.2f}%, "
                       f"DOWN accuracy: {directional_metrics['down_accuracy']:.2f}%")
            logger.info(f"Predictions distribution - UP: {directional_metrics['up_predictions_pct']:.2f}%, "
                       f"DOWN: {directional_metrics['down_predictions_pct']:.2f}%")

            # Prepare model version data
            model_version_data = {
                'event_name': event,
                'language': 'en',
                'mlflow_run_id': run.info.run_id,
                'mlflow_model_uri': f"runs:/{run.info.run_id}/model",
                'model_filename': model_filename,
                'vectorizer_filename': vectorizer_filename,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc
            }

            # Save to custom_model_versions table
            with engine.connect() as connection:
                with connection.begin():  # Use transaction
                    connection.execute(custom_model_versions_table.insert().values(**model_version_data))
                    logger.info(f"Saved model version for event {event} to database")

            # Save predictions to predictions table
            predictions_db = predictions_df[['id', 'event_name', 'actual', 'predicted', 'text']].rename(columns={'id': 'data_id'})
            with engine.connect() as connection:
                with connection.begin():  # Use transaction
                    for _, row in predictions_db.iterrows():
                        connection.execute(predictions_table.insert().values(**row.to_dict()))
                    logger.info(f"Saved predictions for event {event} to database")

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
                'model_filename': model_filename,
                'vectorizer_filename': vectorizer_filename,
                'up_accuracy': directional_metrics['up_accuracy'],
                'down_accuracy': directional_metrics['down_accuracy'],
                'total_up': directional_metrics['total_up'],
                'total_down': directional_metrics['total_down'],
                'correct_up': directional_metrics['correct_up'],
                'correct_down': directional_metrics['correct_down'],
                'up_predictions_pct': directional_metrics['up_predictions_pct'],
                'down_predictions_pct': directional_metrics['down_predictions_pct'],
                'mlflow_run_id': run.info.run_id,
                'mlflow_model_uri': f"runs:/{run.info.run_id}/model"
            }, predictions_df

    except Exception as e:
        logger.error(f"Error processing event {event}: {str(e)}")
        logger.exception("Detailed traceback:")
        return None, None

def train_and_save_all_events_model(df):
    try:
        logger.info(f"Processing all events model, Number of samples: {len(df)}")
        
        # Filter out 'unknown' values
        df = df[df['actual_side'].isin(['UP', 'DOWN'])].copy()
        logger.info(f"After filtering out 'unknown', samples: {len(df)}")
        
        if len(df) < 10:
            logger.warning("Not enough data for all events model after filtering. Skipping.")
            return None, None

        # Text selection logic
        df['text_to_process'] = df.apply(
            lambda row: (
                row['content'] if pd.notna(row['content']) and row['content'] != ''
                else row['title'] if pd.notna(row['title']) and row['title'] != ''
                else ''
            ),
            axis=1
        )
        
        # Remove rows with empty text
        df = df[df['text_to_process'] != '']
        logger.info(f"Number of samples after removing empty text: {len(df)}")
        
        if len(df) < 10:
            logger.warning("Not enough data after text processing. Skipping.")
            return None, None

        df['processed_content'] = df['text_to_process'].apply(preprocess)
        
        # Use actual_side as the target variable
        y = df['actual_side'].map({'UP': 1, 'DOWN': 0})
        
        if len(y.unique()) < 2:
            logger.warning("Only one class present in the target variable for all events. Skipping.")
            return None, None

        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(df['processed_content'])

        # Ensure indices are aligned
        df = df.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )

        # Check if test set has only one class
        if len(np.unique(y_test)) < 2:
            logger.warning("Test set for all events has only one class. Skipping.")
            return None, None

        # Debug shapes
        logger.info(f"y_test shape: {y_test.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"df shape after filtering: {df.shape}")
        logger.info(f"Test indices length: {len(test_idx)}")

        # Prepare input example for MLflow
        input_example = X_train[:1].toarray()

        # Start MLflow run
        with mlflow.start_run(run_name="all_events_classifier") as run:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Check for NaN in predictions
            if np.any(np.isnan(y_pred)):
                logger.warning("NaN values detected in predictions for all events. Replacing with 'DOWN' (0).")
                y_pred = np.where(np.isnan(y_pred), 0, y_pred)

            # Calculate standard metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0

            # Calculate directional metrics
            directional_metrics = calculate_directional_metrics(y_test, y_pred)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metric("up_accuracy", directional_metrics['up_accuracy'])
            mlflow.log_metric("down_accuracy", directional_metrics['down_accuracy'])
            mlflow.log_metric("up_predictions_pct", directional_metrics['up_predictions_pct'])
            mlflow.log_metric("down_predictions_pct", directional_metrics['down_predictions_pct'])

            # Log model to MLflow with input example
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name="all_events_classifier",
                signature=signature,
                input_example=input_example
            )

            # Save model and vectorizer locally
            model_filename = os.path.join(models_dir, 'all_events_classifier_binary.joblib')
            vectorizer_filename = os.path.join(models_dir, 'all_events_tfidf_vectorizer_binary.joblib')
            joblib.dump(model, model_filename)
            joblib.dump(tfidf, vectorizer_filename)

            # Log model and vectorizer as artifacts
            mlflow.log_artifact(model_filename)
            mlflow.log_artifact(vectorizer_filename)

            # Create predictions DataFrame with aligned indices
            predictions_df = pd.DataFrame({
                'id': df.loc[test_idx, 'id'].values if 'id' in df.columns else test_idx,
                'event_name': 'all_events',
                'actual': y_test.map({1: 'UP', 0: 'DOWN'}),
                'predicted': pd.Series(y_pred).map({1: 'UP', 0: 'DOWN'}),
                'text': df.loc[test_idx, 'text_to_process'].values
            }, index=test_idx)

            predictions_csv = os.path.join(reports_dir, 'all_events_predictions.csv')
            predictions_df.to_csv(predictions_csv, index=False)
            logger.info(f"Saved all events predictions to {predictions_csv}")
            mlflow.log_artifact(predictions_csv)

            logger.info("All events model trained successfully")
            logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC-ROC: {auc_roc}")
            logger.info(f"UP accuracy: {directional_metrics['up_accuracy']:.2f}%, "
                       f"DOWN accuracy: {directional_metrics['down_accuracy']:.2f}%")
            logger.info(f"Predictions distribution - UP: {directional_metrics['up_predictions_pct']:.2f}%, "
                       f"DOWN: {directional_metrics['down_predictions_pct']:.2f}%")

            # Prepare model version data
            model_version_data = {
                'event_name': 'all_events',
                'language': 'en',
                'mlflow_run_id': run.info.run_id,
                'mlflow_model_uri': f"runs:/{run.info.run_id}/model",
                'model_filename': model_filename,
                'vectorizer_filename': vectorizer_filename,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc
            }

            # Save to custom_model_versions table
            with engine.connect() as connection:
                with connection.begin():  # Use transaction
                    connection.execute(custom_model_versions_table.insert().values(**model_version_data))
                    logger.info("Saved all events model version to database")

            # Save predictions to predictions table
            predictions_db = predictions_df[['id', 'event_name', 'actual', 'predicted', 'text']].rename(columns={'id': 'data_id'})
            with engine.connect() as connection:
                with connection.begin():  # Use transaction
                    for _, row in predictions_db.iterrows():
                        connection.execute(predictions_table.insert().values(**row.to_dict()))
                    logger.info("Saved all events predictions to database")

            return {
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
                'model_filename': model_filename,
                'vectorizer_filename': vectorizer_filename,
                'up_accuracy': directional_metrics['up_accuracy'],
                'down_accuracy': directional_metrics['down_accuracy'],
                'total_up': directional_metrics['total_up'],
                'total_down': directional_metrics['total_down'],
                'correct_up': directional_metrics['correct_up'],
                'correct_down': directional_metrics['correct_down'],
                'up_predictions_pct': directional_metrics['up_predictions_pct'],
                'down_predictions_pct': directional_metrics['down_predictions_pct'],
                'mlflow_run_id': run.info.run_id,
                'mlflow_model_uri': f"runs:/{run.info.run_id}/model"
            }, predictions_df

    except Exception as e:
        logger.error(f"Error processing all events model: {str(e)}")
        logger.exception("Detailed traceback:")
        return None, None

def train_models_per_event(df):
    results = []
    all_predictions = []
    for event in df['event'].unique():
        try:
            result, predictions_df = train_and_save_model_for_event(event, df)
            if result is not None:
                results.append(result)
                if predictions_df is not None:
                    all_predictions.append(predictions_df)
        except Exception as e:
            logger.error(f"Error training/saving model for event '{event}': {e}")
            logger.exception("Detailed traceback:")
    return results, all_predictions

def process_results(results, all_predictions, df):
    try:
        valid_results = [r for r in results if r['accuracy'] is not None]
        
        results_df = pd.DataFrame(valid_results)
        
        event_counts = df.groupby('event').size().to_dict()
        results_df['total_sample'] = results_df.apply(
            lambda x: event_counts.get(x['event'], 0), 
            axis=1
        )
        
        results_df = results_df.sort_values(by='accuracy', ascending=False)
        
        # Ensure correct data types
        results_df['event'] = results_df['event'].astype(str)
        
        float_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'up_accuracy', 'down_accuracy', 'up_predictions_pct', 'down_predictions_pct']
        for col in float_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(float)
        
        int_columns = ['test_sample', 'training_sample', 'total_sample', 'total_up', 'total_down', 'correct_up', 'correct_down']
        for col in int_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)
        
        # Replace NaN values with None for database compatibility
        results_df = results_df.replace({np.nan: None})
        
        # Save model versions to CSV
        model_versions_df = results_df[[
            'event', 'language', 'mlflow_run_id', 'mlflow_model_uri', 
            'model_filename', 'vectorizer_filename', 'accuracy', 
            'precision', 'recall', 'f1_score', 'auc_roc'
        ]].rename(columns={'event': 'event_name'})
        model_versions_csv = os.path.join(reports_dir, 'model_versions.csv')
        model_versions_df.to_csv(model_versions_csv, index=False)
        logger.info(f'Successfully wrote model versions to {model_versions_csv}')
        mlflow.log_artifact(model_versions_csv)
        
        # Combine all predictions
        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            combined_predictions_csv = os.path.join(reports_dir, 'combined_predictions.csv')
            combined_predictions.to_csv(combined_predictions_csv, index=False)
            logger.info(f'Successfully wrote combined predictions to {combined_predictions_csv}')
            mlflow.log_artifact(combined_predictions_csv)
        
        # Save model results to CSV
        results_csv = os.path.join(reports_dir, 'model_results_binary.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f'Successfully wrote model results to {results_csv}')
        mlflow.log_artifact(results_csv)
        
        # Save results to database
        success = save_results(results_df.rename(columns={'event': 'event_name'}))
        if success:
            logger.info('Successfully wrote results to database')
        else:
            logger.error('Failed to write results to database')
        
        logger.info(f'Average accuracy score: {results_df["accuracy"].mean()}')
    except Exception as e:
        logger.error(f"Error processing/saving results: {e}")
        logger.exception("Detailed traceback:")

def main():
    logger.info("Starting main function")
    
    # Load data from CSV file
    logger.info("Loading data from CSV file")
    merged_df = load_data_from_csv()
    
    logger.info(f"Shape of merged_df: {merged_df.shape}")
    logger.info(f"Columns in merged_df: {merged_df.columns.tolist()}")
    
    if merged_df.empty:
        logger.error("No data loaded from CSV file. Please check the data file.")
        return
    
    # Ensure required columns are present
    required_columns = ['content', 'title', 'event', 'actual_side']
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
    logger.info(f"Value counts of actual_side: {merged_df['actual_side'].value_counts(dropna=False).to_dict()}")
    logger.info(f"Number of unique events: {merged_df['event'].nunique()}")
    logger.info(f"Event value counts:\n{merged_df['event'].value_counts()}")
    logger.info(f"Actual side value counts:\n{merged_df['actual_side'].value_counts()}")
    logger.info(f"Number of rows with actual_side as 'UP' or 'DOWN': {merged_df['actual_side'].isin(['UP', 'DOWN']).sum()}")
    
    if merged_df['actual_side'].isin(['UP', 'DOWN']).sum() == 0:
        logger.error("No valid 'UP' or 'DOWN' values in actual_side column. Cannot train models.")
        return
    
    # Set MLflow experiment
    mlflow.set_experiment("training_binary_classification")
    
    logger.info("Starting to train models for each event")
    # Train models for each event and save them
    results, all_predictions = train_models_per_event(merged_df)
    logger.info(f"Number of events processed: {len(results)}")

    logger.info("Training all events model")
    all_events_result, all_events_predictions = train_and_save_all_events_model(merged_df)
    if all_events_result:
        results.append(all_events_result)
        if all_events_predictions is not None:
            all_predictions.append(all_events_predictions)
        logger.info("All events model added to results")
    else:
        logger.warning("Failed to train all events model")

    if not results:
        logger.warning("No models were trained. Check the data and event filtering.")
        return

    logger.info("Processing and saving results")
    # Process the results and save to CSV files
    process_results(results, all_predictions, merged_df)

    logger.info("Main function completed")

if __name__ == '__main__':
    main()

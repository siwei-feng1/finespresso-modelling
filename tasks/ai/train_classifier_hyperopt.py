import os
import sys
import logging
import pandas as pd
import numpy as np
import spacy
import joblib
import yaml
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from scipy.sparse import hstack
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
import warnings
import random

load_dotenv()

# --- Fix all randomness for reproducibility ---
np.random.seed(42)
random.seed(42)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- Logging setup ---
def setup_logger(name: str) -> logging.Logger:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'classification_hyperopt_mlflow.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

# --- Directories ---
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
reports_dir = os.path.join(base_dir, 'reports')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- spaCy model ---
nlp = spacy.load("en_core_web_sm")

def preprocess(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def load_data_from_csv() -> pd.DataFrame:
    enriched_file = os.path.join(base_dir, 'data', 'feature_engineering', 'final_enriched_data.csv')
    if os.path.exists(enriched_file):
        logger.info(f"Loading data from {enriched_file}")
        df = pd.read_csv(enriched_file)
        logger.info(f"Loaded {len(df)} records from CSV")
        return df
    else:
        logger.error(f"CSV file not found: {enriched_file}")
        return pd.DataFrame()

def calculate_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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

def load_model_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['models']

def get_model_class(module_name, class_name):
    import importlib
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    # If the model supports random_state, inject random_state=42
    def model_with_seed(*args, **kwargs):
        if 'random_state' in model_cls().get_params().keys() and 'random_state' not in kwargs:
            kwargs['random_state'] = 42
        return model_cls(*args, **kwargs)
    return model_with_seed

def build_hyperopt_space(param_config):
    space = {}
    for param, conf in param_config.items():
        if conf['type'] == 'int':
            space[param] = hp.quniform(param, conf['low'], conf['high'], 1)
        elif conf['type'] == 'float':
            space[param] = hp.uniform(param, conf['low'], conf['high'])
        elif conf['type'] == 'categorical':
            space[param] = hp.choice(param, conf['choices'])
    return space

def cast_hyperopt_params(params, param_config):
    # Hyperopt returns float for quniform, so cast to int if needed
    casted = {}
    for param, conf in param_config.items():
        if conf['type'] == 'int':
            casted[param] = int(params[param])
        else:
            casted[param] = params[param]
    return casted

def objective_cv(params, X, y, model_class, param_config, cv=3):
    params = cast_hyperopt_params(params, param_config)
    with mlflow.start_run(nested=True):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        losses = []
        y_np = np.array(y)
        for train_idx, val_idx in skf.split(X, y_np):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]
            model = model_class(**params)
            model.fit(X_train, y_train)
            # Use predict_proba for log_loss
            y_pred_proba = model.predict_proba(X_val)
            loss = log_loss(y_val, y_pred_proba, labels=[0, 1])
            losses.append(loss)
        mean_loss = np.mean(losses)
        mlflow.log_metric('cv_log_loss', float(mean_loss))
        for k, v in params.items():
            mlflow.log_param(k, v)
        return {'loss': mean_loss, 'status': STATUS_OK, 'cv_log_loss': mean_loss, 'params': params}

def train_and_tune_model(event, df, model_cfg, output_prefix):
    event_df = df.copy() if event == 'all_events' else df[df['event'] == event].copy()
    logger.info(f"Processing event: {event}, Number of samples: {len(event_df)}")
    event_df = event_df[event_df['actual_side'].isin(['UP', 'DOWN'])].copy()
    if len(event_df) < 10:
        logger.warning(f"Not enough data for event {event} after filtering. Skipping.")
        return None
    event_df['text_to_process'] = event_df.apply(
        lambda row: (row['content'] if pd.notna(row['content']) and row['content'] != ''
                   else row['title'] if pd.notna(row['title']) and row['title'] != ''
                   else ''),
        axis=1
    )
    event_df = event_df[event_df['text_to_process'] != '']
    if len(event_df) < 10:
        logger.warning(f"Not enough valid text data for event {event} after filtering. Skipping.")
        return None
    event_df['processed_content'] = event_df['text_to_process'].apply(preprocess)
    y = event_df['actual_side'].map({'UP': 1, 'DOWN': 0})
    if len(y.unique()) < 2:
        logger.warning(f"Only one class present in the target variable for event {event}. Skipping.")
        return None
    exclude_cols = ['event', 'content', 'title', 'actual_side', 'price_change_percentage', 'text_to_process', 'processed_content']
    feature_cols = [col for col in event_df.columns if col not in exclude_cols]
    if not feature_cols:
        logger.warning(f"No valid features available for event {event}. Skipping.")
        return None
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(event_df['processed_content'])
    vectorizer_filename = os.path.join(models_dir, f'{output_prefix}_tfidf_vectorizer_binary.joblib')
    joblib.dump(vectorizer, vectorizer_filename)
    logger.info(f"Saved TF-IDF vectorizer to {vectorizer_filename}")
    X = event_df[feature_cols]
    X_combined = hstack([X[feature_cols], X_text])
    # --- Hold-out test set split ---
    class_counts = y.value_counts()
    if (class_counts < 3).any():
        logger.warning(f"Not enough samples per class for event {event}. Skipping.")
        return None
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
    class_counts_trainval = y_trainval.value_counts()
    if (class_counts_trainval < 3).any():
        logger.warning(f"Not enough samples per class for event {event} for cv split. Skipping.")
        return None
    best_result = None
    best_f1 = -1
    best_model = None
    best_params = None
    for model_entry in model_cfg:
        model_name = model_entry['name']
        model_class = get_model_class(model_entry['module'], model_entry['name'])
        param_config = model_entry['params']
        logger.info(f"Starting hyperopt (nested CV) for {model_name} on event {event}")
        mlflow.set_experiment(f"Classifier_{event}")
        with mlflow.start_run(run_name=f"{model_name}_hyperopt_{event}"):
            space = build_hyperopt_space(param_config)
            trials = Trials()
            def hyperopt_objective(params):
                return objective_cv(params, X_trainval, y_trainval, model_class, param_config, cv=3)
            best = fmin(
                fn=hyperopt_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials,
                rstate=np.random.default_rng(42)
            )
            # Get best trial info
            best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
            best_trial = trials.trials[best_trial_idx]
            best_params = cast_hyperopt_params(best_trial['result']['params'], param_config)
            logger.info(f"Best params for {model_name} on {event}: {best_params}")
            # Retrain best model on full trainval set
            best_model_instance = model_class(**best_params)
            best_model_instance.fit(X_trainval, y_trainval)
            # Evaluate on test set
            y_pred = np.array(best_model_instance.predict(X_test))
            y_pred_proba = best_model_instance.predict_proba(X_test)[:, 1] if hasattr(best_model_instance, 'predict_proba') else None
            accuracy = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            auc_roc = float(roc_auc_score(y_test, y_pred_proba)) if y_pred_proba is not None and len(np.unique(y_test)) > 1 else 0.0
            directional_metrics = calculate_directional_metrics(np.array(y_test), y_pred)
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'up_accuracy': float(directional_metrics['up_accuracy']),
                'down_accuracy': float(directional_metrics['down_accuracy']),
                'up_predictions_pct': float(directional_metrics['up_predictions_pct']),
                'down_predictions_pct': float(directional_metrics['down_predictions_pct'])
            })
            mlflow.log_params(best_params)
            # Silence MLflow integer column missing value warning
            warnings.filterwarnings(
                "ignore",
                message=r"Hint: Inferred schema contains integer column\(s\). Integer columns in Python cannot represent missing values.*",
                category=UserWarning,
                module="mlflow.types.utils"
            )
            input_example = X_test[:1]
            signature = infer_signature(X_test, y_test[:1])
            mlflow.sklearn.log_model(best_model_instance, name="model", input_example=input_example, signature=signature)
            mlflow.log_artifact(vectorizer_filename)
            if f1 > best_f1:
                best_f1 = f1
                best_result = {
                    'event': event,
                    'model': model_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc,
                    'test_sample': len(y_test),
                    'training_sample': len(y_trainval),
                    'total_sample': len(event_df),
                    'up_accuracy': float(directional_metrics['up_accuracy']),
                    'down_accuracy': float(directional_metrics['down_accuracy']),
                    'total_up': directional_metrics['total_up'],
                    'total_down': directional_metrics['total_down'],
                    'correct_up': directional_metrics['correct_up'],
                    'correct_down': directional_metrics['correct_down'],
                    'up_predictions_pct': float(directional_metrics['up_predictions_pct']),
                    'down_predictions_pct': float(directional_metrics['down_predictions_pct']),
                    'cv_log_loss': best_trial['result']['cv_log_loss'],
                    'params': best_params
                }
                best_model = best_model_instance
    if best_model is not None and best_result is not None:
        model_filename = os.path.join(models_dir, f'{output_prefix}_{model_name}_classifier.joblib')
        joblib.dump(best_model, model_filename)
        logger.info(f"Saved best model to {model_filename}")
    return best_result

def main():
    logger.info("Starting classifier training with hyperopt and MLflow")
    df = load_data_from_csv()
    if df.empty:
        logger.error("No data loaded from CSV files. Please check the data files.")
        return
    required_columns = ['event', 'content', 'title', 'actual_side']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    df = df.dropna(subset=required_columns)
    logger.info(f"Shape after removing null values: {df.shape}")
    model_cfg = load_model_config(os.path.join(base_dir, 'config', 'model_config.yaml'))
    results = []
    # Per-event models
    for event in df['event'].unique():
        result = train_and_tune_model(event, df, model_cfg, output_prefix=event.replace(" ", "_").lower())
        if result:
            results.append(result)
    # All-events model
    all_events_result = train_and_tune_model('all_events', df, model_cfg, output_prefix="all_events")
    if all_events_result:
        results.append(all_events_result)
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="accuracy", ascending=False)
    results_csv = os.path.join(reports_dir, 'model_results_binary_hyperopt.csv')
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved results summary to {results_csv}")

if __name__ == "__main__":
    main() 
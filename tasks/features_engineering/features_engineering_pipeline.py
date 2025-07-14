import pandas as pd
import os
import logging
from datetime import datetime
from fetch_yfinance_data import enrich_with_yfinance
from time_features import add_time_features
from company_features import add_company_features
from features_selection import select_features

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir,'tasks', 'features_engineering', 'logs', )
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'features_engineering_pipeline.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

def run_feature_engineering_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Run the feature engineering pipeline.

    Args:
        input_path: Path to cleaned input CSV
        output_path: Path to save final enriched CSV

    Returns:
        pd.DataFrame: Enriched DataFrame with selected features
    """
    logger.info("Starting feature engineering pipeline")
    
    # Load cleaned data
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
    except Exception as e:
        logger.error(f"Failed to load {input_path}: {str(e)}")
        raise

    # Step 1: Enrich with Yahoo Finance data
    try:
        df = enrich_with_yfinance(df)
        intermediate_path = os.path.join(os.path.dirname(output_path), 'yfinance_enriched_data.csv')
        os.makedirs(os.path.dirname(intermediate_path), exist_ok=True) 
        df.to_csv(intermediate_path, index=False)
        logger.info(f"Saved Yahoo Finance enriched data to {intermediate_path}")
    except Exception as e:
        logger.error(f"Yahoo Finance enrichment failed: {str(e)}")
        raise

    # Step 2: Add time-based features
    try:
        df = add_time_features(df)
        intermediate_path = os.path.join(os.path.dirname(output_path), 'time_features_data.csv')
        df.to_csv(intermediate_path, index=False)
        logger.info(f"Saved time features data to {intermediate_path}")
    except Exception as e:
        logger.error(f"Time features addition failed: {str(e)}")
        raise

    # Step 3: Add company-specific features
    try:
        df = add_company_features(df)
        intermediate_path = os.path.join(os.path.dirname(output_path), 'company_features_data.csv')
        df.to_csv(intermediate_path, index=False)
        logger.info(f"Saved company features data to {intermediate_path}")
    except Exception as e:
        logger.error(f"Company features addition failed: {str(e)}")
        raise

    # Step 4: Perform feature selection
    try:
        params = {
            'method': 'correlation_and_rf',
            'correlation_threshold': 0.8,
            'importance_threshold': 0.01
        }
        df, selected_features = select_features(df, params)
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise

    # Save final output
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved final enriched data to {output_path} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {str(e)}")
        raise

    logger.info("Feature engineering pipeline completed")
    return df

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(project_root, 'data', 'clean', 'clean_price_moves.csv')
    output_path = os.path.join(project_root, 'data', 'feature_engineering', 'selected_features_data.csv')
    run_feature_engineering_pipeline(input_path, output_path)
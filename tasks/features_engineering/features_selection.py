from datetime import datetime
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs', 'features_engineering')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'features_selection.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

def validate_categorical_columns(df: pd.DataFrame, categorical_cols: List[str]) -> List[str]:
    """Validate categorical columns and log unique values."""
    valid_cols = []
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                valid_cols.append(col)
                logger.info(f"Column {col} has {len(unique_vals)} unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
            else:
                logger.warning(f"Column {col} has no valid values, excluding from encoding")
        else:
            logger.warning(f"Column {col} not found in DataFrame")
    return valid_cols

def select_features(df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Perform feature selection using correlation and random forest importance.
    
    Parameters:
    - df: DataFrame containing features and target
    - params: Dictionary with 'method', 'correlation_threshold', 'importance_threshold'
    
    Returns:
    - df_selected: DataFrame with selected features
    - selected_features: List of selected feature names
    """
    method = params.get('method', 'correlation_and_rf')
    correlation_threshold = params.get('correlation_threshold', 0.8)
    importance_threshold = params.get('importance_threshold', 0.01)
    
    numerical_cols = [
        'market_cap', 'float_shares', 'avg_volume', 'beta', 'recent_volume', 'float_ratio',
        'sector_performance', 'day_of_week', 'hour', 'combined_sentiment', 'prev_news_sentiment',
        'volatility', 'sector_relative_volatility', 'days_since_event'
    ]
    
    categorical_cols = ['event', 'exchange', 'sector', 'industry', 'market_cap_category']
    
    # Validate columns
    available_num_cols = [col for col in numerical_cols if col in df.columns]
    available_cat_cols = validate_categorical_columns(df, categorical_cols)
    
    if not available_num_cols and not available_cat_cols:
        logger.error("No valid features available for selection")
        raise ValueError("No valid features available")
    
    logger.info(f"Available numerical columns: {available_num_cols}")
    logger.info(f"Available categorical columns: {available_cat_cols}")
    
    # Handle missing values
    df[available_num_cols] = df[available_num_cols].fillna(df[available_num_cols].median())
    df[available_cat_cols] = df[available_cat_cols].fillna('Unknown')
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', available_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat_cols)
        ])
    
    try:
        X = preprocessor.fit_transform(df)
        feature_names = available_num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(available_cat_cols))
        logger.info(f"Transformed features: {len(feature_names)} features generated")
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise
    
    # Determine target
    if 'price_change_percentage' in df.columns:
        y = df['price_change_percentage']
        model = RandomForestRegressor(random_state=42)
    elif 'actual_side' in df.columns:
        y = df['actual_side'].map({'UP': 1, 'DOWN': 0})
        model = RandomForestClassifier(random_state=42)
    else:
        logger.error("No valid target column found")
        raise ValueError("No valid target column found")
    
    # Correlation-based selection
    selected_features = feature_names
    if 'correlation' in method:
        try:
            corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            selected_features = [f for f in feature_names if f not in to_drop]
            X = X[:, [i for i, f in enumerate(feature_names) if f in selected_features]]
            logger.info(f"After correlation filter: {len(selected_features)} features remain")
        except Exception as e:
            logger.warning(f"Correlation-based selection failed: {str(e)}. Skipping correlation filter.")
    
    # Random Forest importance-based selection
    if 'rf' in method:
        try:
            model.fit(X, y)
            importances = pd.Series(model.feature_importances_, index=selected_features)
            selected_features = importances[importances > importance_threshold].index.tolist()
            X = X[:, [i for i, f in enumerate(feature_names) if f in selected_features]]
            logger.info(f"After RF importance filter: {len(selected_features)} features remain")
            
            # Save feature importance
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            importance_df = pd.DataFrame({'feature': importances.index, 'importance': importances.values})
            importance_path = os.path.join(base_dir, 'reports', f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Saved feature importance to {importance_path}")
        except Exception as e:
            logger.warning(f"RF importance-based selection failed: {str(e)}. Skipping RF filter.")
    
    # Update DataFrame
    df_selected = df.copy()
    # Keep only original columns and valid selected features
    final_features = [col for col in df.columns if col not in numerical_cols + categorical_cols] + \
                     [f for f in selected_features if f in feature_names]
    # Filter out any features not in DataFrame columns
    final_features = [f for f in final_features if f in df_selected.columns]
    
    if not final_features:
        logger.error("No valid features selected")
        raise ValueError("No valid features selected")
    
    df_selected = df_selected[final_features]
    logger.info(f"Final selected features: {final_features}")
    
    return df_selected, final_features

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = pd.read_csv(os.path.join(base_dir, 'data', 'feature_engineering', 'company_features_data.csv'))
    params = {
        'method': 'correlation_and_rf',
        'correlation_threshold': 0.8,
        'importance_threshold': 0.01
    }
    df_selected, selected_features = select_features(df, params)
    df_selected.to_csv(os.path.join(base_dir, 'data', 'feature_engineering', 'selected_features_data.csv'), index=False)
    logger.info(f"Selected features: {selected_features}")
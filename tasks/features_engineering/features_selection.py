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
    
    available_num_cols = [col for col in numerical_cols if col in df.columns]
    available_cat_cols = [col for col in categorical_cols if col in df.columns]
    
    # Handle missing values
    df[available_num_cols] = df[available_num_cols].fillna(df[available_num_cols].median())
    df[available_cat_cols] = df[available_cat_cols].fillna('Unknown')
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', available_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat_cols)
        ])
    
    X = preprocessor.fit_transform(df)
    feature_names = available_num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(available_cat_cols))
    
    y = df['price_change_percentage'] if 'price_change_percentage' in df.columns else df['actual_side'].map({'UP': 1, 'DOWN': 0})
    
    # Correlation-based selection
    if 'correlation' in method:
        corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        selected_features = [f for f in feature_names if f not in to_drop]
        X = X[:, [i for i, f in enumerate(feature_names) if f not in to_drop]]
    else:
        selected_features = feature_names
    
    # Random Forest importance-based selection
    if 'rf' in method:
        model = RandomForestRegressor(random_state=42) if 'price_change_percentage' in df.columns else RandomForestClassifier(random_state=42)
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=selected_features)
        selected_features = importances[importances > importance_threshold].index.tolist()
        X = X[:, [i for i, f in enumerate(feature_names) if f in selected_features]]
    
    # Update DataFrame
    df_selected = df.copy()
    final_features = [col for col in df.columns if col not in available_num_cols + available_cat_cols] + selected_features
    df_selected = df_selected[final_features]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    importance_df = pd.DataFrame({'feature': importances.index, 'importance': importances.values})
    importance_path = os.path.join(base_dir, 'reports', f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")
    
    return df_selected, selected_features

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
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
import re
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'cleaning.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

class DataCleaner:
    """Class to clean financial news impact prediction data with event-specific handling."""
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        metrics_path: str = 'data/quality_metrics/cleaning_metrics.csv'
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.metrics_path = metrics_path
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}
        self.event_thresholds = {
            'Partnerships': 1.5,
            'Earnings': 2.0,
            'Corporate Action': 1.8,
            'default': 1.5
        }

    def load_data(self) -> None:
        """Load data from CSV file with error handling."""
        try:
            self.df = pd.read_csv(self.input_path, parse_dates=['published_date'], low_memory=False)
            logger.info(f"Loaded {len(self.df)} records from {self.input_path}")
            self.metrics['initial_rows'] = len(self.df)
            self.metrics['initial_columns'] = len(self.df.columns)
        except Exception as e:
            logger.error(f"Failed to load {self.input_path}: {str(e)}")
            raise ValueError(f"Failed to load {self.input_path}: {str(e)}")

    def impute_high_missing_columns(self, threshold: float = 0.9) -> None:
        """Impute columns with missing values above threshold using KNN and mode."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        missing_rates = self.df.isna().mean()
        high_missing_cols = missing_rates[missing_rates > threshold].index.tolist()
        
        if high_missing_cols:
            numerical_cols = self.df[high_missing_cols].select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = self.df[high_missing_cols].select_dtypes(include=['object']).columns
            
            # KNN imputation for numerical columns
            if numerical_cols.any():
                imputer = KNNImputer(n_neighbors=5, weights='uniform')
                self.df[numerical_cols] = pd.DataFrame(
                    imputer.fit_transform(self.df[numerical_cols]),
                    columns=numerical_cols,
                    index=self.df.index
                )
                logger.info(f"KNN imputed numerical columns: {numerical_cols}")
                self.metrics['knn_imputed_columns'] = list(numerical_cols)
            
            # Mode imputation for categorical columns
            for col in categorical_cols:
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'missing'
                self.df[col] = self.df[col].fillna(mode_value)
                logger.info(f"Imputed {col} with mode {mode_value}")
                self.metrics[f'imputed_{col}'] = mode_value

    def impute_missing_values(self) -> None:
        """Impute remaining missing values using event-specific medians for numerical and mode for categorical."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Event-specific imputation for numerical columns
        if 'event' in self.df.columns and numerical_cols.any():
            for col in numerical_cols:
                for event in self.df['event'].unique():
                    mask = self.df['event'] == event
                    if mask.sum() > 0:
                        median_value = self.df.loc[mask, col].median()
                        self.df.loc[mask & self.df[col].isna(), col] = median_value
                        logger.info(f"Imputed {col} for event {event} with median {median_value}")
                        self.metrics[f'imputed_{col}_{event}'] = median_value
        
        # Mode imputation for categorical columns
        for col in categorical_cols:
            if self.df[col].isna().any():
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'missing'
                self.df[col] = self.df[col].fillna(mode_value)
                logger.info(f"Imputed {col} with mode {mode_value}")
                self.metrics[f'imputed_{col}'] = mode_value

    def detect_outliers_iqr(self, columns: List[str]) -> Dict[str, pd.Series]:
        """Detect outliers using event-specific IQR thresholds."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        outlier_flags = {}
        for col in columns:
            if col in self.df.columns:
                for event in self.df['event'].unique():
                    mask = self.df['event'] == event
                    if mask.sum() < 10:  # Skip small event groups
                        continue
                    threshold = self.event_thresholds.get(event, self.event_thresholds['default'])
                    Q1 = self.df.loc[mask, col].quantile(0.25)
                    Q3 = self.df.loc[mask, col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    is_outlier = (self.df.loc[mask, col] < lower_bound) | (self.df[col] > upper_bound)
                    self.df.loc[mask, f'is_outlier_{col}'] = is_outlier.astype(int)
                    outlier_flags[f'{col}_{event}'] = is_outlier
                    logger.info(f"Detected {is_outlier.sum()} outliers in {col} for event {event}")
                    self.metrics[f'outliers_{col}_{event}'] = is_outlier.sum()
        return outlier_flags

    def handle_outliers(self) -> None:
        """Handle outliers with relaxed winsorizing limits."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        numerical_cols = self.df.select_dtypes(include=['float64']).columns
        
        for col in numerical_cols:
            if col in ['price_change_percentage', 'daily_alpha']:
                self.df[col] = winsorize(self.df[col], limits=[0.05, 0.05])
                logger.info(f"Relaxed winsorizing (5%) applied to {col}")
                self.metrics[f'winsorized_{col}'] = '5%'

    def clean_datetime(self) -> None:
        """Clean and standardize datetime columns with flexible parsing."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'published_date' in self.df.columns:
            try:
                self.df['published_date'] = pd.to_datetime(
                    self.df['published_date'], errors='coerce', utc=True, format='mixed'
                )
                invalid_dates = self.df['published_date'].isna()
                if invalid_dates.any():
                    logger.warning(f"Found {invalid_dates.sum()} invalid published_date values")
                    median_date = self.df['published_date'].median()
                    self.df['published_date'] = self.df['published_date'].fillna(median_date)
                    logger.info(f"Imputed {invalid_dates.sum()} invalid dates with median: {median_date}")
                    self.metrics['invalid_dates'] = invalid_dates.sum()
                    self.metrics['imputed_date'] = str(median_date)
            except Exception as e:
                logger.error(f"Datetime cleaning failed: {str(e)}")
                raise

    def clean_text_data(self) -> None:
        """Clean text data with financial-specific handling."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        text_cols = ['content', 'title', 'reason']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: re.sub(r'\$[A-Za-z]+|\d+%|[^\w\s]', ' ', str(x)) if pd.notnull(x) else 'missing'
                )
                self.df[col] = self.df[col].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
                empty_mask = (self.df[col] == '') | (self.df[col].isna())
                if empty_mask.any():
                    self.df.loc[empty_mask, col] = 'missing'
                    logger.info(f"Replaced {empty_mask.sum()} empty {col} values with 'missing'")
                    self.metrics[f'empty_{col}'] = empty_mask.sum()
                
                short_mask = (self.df[col].str.len() < 10) & (self.df[col].notna())
                if short_mask.any():
                    self.df.loc[short_mask, col] = 'missing'
                    logger.info(f"Replaced {short_mask.sum()} short {col} values with 'missing'")
                    self.metrics[f'short_replaced_{col}'] = short_mask.sum()

    def save_cleaned_data(self) -> None:
        """Save cleaned data and metrics to CSV."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        logger.info(f"Saved cleaned data to {self.output_path}")
        
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        pd.DataFrame([self.metrics]).to_csv(self.metrics_path, index=False)
        logger.info(f"Saved cleaning metrics to {self.metrics_path}")

    def clean(self) -> pd.DataFrame:
        """Run the full cleaning pipeline."""
        logger.info("Starting data cleaning pipeline")
        self.load_data()
        self.impute_high_missing_columns()
        self.impute_missing_values()
        self.clean_datetime()
        self.handle_outliers()
        self.clean_text_data()
        self.save_cleaned_data()
        logger.info("Data cleaning pipeline completed")
        return self.df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    cleaner = DataCleaner(
        input_path=os.path.join(data_dir, 'all_price_moves.csv'),
        output_path=os.path.join(data_dir, 'clean', f'cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'),
        metrics_path=os.path.join(data_dir, 'quality_metrics', 'cleaning_metrics.csv')
    )
    cleaned_df = cleaner.clean()
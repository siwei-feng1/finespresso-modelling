import pandas as pd
import os
import logging
from typing import Dict, Optional
import numpy as np

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'validation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

logger = setup_logger(__name__)

class DataValidator:
    """Class to validate financial news impact prediction data for Step 1."""
    
    def __init__(
        self,
        input_path: str,
        output_path: str = 'data/clean/clean_price_moves.csv',
        metrics_path: str = 'data/quality_metrics/validation_metrics.csv'
    ):
        """
        Initialize DataValidator.

        Args:
            input_path (str): Path to input data.
            output_path (str): Path to save cleaned data.
            metrics_path (str): Path to save validation metrics.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.metrics_path = metrics_path
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def load_data(self) -> None:
        """Load data from CSV file."""
        try:
            self.df = pd.read_csv(self.input_path, parse_dates=['published_date'], low_memory=False)
            logger.info(f"Loaded {len(self.df)} records from {self.input_path}")
            self.metrics['initial_rows'] = len(self.df)
        except Exception as e:
            logger.error(f"Failed to load {self.input_path}: {str(e)}")
            raise ValueError(f"Failed to load {self.input_path}: {str(e)}")

    def validate_data_types(self) -> None:
        """Validate and convert data types for expected columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        expected_types = {
            'id': str,
            'ticker': str,
            'published_date': 'datetime64[ns]',
            'event': str,
            'actual_side': str,
            'price_change_percentage': float,
            'daily_alpha': float,
            'content': str,
            'title': str,
            'content_en': str,
            'title_en': str
        }
        
        type_issues = []
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if actual_type != str(expected_type):
                    try:
                        if expected_type == 'datetime64[ns]':
                            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        else:
                            self.df[col] = self.df[col].astype(expected_type, errors='ignore')
                        logger.info(f"Converted {col} to {expected_type}")
                    except Exception as e:
                        type_issues.append(f"{col}: expected {expected_type}, got {actual_type}, error: {str(e)}")
                self.metrics[f'type_valid_{col}'] = actual_type == str(expected_type)
            else:
                self.metrics[f'type_valid_{col}'] = False
                type_issues.append(f"{col}: missing from dataset")
        
        self.metrics['type_issues'] = len(type_issues)
        if type_issues:
            logger.warning(f"Data type issues: {type_issues}")

    def validate_categorical_values(self) -> None:
        """Validate categorical values for event and actual_side columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        # Validate event
        if 'event' in self.df.columns:
            valid_events = ['Partnerships', 'Earnings', 'Corporate Action', 'Product Launch', 'Regulatory', 'Annual General Meeting', 'Bond Fixing']
            invalid_events = self.df['event'][~self.df['event'].isin(valid_events)].unique()
            self.metrics['invalid_events'] = len(invalid_events)
            if invalid_events.size > 0:
                logger.warning(f"Found {len(invalid_events)} invalid event types: {invalid_events.tolist()}")
                # Remove rows with invalid events
                self.df = self.df[self.df['event'].isin(valid_events)]
            else:
                logger.info("All event types are valid")

        # Validate actual_side
        if 'actual_side' in self.df.columns:
            valid_sides = ['UP', 'DOWN']
            invalid_sides = self.df['actual_side'][~self.df['actual_side'].isin(valid_sides)].unique()
            self.metrics['invalid_actual_sides'] = len(invalid_sides)
            if invalid_sides.size > 0:
                logger.warning(f"Found {len(invalid_sides)} invalid actual_side values: {invalid_sides.tolist()}")
                # Remove rows with invalid actual_side
                self.df = self.df[self.df['actual_side'].isin(valid_sides)]
            else:
                logger.info("All actual_side values are valid")

    def validate_price_ranges(self) -> None:
        """Validate price change percentage and daily alpha ranges."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'price_change_percentage' in self.df.columns:
            invalid_prices = self.df[
                (self.df['price_change_percentage'] < -100) | 
                (self.df['price_change_percentage'] > 100) |
                (self.df['price_change_percentage'].isna())
            ]
            self.metrics['invalid_prices'] = len(invalid_prices)
            if len(invalid_prices) > 0:
                logger.warning(f"Found {len(invalid_prices)} invalid price change percentages")
                # Remove rows with extreme or missing prices
                self.df = self.df[
                    (self.df['price_change_percentage'] >= -100) & 
                    (self.df['price_change_percentage'] <= 100) & 
                    (self.df['price_change_percentage'].notna())
                ]
            else:
                logger.info("All price change percentages are within valid range")

        if 'daily_alpha' in self.df.columns:
            invalid_alphas = self.df[
                (self.df['daily_alpha'].isna()) |
                (self.df['daily_alpha'].abs() > abs(1000))  # Reasonable threshold
            ]
            self.metrics['invalid_daily_alpha'] = len(self.invalid_alphas)
            if len(self.invalid_alphas) > 0:
                logger.warning(f"Found {len(self.invalid_alphas)} invalid daily alpha values")
                # Remove rows with missing or extreme daily alpha
                self.df = self.df[
                    (self.df['daily_alpha'].notna()) &
                    (self.df['daily_alpha'].abs() <= abs(1000))
                ]
            else:
                logger.info("All daily alpha values are valid")

    def validate_outliers(self) -> None:
        """Validate outliers in price_change_percentage using IQR method."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'price_change_percentage' in self.df.columns:
            Q1 = self.df['price_change_percentage'].quantile(0.25)
            Q3 = self.df['price_change_percentage'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[
                (self.df['price_change_percentage'] < lower_bound) | 
                (self.df['price_change_percentage'] > upper_bound)
            ]
            self.metrics['outliers_price'] = len(outliers)
            if len(outliers) > 0:
                logger.warning(f"Found {len(outliers)} price change outliers")
                # Remove outliers
                self.df = self.df[
                    (self.df['price_change_percentage'] >= lower_bound) & 
                    (self.df['price_change_percentage'] <= upper_bound)
                ]
            else:
                logger.info("No significant price change outliers detected")

    def validate_text_quality(self) -> None:
        """Validate text quality for content and title columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        text_cols = ['content', 'title', 'content_en', 'title_en']
        for col in text_cols:
            if col in self.df.columns:
                empty_texts = self.df[self.df[col].isna() | (self.df[col].str.strip() == '')]
                short_texts = self.df[(self.df[col].str.len() < 10) & (self.df[col].notna())]
                self.metrics[f'empty_{col}'] = len(empty_texts)
                self.metrics[f'short_{col}'] = len(short_texts)
                if len(empty_texts) > 0:
                    logger.warning(f"Found {len(empty_texts)} empty {col} values")
                    # Remove rows with empty text
                    self.df = self.df[~(self.df[col].isna() | (self.df[col].str.strip() == ''))]
                if len(short_texts) > 0:
                    logger.warning(f"Found {len(short_texts)} short {col} values (<10 chars)")
                if len(empty_texts) == 0 and len(short_texts) == 0:
                    logger.info(f"All {col} values are valid")

    def save_cleaned_data(self) -> None:
        """Save the cleaned dataset to CSV."""
        if self.df is None:
            raise ValueError("Data not loaded")
        try:
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"Saved cleaned data to {self.output_path}")
            self.metrics['final_rows'] = len(self.df)
        except Exception as e:
            logger.error(f"Failed to save cleaned data to {self.output_path}: {str(e)}")

    def save_metrics(self) -> None:
        """Save validation metrics to CSV."""
        pd.DataFrame([self.metrics]).to_csv(self.metrics_path, index=False)
        logger.info(f"Saved validation metrics to {self.metrics_path}")

    def validate(self) -> pd.DataFrame:
        """Run the full validation pipeline."""
        logger.info("Starting data validation for Step 1")
        self.load_data()
        self.validate_data_types()
        self.validate_categorical_values()
        self.validate_price_ranges()
        self.validate_outliers()
        self.validate_text_quality()
        self.save_cleaned_data()
        self.save_metrics()
        logger.info("Data validation pipeline completed")
        return self.df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    validator = DataValidator(
        input_path=os.path.join(data_dir, 'all_price_moves.csv'),
        output_path=os.path.join(data_dir, 'clean', 'clean_price_moves.csv'),
        metrics_path=os.path.join(data_dir, 'quality_metrics', 'validation_metrics.csv')
    )
    validated_df = validator.validate()
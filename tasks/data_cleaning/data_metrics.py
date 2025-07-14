import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'data_cleaning','logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'metrics.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

class DataMetricsMonitor:
    """Class to monitor data quality metrics."""
    
    def __init__(self, input_path: str):
        """
        Initialize DataMetricsMonitor.

        Args:
            input_path (str): Path to input data.
        """
        self.input_path = input_path
        self.df = pd.read_csv(input_path)
        self.metrics: Dict = {}
        self.plots_dir = 'data/quality_metrics/plots'
        os.makedirs(self.plots_dir, exist_ok=True)

    def calculate_completeness(self) -> None:
        """Calculate data completeness."""
        completeness = 1 - self.df.isna().mean()
        self.metrics['completeness'] = completeness.to_dict()
        logger.info("Calculated completeness metrics")

    def calculate_consistency(self) -> None:
        """Calculate data consistency."""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            self.metrics[f'consistency_unique_{col}'] = unique_values
        logger.info("Calculated consistency metrics")

    def calculate_validity(self) -> None:
        """Calculate data validity."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            invalid = (self.df[col] < 0).sum() if col != 'days_since_event' else 0
            self.metrics[f'validity_invalid_{col}'] = invalid
        logger.info("Calculated validity metrics")

    def calculate_outliers(self) -> None:
        """Calculate outliers using IQR."""
        numerical_cols = self.df.select_dtypes(include=['float64']).columns
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + IQR * 1.5))).sum()
            self.metrics[f'outliers_{col}'] = outliers
        logger.info("Calculated outliers metrics")

    def plot_distributions(self) -> None:
        """Plot distributions of numerical columns."""
        numerical_cols = self.df.select_dtypes(include=['float64']).columns
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plot_path = os.path.join(self.plots_dir, f'distribution_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved distribution plot for {col} to {plot_path}")

    def compare_with_baseline(self, baseline_path: Optional[str]) -> None:
        """Compare metrics with baseline."""
        if baseline_path and os.path.exists(baseline_path):
            baseline_df = pd.read_csv(baseline_path)
            baseline_metrics = baseline_df.to_dict('records')[0]
            for key in self.metrics:
                if key in baseline_metrics:
                    diff = abs(self.metrics[key] - baseline_metrics[key])
                    self.metrics[f'diff_baseline_{key}'] = diff
            logger.info("Compared metrics with baseline")

    def save_metrics(self, output_path: str) -> None:
        """Save metrics to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame([self.metrics]).to_csv(output_path, index=False)
        logger.info(f"Saved metrics to {output_path}")

    def monitor(self, baseline_path: Optional[str] = None) -> Dict:
        """Run the full metrics pipeline."""
        logger.info("Starting data metrics monitoring")
        self.calculate_completeness()
        self.calculate_consistency()
        self.calculate_validity()
        self.calculate_outliers()
        self.plot_distributions()
        self.compare_with_baseline(baseline_path)
        output_path = f'data/quality_metrics/metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.save_metrics(output_path)
        logger.info("Data metrics monitoring completed")
        return self.metrics

if __name__ == '__main__':
    monitor = DataMetricsMonitor(input_path='data/clean/clean_price_moves.csv')
    metrics = monitor.monitor()
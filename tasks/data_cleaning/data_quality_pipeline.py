import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime
import logging
from typing import Dict
from tasks.data_cleaning.analyze_data_quality import DataQualityAnalyzer
from tasks.data_cleaning.data_cleaner import DataCleaner
from tasks.data_cleaning.data_metrics import DataMetricsMonitor
from tasks.data_cleaning.data_validation import DataValidator
from tasks.data_cleaning.data_versioning import DataVersioning

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'data_cleaning','logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'pipeline.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger(__name__)

def run_data_quality_pipeline(
    input_path: str = 'data/modeling_data.csv',
    output_path: str = 'data/clean/clean_price_moves.csv',
    metrics_dir: str = 'data/quality_metrics',
    versions_dir: str = 'data/versions',
    lineage_dir: str = 'data/lineage'
) -> Dict:
    """Run the full data quality pipeline for Step 1."""
    logger.info("Starting data quality pipeline")
    
    # Step 1: Analyze Data Quality
    logger.info("Running data quality analysis")
    analyzer = DataQualityAnalyzer(filepath=input_path, output_dir=metrics_dir)
    df = analyzer.analyze()
    
    # Step 2: Clean Data
    logger.info("Running data cleaning")
    cleaner = DataCleaner(
        input_path=input_path,
        output_path=output_path,
        metrics_path=os.path.join(metrics_dir, 'cleaning_metrics.csv')
    )
    cleaned_df = cleaner.clean()
    
    # Step 3: Validate Data
    logger.info("Running data validation")
    validator = DataValidator(
        input_path=output_path,
        output_path=output_path,
        metrics_path=os.path.join(metrics_dir, 'validation_metrics.csv')
    )
    validated_df = validator.validate()
    
    # Step 4: Monitor Data Metrics
    logger.info("Running data metrics monitoring")
    monitor = DataMetricsMonitor(input_path=output_path)
    metrics = monitor.monitor(baseline_path=os.path.join(metrics_dir, 'cleaning_metrics.csv'))
    
    # Step 5: Version Data
    logger.info("Running data versioning")
    versioning = DataVersioning(
        input_path=input_path,
        processed_path=output_path,
        versions_dir=versions_dir,
        lineage_dir=lineage_dir
    )
    processing_steps = [
        'analyze_data_quality',
        'clean_data',
        'validate_data',
        'monitor_metrics'
    ]
    parameters = {
        'input_path': input_path,
        'output_path': output_path,
        'metrics_dir': metrics_dir,
        'versions_dir': versions_dir,
        'lineage_dir': lineage_dir
    }
    version, versioned_path, lineage_path = versioning.version_and_track(processing_steps, parameters)
    
    logger.info("Data quality pipeline completed")
    return {
        'cleaned_data_path': output_path,
        'version': version,
        'versioned_path': versioned_path,
        'lineage_path': lineage_path,
        'metrics': metrics
    }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    result = run_data_quality_pipeline(
        input_path=os.path.join(data_dir, 'modeling_data.csv'),
        output_path=os.path.join(data_dir, 'clean', 'clean_price_moves.csv'),
        metrics_dir=os.path.join(data_dir, 'quality_metrics'),
        versions_dir=os.path.join(data_dir, 'versions'),
        lineage_dir=os.path.join(data_dir, 'lineage')
    )
    logger.info(f"Pipeline completed. Cleaned data saved to {result['cleaned_data_path']}")
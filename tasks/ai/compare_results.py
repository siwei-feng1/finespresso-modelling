import pandas as pd
import os
import sys
import subprocess
import logging
import shutil
import tempfile

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'models_comparaison.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger(__name__)

# Ensure reports directory exists
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports')
os.makedirs(reports_dir, exist_ok=True)

def compare_classification_results(prev_file, curr_file):
    try:
        prev_df = pd.read_csv(prev_file)
        curr_df = pd.read_csv(curr_file)
        merged_df = prev_df[['event', 'accuracy']].merge(
            curr_df[['event', 'accuracy']],
            on='event',
            how='outer',
            suffixes=('_prev', '_curr')
        )
        merged_df = merged_df.fillna({'accuracy_prev': 0, 'accuracy_curr': 0})
        merged_df['accuracy_improvement_pct'] = (
            ((merged_df['accuracy_curr'] - merged_df['accuracy_prev']) / merged_df['accuracy_prev'] * 100)
            .replace([float('inf'), -float('inf')], 0)
            .fillna(0)
        )
        merged_df['above_70_percent'] = merged_df['accuracy_curr'] >= 0.70
        merged_df['prev_accuracy'] = (merged_df['accuracy_prev'] * 100).round(2)
        merged_df['curr_accuracy'] = (merged_df['accuracy_curr'] * 100).round(2)
        result_df = merged_df[[
            'event', 'prev_accuracy', 'curr_accuracy', 'accuracy_improvement_pct', 'above_70_percent'
        ]]
        output_file = os.path.join(reports_dir, 'model_comparison_binary.csv')
        result_df.to_csv(output_file, index=False)
        logger.info(f'Saved classification comparison to {output_file}')
        return result_df
    except Exception as e:
        logger.error(f"Error comparing classification results: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def compare_regression_results(prev_file, curr_file):
    try:
        prev_df = pd.read_csv(prev_file)
        curr_df = pd.read_csv(curr_file)
        merged_df = prev_df[['event', 'r2']].merge(
            curr_df[['event', 'r2']],
            on='event',
            how='outer',
            suffixes=('_prev', '_curr')
        )
        merged_df = merged_df.fillna({'r2_prev': 0, 'r2_curr': 0})
        merged_df['r2_improvement_pct'] = (
            ((merged_df['r2_curr'] - merged_df['r2_prev']) / merged_df['r2_prev'] * 100)
            .replace([float('inf'), -float('inf')], 0)
            .fillna(0)
        )
        merged_df['above_0_5_r2'] = merged_df['r2_curr'] >= 0.5
        merged_df['prev_r2'] = (merged_df['r2_prev'] * 100).round(2)
        merged_df['curr_r2'] = (merged_df['r2_curr'] * 100).round(2)
        result_df = merged_df[[
            'event', 'prev_r2', 'curr_r2', 'r2_improvement_pct', 'above_0_5_r2'
        ]]
        output_file = os.path.join(reports_dir, 'model_comparison_regression.csv')
        result_df.to_csv(output_file, index=False)
        logger.info(f'Saved regression comparison to {output_file}')
        return result_df
    except Exception as e:
        logger.error(f"Error comparing regression results: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def main():
    prev_classifier_file = os.path.join(reports_dir, 'model_results_binary_after_features_eng.csv')
    curr_classifier_file = os.path.join(reports_dir, 'model_results_binary_after_features_eng_update.csv')
    prev_regression_file = os.path.join(reports_dir, 'model_results_regression_after_features_eng.csv')
    curr_regression_file = os.path.join(reports_dir, 'model_results_regression_after_features_eng_update.csv')
    logger.info("Comparing classification results")
    compare_classification_results(prev_classifier_file, curr_classifier_file)
    logger.info("Comparing regression results")
    compare_regression_results(prev_regression_file, curr_regression_file)

if __name__ == '__main__':
    main()
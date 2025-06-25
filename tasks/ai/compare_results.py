import pandas as pd
import os
import sys
import subprocess
import logging
import shutil
import tempfile

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

# Ensure reports directory exists
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports')
os.makedirs(reports_dir, exist_ok=True)

def modify_script_for_data_source(script_path, data_file):
    """Temporarily modify a training script to use a specific data file."""
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace the data file path
    original_line = "price_moves_file = 'data/clean/clean_price_moves.csv'"
    modified_line = f"price_moves_file = '{data_file}'"
    modified_content = content.replace(original_line, modified_line)
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    temp_file.write(modified_content)
    temp_file.close()
    
    return temp_file.name

def run_training_script(script_path):
    """Run a training script and return True if successful."""
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        logger.info(f"Successfully ran {script_path}: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e.stderr}")
        return False

def compare_classification_results(prev_results_file, curr_results_file):
    """Compare classification results and generate comparison CSV."""
    try:
        prev_df = pd.read_csv(prev_results_file)
        curr_df = pd.read_csv(curr_results_file)
        
        # Merge on event
        merged_df = prev_df[['event', 'accuracy']].merge(
            curr_df[['event', 'accuracy']],
            on='event',
            how='outer',
            suffixes=('_prev', '_curr')
        )
        
        # Fill NaN with 0 for events not present in one dataset
        merged_df = merged_df.fillna({'accuracy_prev': 0, 'accuracy_curr': 0})
        
        # Calculate improvement percentage
        merged_df['accuracy_improvement_pct'] = (
            ((merged_df['accuracy_curr'] - merged_df['accuracy_prev']) / merged_df['accuracy_prev'] * 100)
            .replace([float('inf'), -float('inf')], 0)
            .fillna(0)
        )
        
        # Check if current accuracy is above 70%
        merged_df['above_70_percent'] = merged_df['accuracy_curr'] >= 0.70
        
        # Format accuracy as percentage
        merged_df['prev_accuracy'] = (merged_df['accuracy_prev'] * 100).round(2)
        merged_df['curr_accuracy'] = (merged_df['accuracy_curr'] * 100).round(2)
        
        # Select and rename columns
        result_df = merged_df[[
            'event', 'prev_accuracy', 'curr_accuracy', 'accuracy_improvement_pct', 'above_70_percent'
        ]]
        
        # Save to CSV
        output_file = os.path.join(reports_dir, 'model_comparison_binary.csv')
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved classification comparison to {output_file}")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error comparing classification results: {str(e)}")
        return None

def compare_regression_results(prev_results_file, curr_results_file):
    """Compare regression results and generate comparison CSV."""
    try:
        prev_df = pd.read_csv(prev_results_file)
        curr_df = pd.read_csv(curr_results_file)
        
        # Merge on event
        merged_df = prev_df[['event', 'r2']].merge(
            curr_df[['event', 'r2']],
            on='event',
            how='outer',
            suffixes=('_prev', '_curr')
        )
        
        # Fill NaN with 0 for events not present in one dataset
        merged_df = merged_df.fillna({'r2_prev': 0, 'r2_curr': 0})
        
        # Calculate improvement percentage
        merged_df['r2_improvement_pct'] = (
            ((merged_df['r2_curr'] - merged_df['r2_prev']) / merged_df['r2_prev'] * 100)
            .replace([float('inf'), -float('inf')], 0)
            .fillna(0)
        )
        
        # Check if current R2 is positive
        merged_df['r2_positive'] = merged_df['r2_curr'] > 0
        
        # Rename columns
        result_df = merged_df[[
            'event', 'r2_prev', 'r2_curr', 'r2_improvement_pct', 'r2_positive'
        ]].rename(columns={'r2_prev': 'prev_r2', 'r2_curr': 'curr_r2'})
        
        # Save to CSV
        output_file = os.path.join(reports_dir, 'model_comparison_regression.csv')
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved regression comparison to {output_file}")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error comparing regression results: {str(e)}")
        return None

def main():
    logger.info("Starting comparison of model results")
    
    # Paths to training scripts
    classifier_script = 'tasks/ai/train_classifier_enhanced.py'
    regression_script = 'tasks/ai/train_regression_enhanced.py'
    
    # Original and cleaned data files
    original_data = 'data/all_price_moves.csv'
    cleaned_data = 'data/clean/clean_price_moves.csv'
    
    # Temporary result files
    prev_binary_results = os.path.join(reports_dir, 'model_results_binary.csv')
    curr_binary_results = os.path.join(reports_dir, 'model_results_binary_after_cleaning.csv')
    prev_regression_results = os.path.join(reports_dir, 'model_results_regression.csv')
    curr_regression_results = os.path.join(reports_dir, 'model_results_regression_after_cleaning.csv')
    
    # Run classifier with original data
    logger.info("Running classifier with original data")
    temp_classifier = modify_script_for_data_source(classifier_script, original_data)
    if run_training_script(temp_classifier):
        shutil.move(curr_binary_results, prev_binary_results)
    else:
        logger.error("Failed to run classifier with original data")
        os.unlink(temp_classifier)
        return
    
    # Run classifier with cleaned data (restore original script)
    logger.info("Running classifier with cleaned data")
    if run_training_script(classifier_script):
        logger.info("Classifier results generated for cleaned data")
    else:
        logger.error("Failed to run classifier with cleaned data")
        os.unlink(temp_classifier)
        return
    
    # Compare classification results
    compare_classification_results(prev_binary_results, curr_binary_results)
    
    # Run regression with original data
    logger.info("Running regression with original data")
    temp_regression = modify_script_for_data_source(regression_script, original_data)
    if run_training_script(temp_regression):
        shutil.move(curr_regression_results, prev_regression_results)
    else:
        logger.error("Failed to run regression with original data")
        os.unlink(temp_regression)
        os.unlink(temp_classifier)
        return
    
    # Run regression with cleaned data (restore original script)
    logger.info("Running regression with cleaned data")
    if run_training_script(regression_script):
        logger.info("Regression results generated for cleaned data")
    else:
        logger.error("Failed to run regression with cleaned data")
        os.unlink(temp_regression)
        os.unlink(temp_classifier)
        return
    
    # Compare regression results
    compare_regression_results(prev_regression_results, curr_regression_results)
    
    # Clean up temporary files
    os.unlink(temp_classifier)
    os.unlink(temp_regression)
    
    logger.info("Comparison completed")

if __name__ == '__main__':
    main()
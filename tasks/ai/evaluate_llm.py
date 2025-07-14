import pandas as pd
import numpy as np
import os
import sys
import argparse
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import re
import logging

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from utils.db.news_db_util import get_news_df
from utils.db.price_move_db_util import get_price_moves
from utils.logging.log_util import get_logger
from utils.db.model_db_util import save_results

logger = get_logger(__name__)

# Verify OpenAI API key is available
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Ensure reports directory exists at the top-level
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports')
os.makedirs(reports_dir, exist_ok=True)

# Set up a logger for raw LLM outputs
raw_llm_logger = logging.getLogger("llm_raw_output")
raw_llm_logger.setLevel(logging.INFO)
raw_llm_handler = logging.FileHandler("llm_raw_outputs.log")
raw_llm_logger.addHandler(raw_llm_handler)

# Set up a logger for timing and progress
timing_logger = logging.getLogger("timing_progress")
timing_logger.setLevel(logging.INFO)
timing_handler = logging.FileHandler("timing_progress.log")
timing_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
timing_handler.setFormatter(timing_formatter)
timing_logger.addHandler(timing_handler)

class PricePrediction(BaseModel):
    """Pydantic model for structured price prediction output"""
    prediction: str = Field(description="Predicted direction: 'UP' or 'DOWN'")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief reasoning for the prediction")

def log_timing(message: str, start_time: float = None):
    """Log timing information with timestamps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        timing_logger.info(f"[{timestamp}] {message} (elapsed: {elapsed:.2f}s)")
        logger.info(f"[{timestamp}] {message} (elapsed: {elapsed:.2f}s)")
    else:
        timing_logger.info(f"[{timestamp}] {message}")
        logger.info(f"[{timestamp}] {message}")

def load_sample_data(n_samples: int = 5) -> pd.DataFrame:
    """
    Load n samples of news items with their price moves from database
    
    Args:
        n_samples: Number of samples to load per event type
        
    Returns:
        DataFrame with news and price move data
    """
    log_timing(f"Loading {n_samples} samples per event type from database...")
    
    try:
        # Get price moves data (which already includes joined news data)
        df = get_price_moves()
        log_timing(f"Loaded {len(df)} total records from database")
        
        if df.empty:
            log_timing("No data returned from database")
            return pd.DataFrame()
        
        # Filter out 'unknown' values and ensure we have valid data
        df = df[df['actual_side'].isin(['UP', 'DOWN'])]
        log_timing(f"After filtering, {len(df)} records with valid actual_side")
        
        # Get unique events
        events = df['event'].unique()
        log_timing(f"Found {len(events)} unique events: {events}")
        
        # Sample n items per event
        sampled_data = []
        for event in events:
            event_df = df[df['event'] == event]
            if len(event_df) >= n_samples:
                # Sample n items randomly
                sampled = event_df.sample(n=n_samples, random_state=42)
            else:
                # Use all available items if less than n_samples
                sampled = event_df
                log_timing(f"Event '{event}' has only {len(event_df)} items, using all")
            
            sampled_data.append(sampled)
        
        if sampled_data:
            result_df = pd.concat(sampled_data, ignore_index=True)
            log_timing(f"Final sampled dataset: {len(result_df)} records across {len(events)} events")
            return result_df
        else:
            log_timing("No valid data found after sampling")
            return pd.DataFrame()
            
    except Exception as e:
        log_timing(f"Error loading sample data from database: {str(e)}")
        logger.error(f"Error loading sample data from database: {str(e)}")
        logger.exception("Detailed traceback:")
        return pd.DataFrame()

def create_few_shot_examples(sample_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create few-shot examples from sample data
    
    Args:
        sample_data: DataFrame with news and price move data
        
    Returns:
        List of example dictionaries for few-shot learning
    """
    examples = []
    
    for _, row in sample_data.iterrows():
        # Get the text to use (priority order)
        text = (row['content_en'] if pd.notna(row['content_en']) and row['content_en'] != '' 
               else row['title_en'] if pd.notna(row['title_en']) and row['title_en'] != ''
               else row['content'] if pd.notna(row['content']) and row['content'] != ''
               else row['title'])
        
        if pd.isna(text) or text == '':
            continue
            
        example = {
            "news_text": text[:500] + "..." if len(text) > 500 else text,  # Truncate long text
            "event_type": row['event'],
            "actual_direction": row['actual_side'],
            "price_change": f"{row['price_change_percentage']:.2f}%" if pd.notna(row['price_change_percentage']) else "N/A"
        }
        examples.append(example)
    
    log_timing(f"Created {len(examples)} few-shot examples")
    return examples

def create_few_shot_prompt_template(examples: List[Dict[str, Any]]) -> FewShotChatMessagePromptTemplate:
    """
    Create a few-shot prompt template with examples
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        FewShotChatMessagePromptTemplate
    """
    # Create the example template
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "News: {news_text}\nEvent Type: {event_type}\nActual Price Movement: {actual_direction} ({price_change})"),
        ("assistant", "Based on this news, I predict the stock price will move {actual_direction}.")
    ])
    
    # Create the few-shot prompt template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    
    return few_shot_prompt

def create_main_prompt_template() -> ChatPromptTemplate:
    """
    Create the main prompt template for prediction
    
    Returns:
        ChatPromptTemplate
    """
    system_template = """You are a financial analyst specializing in predicting stock price movements based on news events.\n\nYour task is to analyze news content and predict whether a stock's price will move UP or DOWN.\n\nConsider these factors:\n- The type of event (clinical trial results, FDA approval, earnings, etc.)\n- The sentiment and content of the news\n- The company's industry and market context\n- Historical patterns for similar events\n\nReturn your prediction in the following JSON format:\n{{\n    \"prediction\": \"UP\" or \"DOWN\",\n    \"confidence\": 0.0 to 1.0,\n    \"reasoning\": \"Brief explanation of your reasoning\"\n}}\n\nRespond with only the JSON object and nothing else. Begin your response with '{{' and end with '}}'. Do not include anything before or after the JSON object, not even code fences, explanations, or extra whitespace."""

    human_template = """Analyze this news and predict the stock price movement:\n\nNews: {news_text}\nEvent Type: {event_type}\n\n{format_instructions}"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

def predict_with_llm(prompt, parser, llm, raw_llm_logger=None):
    start_time = time.time()
    try:
        response = llm.invoke(prompt)
        llm_time = time.time() - start_time
        log_timing(f"LLM call completed in {llm_time:.2f}s", start_time)
        
        try:
            result = parser.parse(response.content)
            parse_time = time.time() - start_time - llm_time
            log_timing(f"Parsing successful in {parse_time:.2f}s")
            return result
        except Exception as e:
            # Try regex fallback
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                try:
                    result = parser.parse(match.group(0))
                    parse_time = time.time() - start_time - llm_time
                    log_timing(f"Regex fallback parsing successful in {parse_time:.2f}s")
                    return result
                except Exception as e2:
                    if raw_llm_logger:
                        raw_llm_logger.error(f"Raw LLM output (regex fallback failed): {response.content}")
                    log_timing(f"Regex fallback parsing failed after {time.time() - start_time:.2f}s")
                    raise e2
            else:
                if raw_llm_logger:
                    raw_llm_logger.error(f"Raw LLM output (no JSON found): {response.content}")
                log_timing(f"No JSON found in output after {time.time() - start_time:.2f}s")
                raise e
    except Exception as e:
        if raw_llm_logger:
            raw_llm_logger.error(f"Raw LLM output (outer exception): {getattr(e, 'content', str(e))}")
        log_timing(f"LLM call failed after {time.time() - start_time:.2f}s")
        raise e

def train_and_evaluate_llm_model(event: str, sample_data: pd.DataFrame, 
                                n_samples: int = 5) -> Optional[Dict[str, Any]]:
    """
    Train and evaluate LLM model for a specific event type
    
    Args:
        event: Event type to train on
        sample_data: Full sample dataset
        n_samples: Number of samples per event for few-shot learning
        
    Returns:
        Dictionary with model results or None if training fails
    """
    event_start_time = time.time()
    log_timing(f"Starting training for event: {event}")
    
    try:
        # Filter data for this event
        event_data = sample_data[sample_data['event'] == event].copy()
        
        if len(event_data) < 2:
            log_timing(f"Skipping event '{event}' - insufficient data ({len(event_data)} samples)")
            return None
        
        log_timing(f"Event '{event}' has {len(event_data)} samples")
        
        # Split into train/test (80/20)
        train_size = int(len(event_data) * 0.8)
        train_data = event_data.head(train_size)
        test_data = event_data.tail(len(event_data) - train_size)
        
        log_timing(f"Split: {len(train_data)} train, {len(test_data)} test samples")
        
        # Create few-shot examples from training data
        examples_start = time.time()
        examples = create_few_shot_examples(train_data)
        log_timing(f"Created {len(examples)} few-shot examples", examples_start)
        
        # Create prompts and LLM
        setup_start = time.time()
        few_shot_prompt = create_few_shot_prompt_template(examples)
        main_prompt = create_main_prompt_template()
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        parser = PydanticOutputParser(pydantic_object=PricePrediction)
        log_timing("Setup prompts and LLM", setup_start)
        
        # Make predictions on test data
        predictions = []
        actuals = []
        prediction_start = time.time()
        
        log_timing(f"Starting predictions for {len(test_data)} test samples")
        
        for idx, (_, row) in enumerate(test_data.iterrows()):
            sample_start = time.time()
            
            # Get the text to use (priority order)
            text = (row['content_en'] if pd.notna(row['content_en']) and row['content_en'] != '' 
                   else row['title_en'] if pd.notna(row['title_en']) and row['title_en'] != ''
                   else row['content'] if pd.notna(row['content']) and row['content'] != ''
                   else row['title'])
            
            if pd.isna(text) or text == '':
                log_timing(f"Sample {idx+1}/{len(test_data)}: Skipped (empty text)")
                continue
            
            # Truncate text if too long
            truncated_text = text[:500] + "..." if len(text) > 500 else text
            
            # Create the full prompt with few-shot examples
            full_prompt = main_prompt.format_messages(
                news_text=truncated_text,
                event_type=row['event'],
                format_instructions=parser.get_format_instructions()
            )
            
            # Add few-shot examples to the prompt
            few_shot_messages = few_shot_prompt.format_messages()
            full_messages = few_shot_messages + full_prompt
            
            # Make prediction
            try:
                prediction = predict_with_llm(
                    full_messages,
                    parser, llm, raw_llm_logger
                )
                
                if prediction:
                    predictions.append(prediction.prediction)
                    actuals.append(row['actual_side'])
                    log_timing(f"Sample {idx+1}/{len(test_data)}: Predicted {prediction.prediction} (actual: {row['actual_side']}) in {time.time() - sample_start:.2f}s")
                else:
                    log_timing(f"Sample {idx+1}/{len(test_data)}: No prediction returned")
                    
            except Exception as e:
                log_timing(f"Sample {idx+1}/{len(test_data)}: Prediction failed - {str(e)}")
                continue
        
        total_prediction_time = time.time() - prediction_start
        log_timing(f"Completed {len(predictions)} predictions in {total_prediction_time:.2f}s (avg: {total_prediction_time/len(predictions):.2f}s per prediction)")
        
        if not predictions:
            log_timing(f"Event '{event}': No successful predictions")
            return None
        
        # Calculate metrics
        metrics_start = time.time()
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        accuracy = (correct / len(predictions)) * 100
        
        # Calculate UP/DOWN specific accuracies
        up_mask = [a == 'UP' for a in actuals]
        down_mask = [a == 'DOWN' for a in actuals]
        
        up_accuracy = 0.0
        down_accuracy = 0.0
        
        if any(up_mask):
            up_correct = sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                           if up_mask[i] and p == a)
            up_accuracy = (up_correct / sum(up_mask)) * 100
        
        if any(down_mask):
            down_correct = sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                             if down_mask[i] and p == a)
            down_accuracy = (down_correct / sum(down_mask)) * 100
        
        # Calculate prediction distribution
        up_predictions = sum(1 for p in predictions if p == 'UP')
        down_predictions = sum(1 for p in predictions if p == 'DOWN')
        
        up_pred_pct = (up_predictions / len(predictions)) * 100 if predictions else 0
        down_pred_pct = (down_predictions / len(predictions)) * 100 if predictions else 0
        
        log_timing(f"Calculated metrics in {time.time() - metrics_start:.2f}s")
        
        result = {
            'event': event,
            'language': 'en',
            'accuracy': accuracy,
            'precision': accuracy,  # Simplified for binary classification
            'recall': accuracy,     # Simplified for binary classification
            'f1_score': accuracy,   # Simplified for binary classification
            'auc_roc': 0.0,         # Not calculated for LLM
            'test_sample': len(test_data),
            'training_sample': len(train_data),
            'total_sample': len(event_data),
            'model_version': f'llm_fewshot_{n_samples}',
            'vectorizer_version': 'none',
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'total_up': sum(up_mask),
            'total_down': sum(down_mask),
            'correct_up': sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                            if up_mask[i] and p == a),
            'correct_down': sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                              if down_mask[i] and p == a),
            'up_predictions_pct': up_pred_pct,
            'down_predictions_pct': down_pred_pct,
            'model_type': 'llm_fewshot'
        }
        
        total_event_time = time.time() - event_start_time
        log_timing(f"Event '{event}' completed in {total_event_time:.2f}s", event_start_time)
        log_timing(f"Event '{event}' results - Accuracy: {accuracy:.3f}, UP: {up_accuracy:.2f}%, DOWN: {down_accuracy:.2f}%")
        
        return result
        
    except Exception as e:
        log_timing(f"Error training model for event '{event}': {str(e)}")
        logger.error(f"Error training model for event '{event}': {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def train_all_events_llm_model(sample_data: pd.DataFrame, n_samples: int = 5) -> Optional[Dict[str, Any]]:
    """
    Train LLM model for all events combined
    
    Args:
        sample_data: Sample data for few-shot learning
        n_samples: Number of samples used
        
    Returns:
        Dictionary with evaluation results or None if failed
    """
    try:
        log_timing("Starting all events LLM model training")
        
        if len(sample_data) < 10:
            log_timing("Skipping all events model - insufficient data")
            return None
        
        # Split into training and testing
        train_size = min(n_samples * 2, len(sample_data) - 5)  # Leave at least 5 for testing
        train_data = sample_data.head(train_size)
        test_data = sample_data.tail(len(sample_data) - train_size)
        
        if len(test_data) < 5:
            log_timing("Skipping all events model - insufficient test data")
            return None
        
        log_timing(f"Training with {len(train_data)} examples, testing with {len(test_data)} examples")
        
        # Create few-shot examples from training data
        examples = create_few_shot_examples(train_data)
        
        if not examples:
            log_timing("Skipping all events model - no valid examples created")
            return None
        
        # Create prompts
        few_shot_prompt = create_few_shot_prompt_template(examples)
        main_prompt = create_main_prompt_template()
        
        # Initialize LLM and parser
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=200,
            openai_api_key=api_key
        )
        parser = PydanticOutputParser(pydantic_object=PricePrediction)
        
        # Make predictions on test data
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            # Get text for prediction
            text = (row['content_en'] if pd.notna(row['content_en']) and row['content_en'] != '' 
                   else row['title_en'] if pd.notna(row['title_en']) and row['title_en'] != ''
                   else row['content'] if pd.notna(row['content']) and row['content'] != ''
                   else row['title'])
            
            if pd.isna(text) or text == '':
                continue
            
            # Make prediction
            prediction = predict_with_llm(
                main_prompt.format_messages(
                    news_text=text,
                    event_type=row['event'],
                    format_instructions=parser.get_format_instructions()
                ),
                parser, llm, raw_llm_logger
            )
            
            if prediction:
                predictions.append(prediction.prediction)
                actuals.append(row['actual_side'])
            else:
                log_timing(f"Failed to get prediction for test item in all_events model")
        
        if not predictions:
            log_timing("No successful predictions for all events model")
            return None
        
        # Calculate metrics
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        accuracy = correct / len(predictions) if predictions else 0
        
        # Calculate directional metrics
        up_mask = [a == 'UP' for a in actuals]
        down_mask = [a == 'DOWN' for a in actuals]
        
        up_accuracy = 0
        down_accuracy = 0
        if any(up_mask):
            up_correct = sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                           if up_mask[i] and p == a)
            up_accuracy = (up_correct / sum(up_mask)) * 100
        
        if any(down_mask):
            down_correct = sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                             if down_mask[i] and p == a)
            down_accuracy = (down_correct / sum(down_mask)) * 100
        
        # Calculate prediction distribution
        up_predictions = sum(1 for p in predictions if p == 'UP')
        down_predictions = sum(1 for p in predictions if p == 'DOWN')
        
        up_pred_pct = (up_predictions / len(predictions)) * 100 if predictions else 0
        down_pred_pct = (down_predictions / len(predictions)) * 100 if predictions else 0
        
        result = {
            'event': 'all_events',
            'language': 'en',
            'accuracy': accuracy,
            'precision': accuracy,  # Simplified for binary classification
            'recall': accuracy,     # Simplified for binary classification
            'f1_score': accuracy,   # Simplified for binary classification
            'auc_roc': 0.0,         # Not calculated for LLM
            'test_sample': len(test_data),
            'training_sample': len(train_data),
            'total_sample': len(sample_data),
            'model_version': f'llm_fewshot_{n_samples}',
            'vectorizer_version': 'none',
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'total_up': sum(up_mask),
            'total_down': sum(down_mask),
            'correct_up': sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                            if up_mask[i] and p == a),
            'correct_down': sum(1 for i, (p, a) in enumerate(zip(predictions, actuals)) 
                              if down_mask[i] and p == a),
            'up_predictions_pct': up_pred_pct,
            'down_predictions_pct': down_pred_pct,
            'model_type': 'llm_fewshot'
        }
        
        log_timing("All events LLM model results:")
        log_timing(f"  Accuracy: {accuracy:.3f}")
        log_timing(f"  UP accuracy: {up_accuracy:.2f}%, DOWN accuracy: {down_accuracy:.2f}%")
        log_timing(f"  Predictions - UP: {up_pred_pct:.2f}%, DOWN: {down_pred_pct:.2f}%")
        
        return result
        
    except Exception as e:
        log_timing(f"Error training all events LLM model: {str(e)}")
        logger.error(f"Error training all events LLM model: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def process_results(results: List[Dict[str, Any]], sample_data: pd.DataFrame):
    """
    Process and save results
    
    Args:
        results: List of result dictionaries
        sample_data: Original sample data
    """
    try:
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            log_timing("No valid results to process")
            return
        
        results_df = pd.DataFrame(valid_results)
        
        # Calculate event counts
        event_counts = sample_data.groupby('event').size().to_dict()
        results_df['total_sample'] = results_df.apply(
            lambda x: event_counts.get(x['event'], 0), 
            axis=1
        )
        
        results_df = results_df.sort_values(by='accuracy', ascending=False)
        
        # Ensure correct data types
        results_df['event'] = results_df['event'].astype(str)
        
        # Handle potential non-finite values for float columns
        float_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        for col in float_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(float)
        
        # Handle potential non-finite values for integer columns
        int_columns = ['test_sample', 'training_sample', 'total_sample']
        for col in int_columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).astype(int)
        
        # Replace NaN values with None for database compatibility
        results_df = results_df.replace({np.nan: None})
        
        # Save to CSV
        results_csv = os.path.join(reports_dir, 'model_results_binary_llm.csv')
        results_df.to_csv(results_csv, index=False)
        log_timing(f'Successfully wrote LLM results to {results_csv}')
        
        # Save results to the database
        success = save_results(results_df)
        if success:
            log_timing('Successfully wrote LLM results to database')
        else:
            log_timing('Failed to write LLM results to database')
        
        log_timing(f'Average accuracy score: {results_df["accuracy"].mean():.3f}')
        
    except Exception as e:
        log_timing(f"Error processing/saving results: {e}")
        logger.error(f"Error processing/saving results: {e}")
        logger.exception("Detailed traceback:")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LLM classifier models with few-shot learning')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of samples per event type for few-shot learning (default: 5)')
    parser.add_argument("--test_single", action="store_true", help="Run a minimal test with a single example and log raw LLM output.")
    args = parser.parse_args()
    
    overall_start_time = time.time()
    log_timing("Starting LLM classifier training")
    log_timing(f"Using {args.n_samples} samples per event type")
    
    # Load sample data
    data_load_start = time.time()
    sample_data = load_sample_data(args.n_samples)
    log_timing("Data loading completed", data_load_start)
    
    if sample_data.empty:
        log_timing("ERROR: No data loaded from database. Please check the data.")
        return
    
    log_timing(f"Loaded {len(sample_data)} sample records")
    log_timing(f"Events found: {len(sample_data['event'].unique())} unique events")
    
    if args.test_single:
        print("Running minimal LLM test with a single example...")
        # Example data (replace with a real one if available)
        test_example = {
            "news_text": "Biotech company X announced positive results from its phase 3 clinical trial.",
            "event_type": "clinical trial results",
            "actual_direction": "UP",
            "price_change": "+12.5%"
        }
        main_prompt = create_main_prompt_template()
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import PydanticOutputParser
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        parser = PydanticOutputParser(pydantic_object=PricePrediction)
        prompt = main_prompt.format_messages(
            news_text=test_example["news_text"],
            event_type=test_example["event_type"],
            format_instructions=parser.get_format_instructions()
        )
        try:
            response = llm.invoke(prompt)
            print("Raw LLM output:", response.content)
            raw_llm_logger.info(f"Minimal test raw output: {response.content}")
            try:
                parsed = parser.parse(response.content)
                print("Parsed output:", parsed)
            except Exception as e:
                print("Parsing failed:", e)
        except Exception as e:
            print("Error during minimal LLM test:", e)
        exit(0)

    # Train models for each event
    results = []
    events = sample_data['event'].unique()
    
    log_timing(f"Starting training for {len(events)} events")
    
    for event_idx, event in enumerate(events):
        event_start = time.time()
        log_timing(f"Processing event {event_idx+1}/{len(events)}: {event}")
        
        try:
            result = train_and_evaluate_llm_model(event, sample_data, args.n_samples)
            if result is not None:
                results.append(result)
                log_timing(f"Event {event_idx+1}/{len(events)} completed successfully")
            else:
                log_timing(f"Event {event_idx+1}/{len(events)} failed or skipped")
        except Exception as e:
            log_timing(f"Event {event_idx+1}/{len(events)} error: {e}")
            logger.error(f"Error training model for event '{event}': {e}")
        
        # Log progress
        progress = (event_idx + 1) / len(events) * 100
        elapsed = time.time() - overall_start_time
        estimated_total = elapsed / (event_idx + 1) * len(events)
        remaining = estimated_total - elapsed
        log_timing(f"Progress: {progress:.1f}% ({event_idx+1}/{len(events)}) - Elapsed: {elapsed/60:.1f}min, Estimated remaining: {remaining/60:.1f}min")
    
    # Train all events model
    log_timing("Starting all events LLM model training")
    all_events_start = time.time()
    all_events_result = train_all_events_llm_model(sample_data, args.n_samples)
    log_timing("All events model training completed", all_events_start)
    
    if all_events_result:
        results.append(all_events_result)
        log_timing("All events LLM model added to results")
    else:
        log_timing("Failed to train all events LLM model")
    
    if not results:
        log_timing("WARNING: No models were trained. Check the data and event filtering.")
        return
    
    # Process and save results
    log_timing("Processing and saving LLM results")
    save_start = time.time()
    process_results(results, sample_data)
    log_timing("Results processing and saving completed", save_start)
    
    total_time = time.time() - overall_start_time
    log_timing(f"LLM classifier training completed in {total_time/60:.1f} minutes", overall_start_time)

if __name__ == '__main__':
    main() 
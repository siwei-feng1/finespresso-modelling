#!/usr/bin/env python3
"""
Test script for LLM classifier training
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path to import tasks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ai.train_classifier_llm import (
    load_sample_data, 
    create_few_shot_examples,
    create_few_shot_prompt_template,
    create_main_prompt_template,
    train_and_evaluate_llm_model,
    train_all_events_llm_model
)
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

def test_data_loading():
    """Test data loading functionality"""
    logger.info("=" * 60)
    logger.info("TESTING DATA LOADING")
    logger.info("=" * 60)
    
    # Test with different sample sizes
    for n_samples in [3, 5, 10]:
        logger.info(f"\nTesting with {n_samples} samples per event...")
        
        try:
            sample_data = load_sample_data(n_samples)
            
            if not sample_data.empty:
                logger.info(f"✅ Successfully loaded {len(sample_data)} records")
                logger.info(f"   Events found: {sample_data['event'].unique()}")
                logger.info(f"   Columns: {sample_data.columns.tolist()}")
                
                # Check data quality
                valid_actual_sides = sample_data['actual_side'].isin(['UP', 'DOWN']).sum()
                logger.info(f"   Valid actual_side values: {valid_actual_sides}/{len(sample_data)}")
                
                # Show sample of data
                logger.info(f"   Sample data:")
                for _, row in sample_data.head(2).iterrows():
                    text = (row['content_en'] if pd.notna(row['content_en']) and row['content_en'] != '' 
                           else row['title_en'] if pd.notna(row['title_en']) and row['title_en'] != ''
                           else row['content'] if pd.notna(row['content']) and row['content'] != ''
                           else row['title'])
                    logger.info(f"     Event: {row['event']}, Direction: {row['actual_side']}, Text: {text[:100]}...")
            else:
                logger.warning(f"⚠️  No data loaded for {n_samples} samples")
                
        except Exception as e:
            logger.error(f"❌ Error loading data with {n_samples} samples: {str(e)}")

def test_few_shot_examples():
    """Test few-shot example creation"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING FEW-SHOT EXAMPLE CREATION")
    logger.info("=" * 60)
    
    try:
        # Load sample data
        sample_data = load_sample_data(3)
        
        if sample_data.empty:
            logger.warning("⚠️  No data available for testing few-shot examples")
            return
        
        # Create examples
        examples = create_few_shot_examples(sample_data)
        
        logger.info(f"✅ Created {len(examples)} few-shot examples")
        
        # Show sample examples
        for i, example in enumerate(examples[:3]):
            logger.info(f"   Example {i+1}:")
            logger.info(f"     Event: {example['event_type']}")
            logger.info(f"     Direction: {example['actual_direction']}")
            logger.info(f"     Price Change: {example['price_change']}")
            logger.info(f"     Text: {example['news_text'][:100]}...")
        
        # Test prompt template creation
        few_shot_prompt = create_few_shot_prompt_template(examples)
        main_prompt = create_main_prompt_template()
        
        logger.info("✅ Successfully created prompt templates")
        
    except Exception as e:
        logger.error(f"❌ Error testing few-shot examples: {str(e)}")

def test_llm_training():
    """Test LLM model training (with minimal samples to avoid API costs)"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING LLM MODEL TRAINING")
    logger.info("=" * 60)
    
    try:
        # Load minimal sample data
        sample_data = load_sample_data(2)  # Use minimal samples to reduce API calls
        
        if sample_data.empty:
            logger.warning("⚠️  No data available for testing LLM training")
            return
        
        # Get unique events
        events = sample_data['event'].unique()
        logger.info(f"Found {len(events)} events: {events}")
        
        # Test training for first event only (to minimize API costs)
        if len(events) > 0:
            first_event = events[0]
            logger.info(f"Testing LLM training for event: {first_event}")
            
            result = train_and_evaluate_llm_model(first_event, sample_data, 2)
            
            if result:
                logger.info("✅ LLM training successful!")
                logger.info(f"   Accuracy: {result['accuracy']:.3f}")
                logger.info(f"   UP accuracy: {result['up_accuracy']:.2f}%")
                logger.info(f"   DOWN accuracy: {result['down_accuracy']:.2f}%")
                logger.info(f"   Test samples: {result['test_sample']}")
                logger.info(f"   Training samples: {result['training_sample']}")
            else:
                logger.warning("⚠️  LLM training failed or returned no results")
        
        # Test all events model
        logger.info("\nTesting all events LLM model...")
        all_events_result = train_all_events_llm_model(sample_data, 2)
        
        if all_events_result:
            logger.info("✅ All events LLM training successful!")
            logger.info(f"   Accuracy: {all_events_result['accuracy']:.3f}")
            logger.info(f"   UP accuracy: {all_events_result['up_accuracy']:.2f}%")
            logger.info(f"   DOWN accuracy: {all_events_result['down_accuracy']:.2f}%")
        else:
            logger.warning("⚠️  All events LLM training failed or returned no results")
        
    except Exception as e:
        logger.error(f"❌ Error testing LLM training: {str(e)}")
        logger.exception("Detailed traceback:")

def main():
    """Run all tests"""
    logger.info("🚀 STARTING LLM CLASSIFIER TRAINING TESTS")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Verify OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        return
    
    # Test data loading
    test_data_loading()
    
    # Test few-shot example creation
    test_few_shot_examples()
    
    # Test LLM training (with minimal samples)
    test_llm_training()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 LLM CLASSIFIER TRAINING TESTS COMPLETED")
    logger.info("=" * 60)

if __name__ == '__main__':
    main() 
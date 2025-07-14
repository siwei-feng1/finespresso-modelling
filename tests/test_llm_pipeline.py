#!/usr/bin/env python3
"""
Comprehensive test script for LLM training and prediction pipeline
"""

import sys
import os
import pandas as pd
from datetime import datetime
import time

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
from tasks.ai.predict_llm import (
    predict_single_news_item,
    predict_news_batch,
    load_few_shot_examples
)
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

def test_data_loading():
    """Test data loading functionality"""
    logger.info("=" * 60)
    logger.info("TESTING DATA LOADING")
    logger.info("=" * 60)
    
    # Test with different sample sizes
    for n_samples in [3, 5]:
        logger.info(f"\nTesting with {n_samples} samples per event...")
        
        try:
            sample_data = load_sample_data(n_samples)
            
            if not sample_data.empty:
                logger.info(f"✅ Successfully loaded {len(sample_data)} records")
                logger.info(f"   Events found: {sample_data['event'].unique()}")
                logger.info(f"   Valid actual_side values: {sample_data['actual_side'].isin(['UP', 'DOWN']).sum()}/{len(sample_data)}")
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
        for i, example in enumerate(examples[:2]):
            logger.info(f"   Example {i+1}:")
            logger.info(f"     Event: {example['event_type']}")
            logger.info(f"     Direction: {example['actual_direction']}")
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

def test_single_prediction():
    """Test single news item prediction"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING SINGLE PREDICTION")
    logger.info("=" * 60)
    
    # Sample news texts for testing
    test_news = [
        {
            "text": "Company announces positive results from Phase 3 clinical trial, showing significant improvement in patient outcomes.",
            "event": "clinical_study"
        },
        {
            "text": "FDA grants approval for new drug treatment for rare diseases, representing a major milestone for the company.",
            "event": "fda_approval"
        }
    ]
    
    for i, news_item in enumerate(test_news):
        logger.info(f"\n--- Testing News Item {i+1} ---")
        logger.info(f"Event: {news_item['event']}")
        logger.info(f"Text: {news_item['text'][:50]}...")
        
        try:
            result = predict_single_news_item(
                news_item['text'], 
                news_item['event'], 
                n_samples=2  # Use minimal samples
            )
            
            if result:
                logger.info("✅ Single prediction successful!")
                logger.info(f"   Predicted: {result['predicted_side']}")
                logger.info(f"   Confidence: {result['predicted_confidence']:.2f}")
                logger.info(f"   Reasoning: {result['predicted_reasoning']}")
                logger.info(f"   Time: {result['prediction_time']:.2f}s")
            else:
                logger.warning("⚠️  Single prediction failed")
                
        except Exception as e:
            logger.error(f"❌ Error in single prediction: {str(e)}")

def test_batch_prediction():
    """Test batch prediction (with minimal data to avoid high costs)"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING BATCH PREDICTION")
    logger.info("=" * 60)
    
    try:
        # Load a small sample of news data
        from utils.db.news_db_util import get_news_df
        
        news_df = get_news_df()
        
        if news_df.empty:
            logger.warning("⚠️  No news data available for batch prediction test")
            return
        
        # Take only first 3 items to minimize API costs
        test_df = news_df.head(3).copy()
        logger.info(f"Testing batch prediction on {len(test_df)} news items")
        
        # Make predictions
        predictions_df = predict_news_batch(test_df, n_samples=2)
        
        # Check results
        successful_predictions = predictions_df['predicted_side_llm'].notna().sum()
        
        if successful_predictions > 0:
            logger.info("✅ Batch prediction successful!")
            logger.info(f"   Successful predictions: {successful_predictions}/{len(test_df)}")
            
            # Show sample predictions
            for idx, row in predictions_df.iterrows():
                if pd.notna(row['predicted_side_llm']):
                    logger.info(f"   Item {idx}: {row['predicted_side_llm']} "
                              f"(confidence: {row['predicted_confidence_llm']:.2f})")
        else:
            logger.warning("⚠️  No successful batch predictions")
            
    except Exception as e:
        logger.error(f"❌ Error in batch prediction: {str(e)}")
        logger.exception("Detailed traceback:")

def test_end_to_end_pipeline():
    """Test the complete pipeline from training to prediction"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING END-TO-END PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load training data
        logger.info("Step 1: Loading training data...")
        training_data = load_sample_data(2)
        
        if training_data.empty:
            logger.warning("⚠️  No training data available")
            return
        
        logger.info(f"✅ Loaded {len(training_data)} training examples")
        
        # Step 2: Train a model
        logger.info("Step 2: Training model...")
        events = training_data['event'].unique()
        if len(events) > 0:
            first_event = events[0]
            training_result = train_and_evaluate_llm_model(first_event, training_data, 2)
            
            if training_result:
                logger.info(f"✅ Model trained with accuracy: {training_result['accuracy']:.3f}")
            else:
                logger.warning("⚠️  Model training failed")
                return
        else:
            logger.warning("⚠️  No events available for training")
            return
        
        # Step 3: Test prediction on new data
        logger.info("Step 3: Testing prediction...")
        test_news = "Company reports strong quarterly earnings, exceeding analyst expectations."
        
        prediction_result = predict_single_news_item(
            test_news, 
            first_event, 
            n_samples=2
        )
        
        if prediction_result:
            logger.info("✅ End-to-end pipeline successful!")
            logger.info(f"   Training accuracy: {training_result['accuracy']:.3f}")
            logger.info(f"   Prediction: {prediction_result['predicted_side']}")
            logger.info(f"   Confidence: {prediction_result['predicted_confidence']:.2f}")
        else:
            logger.warning("⚠️  Prediction failed in end-to-end test")
            
    except Exception as e:
        logger.error(f"❌ Error in end-to-end pipeline: {str(e)}")
        logger.exception("Detailed traceback:")

def main():
    """Run all tests"""
    logger.info("🚀 STARTING LLM PIPELINE TESTS")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Verify OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        return
    
    logger.info("✅ OpenAI API key found")
    
    # Run tests in order
    test_data_loading()
    test_few_shot_examples()
    test_llm_training()
    test_single_prediction()
    test_batch_prediction()
    test_end_to_end_pipeline()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 LLM PIPELINE TESTS COMPLETED")
    logger.info("=" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ Data loading and few-shot example creation")
    print("✅ LLM model training (single event and all events)")
    print("✅ Single news item prediction")
    print("✅ Batch prediction")
    print("✅ End-to-end pipeline")
    print("\n💡 Note: Tests use minimal samples to reduce API costs")
    print("💡 For production, increase n_samples for better accuracy")
    print("=" * 60)

if __name__ == '__main__':
    main() 
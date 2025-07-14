#!/usr/bin/env python3
"""
Example script demonstrating LLM training and prediction pipeline
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path to import tasks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ai.train_classifier_llm import train_and_evaluate_llm_model, train_all_events_llm_model, load_sample_data
from tasks.ai.predict_llm import predict_single_news_item, predict_news_batch
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

def example_training():
    """Example of training LLM models"""
    print("=" * 60)
    print("LLM TRAINING EXAMPLE")
    print("=" * 60)
    
    # Load sample data
    print("Loading training data...")
    sample_data = load_sample_data(n_samples=3)  # Use 3 samples per event
    
    if sample_data.empty:
        print("❌ No training data available")
        return
    
    print(f"✅ Loaded {len(sample_data)} training examples")
    print(f"Events: {sample_data['event'].unique()}")
    
    # Train model for first event
    events = sample_data['event'].unique()
    if len(events) > 0:
        first_event = events[0]
        print(f"\nTraining model for event: {first_event}")
        
        result = train_and_evaluate_llm_model(first_event, sample_data, 3)
        
        if result:
            print(f"✅ Training successful!")
            print(f"   Accuracy: {result['accuracy']:.3f}")
            print(f"   UP accuracy: {result['up_accuracy']:.2f}%")
            print(f"   DOWN accuracy: {result['down_accuracy']:.2f}%")
        else:
            print("❌ Training failed")
    
    # Train all-events model
    print(f"\nTraining all-events model...")
    all_events_result = train_all_events_llm_model(sample_data, 3)
    
    if all_events_result:
        print(f"✅ All-events training successful!")
        print(f"   Accuracy: {all_events_result['accuracy']:.3f}")
        print(f"   UP accuracy: {all_events_result['up_accuracy']:.2f}%")
        print(f"   DOWN accuracy: {all_events_result['down_accuracy']:.2f}%")
    else:
        print("❌ All-events training failed")

def example_single_prediction():
    """Example of single news prediction"""
    print("\n" + "=" * 60)
    print("SINGLE PREDICTION EXAMPLE")
    print("=" * 60)
    
    # Sample news texts
    test_cases = [
        {
            "text": "Biotech company announces breakthrough results from Phase 2 clinical trial, showing 80% improvement in patient outcomes.",
            "event": "clinical_study",
            "description": "Positive clinical trial results"
        },
        {
            "text": "Pharmaceutical company receives FDA approval for new cancer treatment drug, marking a significant milestone.",
            "event": "fda_approval", 
            "description": "FDA drug approval"
        },
        {
            "text": "Company reports disappointing quarterly earnings, missing analyst expectations by 15%.",
            "event": "earnings",
            "description": "Negative earnings report"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"Event: {test_case['event']}")
        print(f"Text: {test_case['text'][:80]}...")
        
        result = predict_single_news_item(
            test_case['text'],
            test_case['event'],
            n_samples=3
        )
        
        if result:
            print(f"✅ Prediction: {result['predicted_side']}")
            print(f"   Confidence: {result['predicted_confidence']:.2f}")
            print(f"   Reasoning: {result['predicted_reasoning']}")
            print(f"   Time: {result['prediction_time']:.2f}s")
        else:
            print("❌ Prediction failed")

def example_batch_prediction():
    """Example of batch prediction"""
    print("\n" + "=" * 60)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 60)
    
    # Load a small sample of news data
    from utils.db.news_db_util import get_news_df
    
    print("Loading news data...")
    news_df = get_news_df()
    
    if news_df.empty:
        print("❌ No news data available")
        return
    
    # Take only first 5 items for demonstration
    test_df = news_df.head(5).copy()
    print(f"✅ Loaded {len(test_df)} news items for batch prediction")
    
    # Make predictions
    print("Making batch predictions...")
    predictions_df = predict_news_batch(test_df, n_samples=3)
    
    # Show results
    successful_predictions = predictions_df['predicted_side_llm'].notna().sum()
    print(f"\n✅ Batch prediction completed: {successful_predictions}/{len(test_df)} successful")
    
    if successful_predictions > 0:
        print("\nSample predictions:")
        for idx, row in predictions_df.iterrows():
            if pd.notna(row['predicted_side_llm']):
                title = row.get('title', 'No title')[:50] + "..." if len(str(row.get('title', ''))) > 50 else row.get('title', 'No title')
                print(f"   {title}")
                print(f"   → {row['predicted_side_llm']} (confidence: {row['predicted_confidence_llm']:.2f})")
                print()

def main():
    """Run all examples"""
    print("🚀 LLM TRAINING AND PREDICTION EXAMPLES")
    print(f"Timestamp: {datetime.now()}")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    print("✅ OpenAI API key found")
    
    try:
        # Run examples
        example_training()
        example_single_prediction()
        example_batch_prediction()
        
        print("\n" + "=" * 60)
        print("🎉 EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        logger.exception("Detailed traceback:")

if __name__ == '__main__':
    main() 
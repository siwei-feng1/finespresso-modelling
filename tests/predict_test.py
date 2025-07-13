import pandas as pd
import json
import os
import sys
import time
from datetime import datetime
import numpy as np

# Add the parent directory to the path to import tasks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ai.predict import predict, predict_batch, get_cache_stats, clear_model_cache
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

# Sample news articles for testing
SAMPLE_NEWS = [
    {
        "id": "test_1",
        "event": "clinical_study",
        "title": "Company Announces Positive Results from Phase 3 Clinical Trial",
        "content": "The company reported positive results from its Phase 3 clinical trial, showing significant improvement in patient outcomes. The study met all primary and secondary endpoints with statistical significance.",
        "content_en": "The company reported positive results from its Phase 3 clinical trial, showing significant improvement in patient outcomes. The study met all primary and secondary endpoints with statistical significance."
    },
    {
        "id": "test_2",
        "event": "fda_approval",
        "title": "FDA Grants Approval for New Drug Treatment",
        "content": "The Food and Drug Administration has approved the company's new drug for the treatment of rare diseases. This approval represents a major milestone for the company and patients.",
        "content_en": "The Food and Drug Administration has approved the company's new drug for the treatment of rare diseases. This approval represents a major milestone for the company and patients."
    },
    {
        "id": "test_3",
        "event": "earnings_report",
        "title": "Company Reports Strong Q4 Earnings",
        "content": "The company announced strong fourth-quarter earnings that exceeded analyst expectations. Revenue grew by 15% year-over-year, driven by strong product sales.",
        "content_en": "The company announced strong fourth-quarter earnings that exceeded analyst expectations. Revenue grew by 15% year-over-year, driven by strong product sales."
    },
    {
        "id": "test_4",
        "event": "partnership",
        "title": "Major Partnership Announced with Global Pharma",
        "content": "The company has entered into a strategic partnership with a global pharmaceutical company to develop and commercialize innovative treatments. This partnership is expected to accelerate development timelines.",
        "content_en": "The company has entered into a strategic partnership with a global pharmaceutical company to develop and commercialize innovative treatments. This partnership is expected to accelerate development timelines."
    },
    {
        "id": "test_5",
        "event": "clinical_study",
        "title": "Clinical Trial Fails to Meet Primary Endpoint",
        "content": "The company announced that its Phase 2 clinical trial failed to meet the primary endpoint. The study results showed no statistically significant difference compared to placebo.",
        "content_en": "The company announced that its Phase 2 clinical trial failed to meet the primary endpoint. The study results showed no statistically significant difference compared to placeholder."
    }
]

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def run_prediction_test():
    """Run comprehensive prediction test with timing and caching metrics"""
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION TEST")
    logger.info("=" * 60)
    
    # Clear cache at start
    logger.info("Clearing model cache at start...")
    clear_model_cache()
    
    # Test individual predictions with timing
    logger.info("\n" + "=" * 40)
    logger.info("TESTING INDIVIDUAL PREDICTIONS")
    logger.info("=" * 40)
    
    individual_results = []
    total_individual_time = 0
    
    for i, article in enumerate(SAMPLE_NEWS):
        logger.info(f"\n--- Testing Article {i+1}: {article['id']} ---")
        logger.info(f"Event: {article['event']}")
        logger.info(f"Title: {article['title'][:50]}...")
        
        start_time = time.time()
        result = predict(article, article['event'])
        end_time = time.time()
        
        article_time = end_time - start_time
        total_individual_time += article_time
        
        if result:
            logger.info(f"✅ Prediction successful in {article_time:.3f}s")
            logger.info(f"   Predicted side: {result['predicted_side']}")
            logger.info(f"   Predicted move: {result['predicted_move']:.2f}%")
            logger.info(f"   Side probability: {result['side_probability']}")
            
            result['article_id'] = article['id']
            result['event'] = article['event']
            result['title'] = article['title']
            result['total_time'] = article_time
            individual_results.append(result)
        else:
            logger.error(f"❌ Prediction failed for article {article['id']}")
    
    # Test batch predictions
    logger.info("\n" + "=" * 40)
    logger.info("TESTING BATCH PREDICTIONS")
    logger.info("=" * 40)
    
    # Test with clinical_study event (should use cached models)
    logger.info("\n--- Batch Test 1: clinical_study event (cached models) ---")
    clinical_articles = [article for article in SAMPLE_NEWS if article['event'] == 'clinical_study']
    
    start_time = time.time()
    batch_results_1 = predict_batch(clinical_articles, 'clinical_study')
    end_time = time.time()
    
    batch_time_1 = end_time - start_time
    logger.info(f"Batch prediction completed in {batch_time_1:.3f}s")
    logger.info(f"Results: {len(batch_results_1)} predictions")
    
    # Test with mixed events (should use cached models for some)
    logger.info("\n--- Batch Test 2: mixed events (some cached, some new) ---")
    start_time = time.time()
    batch_results_2 = predict_batch(SAMPLE_NEWS, 'all_events')
    end_time = time.time()
    
    batch_time_2 = end_time - start_time
    logger.info(f"Batch prediction completed in {batch_time_2:.3f}s")
    logger.info(f"Results: {len(batch_results_2)} predictions")
    
    # Get cache statistics
    logger.info("\n" + "=" * 40)
    logger.info("CACHE STATISTICS")
    logger.info("=" * 40)
    
    cache_stats = get_cache_stats()
    logger.info(f"Cached models: {cache_stats['cached_models']}")
    logger.info(f"Cache keys: {cache_stats['cache_keys']}")
    
    # Performance summary
    logger.info("\n" + "=" * 40)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 40)
    
    avg_individual_time = total_individual_time / len(SAMPLE_NEWS) if SAMPLE_NEWS else 0
    logger.info(f"Individual predictions:")
    logger.info(f"  Total time: {total_individual_time:.3f}s")
    logger.info(f"  Average per article: {avg_individual_time:.3f}s")
    logger.info(f"  Success rate: {len(individual_results)}/{len(SAMPLE_NEWS)} ({len(individual_results)/len(SAMPLE_NEWS)*100:.1f}%)")
    
    logger.info(f"\nBatch predictions:")
    logger.info(f"  Batch 1 (clinical_study): {batch_time_1:.3f}s for {len(clinical_articles)} articles")
    logger.info(f"  Batch 2 (mixed): {batch_time_2:.3f}s for {len(SAMPLE_NEWS)} articles")
    
    # Prepare results for JSON output
    results_dict = {
        'test_timestamp': datetime.now().isoformat(),
        'performance_metrics': {
            'individual_predictions': {
                'total_time': total_individual_time,
                'average_time': avg_individual_time,
                'success_rate': len(individual_results) / len(SAMPLE_NEWS) if SAMPLE_NEWS else 0,
                'total_articles': len(SAMPLE_NEWS),
                'successful_predictions': len(individual_results)
            },
            'batch_predictions': {
                'batch_1_time': batch_time_1,
                'batch_1_articles': len(clinical_articles),
                'batch_2_time': batch_time_2,
                'batch_2_articles': len(SAMPLE_NEWS)
            },
            'caching': {
                'cached_models': cache_stats['cached_models'],
                'cache_keys': cache_stats['cache_keys']
            }
        },
        'individual_results': individual_results,
        'batch_results_1': batch_results_1,
        'batch_results_2': batch_results_2
    }
    
    # Convert numpy types for JSON serialization
    results_dict = convert_numpy_types(results_dict)
    
    # Ensure test-data directory exists
    os.makedirs('test-data', exist_ok=True)
    
    # Write results to JSON file
    output_file = 'test-data/prediction_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("=" * 60)
    logger.info("PREDICTION TEST COMPLETED")
    logger.info("=" * 60)
    
    return results_dict

if __name__ == "__main__":
    run_prediction_test() 
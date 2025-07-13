import pandas as pd
import joblib
from utils.db import news_db_util
from utils.db.model_db_util import load_model_from_db
import os
import logging
import time
from typing import Dict, Tuple, Optional
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache for loaded models
_model_cache: Dict[str, Tuple] = {}

def get_cache_key(name: str, event: str, model_type: str) -> str:
    """Generate a cache key for model lookup"""
    return f"{name}_{event}_{model_type}"

def load_models_from_db(event, model_type):
    """Load models from database with caching"""
    if model_type == 'classifier_binary':
        model_name = f'{event}_classifier_binary'
        vectorizer_name = f'{event}_tfidf_vectorizer_binary'
    else:
        model_name = f'{event}_{model_type}'
        vectorizer_name = f'{event}_tfidf_vectorizer_{model_type}'
    
    # Check cache first
    model_cache_key = get_cache_key(model_name, event, model_type)
    vectorizer_cache_key = get_cache_key(vectorizer_name, event, 'vectorizer')
    
    # Load model from cache or database
    if model_cache_key in _model_cache:
        logger.info(f"Loading model from cache: {model_name}")
        model = _model_cache[model_cache_key]
    else:
        logger.info(f"Loading model from database: {model_name}")
        start_time = time.time()
        model, version = load_model_from_db(model_name, event, model_type)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.3f}s (version {version})")
        
        if model is not None:
            _model_cache[model_cache_key] = model
    
    # Load vectorizer from cache or database
    if vectorizer_cache_key in _model_cache:
        logger.info(f"Loading vectorizer from cache: {vectorizer_name}")
        vectorizer = _model_cache[vectorizer_cache_key]
    else:
        logger.info(f"Loading vectorizer from database: {vectorizer_name}")
        start_time = time.time()
        vectorizer, version = load_model_from_db(vectorizer_name, event, 'vectorizer')
        load_time = time.time() - start_time
        logger.info(f"Vectorizer loaded in {load_time:.3f}s (version {version})")
        
        if vectorizer is not None:
            _model_cache[vectorizer_cache_key] = vectorizer
    
    return model, vectorizer

def clear_model_cache():
    """Clear the model cache"""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")

def get_cache_stats():
    """Get cache statistics"""
    return {
        'cached_models': len(_model_cache),
        'cache_keys': list(_model_cache.keys())
    }

def predict(article, event):
    """Predict price movement for a single article"""
    start_time = time.time()
    
    # Prepare text for prediction
    text = f"{article['title']} {article['content']}"
    
    # Try to load models for the specific event
    model, vectorizer = load_models_from_db(event, 'classifier_binary')
    regression_model, regression_vectorizer = load_models_from_db(event, 'regression')
    
    # If specific event models not found, try fallback to 'all_events'
    if model is None or vectorizer is None:
        logger.warning(f"Models not found for event '{event}', trying 'all_events' fallback")
        model, vectorizer = load_models_from_db('all_events', 'classifier_binary')
        regression_model, regression_vectorizer = load_models_from_db('all_events', 'regression')
    
    # If still no models, return None
    if model is None or vectorizer is None:
        logger.error(f"No models available for event '{event}' or 'all_events'")
        return None
    
    # Make predictions
    try:
        # Vectorize text
        text_vectorized = vectorizer.transform([text])
        
        # Predict side (up/down)
        predicted_side = model.predict(text_vectorized)[0]
        side_probability = model.predict_proba(text_vectorized)[0]
        
        # Predict move percentage
        if regression_model is not None and regression_vectorizer is not None:
            text_regression_vectorized = regression_vectorizer.transform([text])
            predicted_move = regression_model.predict(text_regression_vectorized)[0]
        else:
            predicted_move = 0.0
        
        prediction_time = time.time() - start_time
        logger.info(f"Prediction completed in {prediction_time:.3f}s")
        
        return {
            'predicted_side': predicted_side,
            'side_probability': side_probability.tolist(),
            'predicted_move': predicted_move,
            'prediction_time': prediction_time
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

def predict_batch(articles, event):
    """Predict price movements for multiple articles"""
    start_time = time.time()
    results = []
    
    logger.info(f"Starting batch prediction for {len(articles)} articles")
    
    for i, article in enumerate(articles):
        article_start_time = time.time()
        result = predict(article, event)
        
        if result:
            result['article_id'] = article.get('id', f'article_{i}')
            result['article_time'] = time.time() - article_start_time
            results.append(result)
        else:
            logger.warning(f"Failed to predict for article {i}")
    
    total_time = time.time() - start_time
    avg_time = total_time / len(articles) if articles else 0
    
    logger.info(f"Batch prediction completed: {len(results)}/{len(articles)} successful")
    logger.info(f"Total time: {total_time:.3f}s, Average per article: {avg_time:.3f}s")
    
    return results

def main():
    # Get all news
    logger.info("Fetching all news data")
    news_df = news_db_util.get_news_df()
    
    # Make predictions
    logger.info("Starting predictions")
    pred_df = predict(news_df)
    
    # Update the news table with predictions
    logger.info("Updating news table with predictions")
    news_db_util.update_news_predictions(pred_df)
    
    logger.info("Predictions completed and news table updated.")

if __name__ == '__main__':
    main()

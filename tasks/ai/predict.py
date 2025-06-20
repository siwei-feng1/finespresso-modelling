import pandas as pd
import joblib
from utils.db import news_db_util
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models(event, model_type):
    if model_type == 'classifier_binary':
        model_filename = f'models/{event}_classifier_binary.joblib'
        vectorizer_filename = f'models/{event}_tfidf_vectorizer_binary.joblib'
    else:
        model_filename = f'models/{event}_{model_type}.joblib'
        vectorizer_filename = f'models/{event}_tfidf_vectorizer_{model_type}.joblib'
    
    logger.info(f"Looking for model file: {model_filename}")
    logger.info(f"Looking for vectorizer file: {vectorizer_filename}")
    
    if os.path.exists(model_filename) and os.path.exists(vectorizer_filename):
        logger.info(f"Loading model and vectorizer for event: {event}")
        model = joblib.load(model_filename)
        vectorizer = joblib.load(vectorizer_filename)
        return model, vectorizer
    else:
        logger.warning(f"Model or vectorizer not found for event: {event}")
        logger.warning(f"Model file exists: {os.path.exists(model_filename)}")
        logger.warning(f"Vectorizer file exists: {os.path.exists(vectorizer_filename)}")
        return None, None

def predict(df):
    move_models = {}
    move_vectorizers = {}
    side_models = {}
    side_vectorizers = {}

    # Load all_events models
    all_events_move_model, all_events_move_vectorizer = load_models('all_events', 'regression')
    all_events_side_model, all_events_side_vectorizer = load_models('all_events', 'classifier_binary')

    for index, row in df.iterrows():
        # Check if event is None or NaN
        if pd.isna(row['event']):
            logger.warning(f"Skipping prediction for row {index}: Event is None or NaN")
            continue

        event = row['event'].lower().replace(' ', '_')
        
        # Update text selection logic
        if pd.notna(row.get('content_en')) and row['content_en'] != '':
            text_for_prediction = row['content_en']
        elif pd.notna(row.get('title_en')) and row['title_en'] != '':
            text_for_prediction = row['title_en']
        elif pd.notna(row.get('content')) and row['content'] != '':
            text_for_prediction = row['content']
        elif pd.notna(row.get('title')) and row['title'] != '':
            text_for_prediction = row['title']
        else:
            logger.warning(f"Skipping prediction for row {index}: No content or title available")
            continue

        # Predict move
        if 'predicted_move' not in df.columns or pd.isnull(row['predicted_move']):
            if event not in move_models:
                move_models[event], move_vectorizers[event] = load_models(event, 'regression')
            
            model = move_models[event] if move_models[event] else all_events_move_model
            vectorizer = move_vectorizers[event] if move_vectorizers[event] else all_events_move_vectorizer
            
            if model and vectorizer:
                try:
                    logger.info(f"Predicting move for row {index}, event: {event}")
                    transformed_content = vectorizer.transform([text_for_prediction])
                    prediction = model.predict(transformed_content)
                    df.at[index, 'predicted_move'] = prediction[0]
                    logger.info(f"Move prediction for row {index}: {prediction[0]}")
                except Exception as e:
                    logger.error(f"Error predicting move for row {index}: {e}", exc_info=True)
                    df.at[index, 'predicted_move'] = None
            else:
                logger.error(f"No move model available for event: {event} or all_events")
                df.at[index, 'predicted_move'] = None
        
        # Predict side
        if 'predicted_side' not in df.columns or pd.isnull(row['predicted_side']):
            if event not in side_models:
                side_models[event], side_vectorizers[event] = load_models(event, 'classifier_binary')
            
            model = side_models[event] if side_models[event] else all_events_side_model
            vectorizer = side_vectorizers[event] if side_vectorizers[event] else all_events_side_vectorizer
            
            if model and vectorizer:
                try:
                    logger.info(f"Predicting side for row {index}, event: {event}")
                    transformed_content = vectorizer.transform([text_for_prediction])
                    prediction = model.predict(transformed_content)
                    df.at[index, 'predicted_side'] = 'UP' if prediction[0] == 1 else 'DOWN'
                    logger.info(f"Side prediction for row {index}: {df.at[index, 'predicted_side']}")
                except Exception as e:
                    logger.error(f"Error predicting side for row {index}: {e}", exc_info=True)
                    df.at[index, 'predicted_side'] = None
            else:
                logger.error(f"No side model available for event: {event} or all_events")
                df.at[index, 'predicted_side'] = None
        else:
            logger.info(f"Skipping side prediction for row {index}: predicted_side is not null")
    
    return df

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

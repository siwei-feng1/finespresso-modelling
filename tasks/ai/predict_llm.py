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

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from utils.db.news_db_util import get_news_df
from utils.db.price_move_db_util import get_price_moves
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

# Verify OpenAI API key is available
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

class PricePrediction(BaseModel):
    """Pydantic model for structured price prediction output"""
    prediction: str = Field(description="Predicted direction: 'UP' or 'DOWN'")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief reasoning for the prediction")

def load_few_shot_examples(n_samples: int = 5) -> pd.DataFrame:
    """
    Load n samples per event type for few-shot learning
    
    Args:
        n_samples: Number of samples to load per event type
        
    Returns:
        DataFrame with few-shot examples
    """
    logger.info(f"Loading {n_samples} samples per event type for few-shot learning...")
    
    try:
        # Get price moves data (which already includes joined news data)
        df = get_price_moves()
        logger.info(f"Loaded {len(df)} total records from database")
        
        if df.empty:
            logger.warning("No data returned from database")
            return pd.DataFrame()
        
        # Filter out 'unknown' values and ensure we have valid data
        df = df[df['actual_side'].isin(['UP', 'DOWN'])]
        logger.info(f"After filtering, {len(df)} records with valid actual_side")
        
        # Get unique events
        events = df['event'].unique()
        logger.info(f"Found {len(events)} unique events: {events}")
        
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
                logger.warning(f"Event '{event}' has only {len(event_df)} items, using all")
            
            sampled_data.append(sampled)
        
        if sampled_data:
            result_df = pd.concat(sampled_data, ignore_index=True)
            logger.info(f"Final few-shot dataset: {len(result_df)} records across {len(events)} events")
            return result_df
        else:
            logger.error("No valid data found after sampling")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading few-shot examples: {str(e)}")
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
    
    logger.info(f"Created {len(examples)} few-shot examples")
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
        examples=examples,
        separator="\n\n"
    )
    
    return few_shot_prompt

def create_main_prompt_template() -> ChatPromptTemplate:
    """
    Create the main prompt template for prediction
    
    Returns:
        ChatPromptTemplate
    """
    system_template = """You are a financial analyst specializing in predicting stock price movements based on news events. 

Your task is to analyze news content and predict whether a stock's price will move UP or DOWN.

Consider these factors:
- The type of event (clinical trial results, FDA approval, earnings, etc.)
- The sentiment and content of the news
- The company's industry and market context
- Historical patterns for similar events

For the confidence score:
- 0.9-1.0: Very high confidence (clear, strong signals)
- 0.7-0.8: High confidence (good signals with some uncertainty)
- 0.5-0.6: Moderate confidence (mixed signals)
- 0.3-0.4: Low confidence (weak or conflicting signals)
- 0.1-0.2: Very low confidence (unclear or insufficient information)

Return your prediction in the following JSON format:
{
    "prediction": "UP" or "DOWN",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your reasoning"
}"""

    human_template = """Analyze this news and predict the stock price movement:

News: {news_text}
Event Type: {event_type}

{format_instructions}"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

def predict_single_news(news_text: str, event_type: str, llm: ChatOpenAI, 
                       few_shot_prompt: FewShotChatMessagePromptTemplate,
                       main_prompt: ChatPromptTemplate,
                       parser: PydanticOutputParser) -> Optional[PricePrediction]:
    """
    Make a prediction for a single news item using LLM with few-shot learning
    
    Args:
        news_text: The news text to analyze
        event_type: The type of event
        llm: The LangChain LLM instance
        few_shot_prompt: Few-shot prompt template
        main_prompt: Main prompt template
        parser: Output parser
        
    Returns:
        PricePrediction object or None if prediction fails
    """
    try:
        # Truncate text if too long
        truncated_text = news_text[:500] + "..." if len(news_text) > 500 else news_text
        
        # Create the full prompt with few-shot examples
        full_prompt = main_prompt.format_messages(
            news_text=truncated_text,
            event_type=event_type,
            format_instructions=parser.get_format_instructions()
        )
        
        # Add few-shot examples to the prompt
        few_shot_messages = few_shot_prompt.format_messages()
        full_messages = few_shot_messages + full_prompt
        
        # Get prediction from LLM
        response = llm.invoke(full_messages)
        
        # Parse the response
        prediction = parser.parse(response.content)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error making LLM prediction: {str(e)}")
        return None

def predict_news_batch(news_df: pd.DataFrame, n_samples: int = 5) -> pd.DataFrame:
    """
    Predict price movements for a batch of news items
    
    Args:
        news_df: DataFrame with news items to predict
        n_samples: Number of few-shot examples per event type
        
    Returns:
        DataFrame with predictions added
    """
    logger.info(f"Starting batch prediction for {len(news_df)} news items")
    
    # Load few-shot examples
    few_shot_data = load_few_shot_examples(n_samples)
    
    if few_shot_data.empty:
        logger.error("No few-shot examples available. Cannot make predictions.")
        return news_df
    
    # Create few-shot examples
    examples = create_few_shot_examples(few_shot_data)
    
    if not examples:
        logger.error("No valid few-shot examples created. Cannot make predictions.")
        return news_df
    
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
    
    # Add prediction columns to DataFrame
    news_df['predicted_side_llm'] = None
    news_df['predicted_confidence_llm'] = None
    news_df['predicted_reasoning_llm'] = None
    news_df['prediction_time_llm'] = None
    
    # Make predictions
    successful_predictions = 0
    total_predictions = 0
    
    for idx, row in news_df.iterrows():
        total_predictions += 1
        
        # Get text for prediction
        text = (row['content_en'] if pd.notna(row['content_en']) and row['content_en'] != '' 
               else row['title_en'] if pd.notna(row['title_en']) and row['title_en'] != '' 
               else row['content'] if pd.notna(row['content']) and row['content'] != '' 
               else row['title'])
        
        if pd.isna(text) or text == '':
            logger.warning(f"No text available for news item {idx}")
            continue
        
        # Get event type
        event_type = row.get('event', 'unknown')
        
        # Make prediction
        start_time = time.time()
        prediction = predict_single_news(
            text, event_type, llm, few_shot_prompt, main_prompt, parser
        )
        prediction_time = time.time() - start_time
        
        if prediction:
            # Store prediction results
            news_df.at[idx, 'predicted_side_llm'] = prediction.prediction
            news_df.at[idx, 'predicted_confidence_llm'] = prediction.confidence
            news_df.at[idx, 'predicted_reasoning_llm'] = prediction.reasoning
            news_df.at[idx, 'prediction_time_llm'] = prediction_time
            
            successful_predictions += 1
            
            logger.info(f"Prediction {successful_predictions}: {prediction.prediction} "
                       f"(confidence: {prediction.confidence:.2f}, time: {prediction_time:.2f}s)")
        else:
            logger.warning(f"Failed to get prediction for news item {idx}")
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    logger.info(f"Batch prediction completed: {successful_predictions}/{total_predictions} successful")
    
    return news_df

def predict_single_news_item(news_text: str, event_type: str = "unknown", n_samples: int = 5) -> Optional[Dict[str, Any]]:
    """
    Predict price movement for a single news item
    
    Args:
        news_text: The news text to analyze
        event_type: The type of event (default: "unknown")
        n_samples: Number of few-shot examples per event type
        
    Returns:
        Dictionary with prediction results or None if failed
    """
    logger.info(f"Predicting for single news item (event: {event_type})")
    
    # Load few-shot examples
    few_shot_data = load_few_shot_examples(n_samples)
    
    if few_shot_data.empty:
        logger.error("No few-shot examples available. Cannot make prediction.")
        return None
    
    # Create few-shot examples
    examples = create_few_shot_examples(few_shot_data)
    
    if not examples:
        logger.error("No valid few-shot examples created. Cannot make prediction.")
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
    
    # Make prediction
    start_time = time.time()
    prediction = predict_single_news(
        news_text, event_type, llm, few_shot_prompt, main_prompt, parser
    )
    prediction_time = time.time() - start_time
    
    if prediction:
        result = {
            'predicted_side': prediction.prediction,
            'predicted_confidence': prediction.confidence,
            'predicted_reasoning': prediction.reasoning,
            'prediction_time': prediction_time,
            'event_type': event_type
        }
        
        logger.info(f"Prediction successful: {prediction.prediction} "
                   f"(confidence: {prediction.confidence:.2f}, time: {prediction_time:.2f}s)")
        
        return result
    else:
        logger.error("Failed to get prediction")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict price movements using LLM few-shot learning')
    parser.add_argument('--mode', choices=['batch', 'single'], default='batch',
                       help='Prediction mode: "batch" for multiple news items or "single" for one item (default: batch)')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of few-shot examples per event type (default: 5)')
    parser.add_argument('--news_text', type=str, default=None,
                       help='News text for single prediction mode')
    parser.add_argument('--event_type', type=str, default='unknown',
                       help='Event type for single prediction mode (default: unknown)')
    parser.add_argument('--output', type=str, default='predictions_llm.csv',
                       help='Output file for batch predictions (default: predictions_llm.csv)')
    
    args = parser.parse_args()
    
    logger.info("Starting LLM prediction")
    logger.info(f"Mode: {args.mode}, Few-shot samples: {args.n_samples}")
    
    if args.mode == 'single':
        if not args.news_text:
            logger.error("News text is required for single prediction mode")
            return
        
        # Single prediction
        result = predict_single_news_item(args.news_text, args.event_type, args.n_samples)
        
        if result:
            print("\n" + "=" * 60)
            print("LLM PREDICTION RESULT")
            print("=" * 60)
            print(f"Predicted Direction: {result['predicted_side']}")
            print(f"Confidence: {result['predicted_confidence']:.2f}")
            print(f"Reasoning: {result['predicted_reasoning']}")
            print(f"Event Type: {result['event_type']}")
            print(f"Prediction Time: {result['prediction_time']:.2f}s")
            print("=" * 60)
        else:
            logger.error("Prediction failed")
    
    else:
        # Batch prediction
        logger.info("Loading news data for batch prediction...")
        
        # Load news data
        news_df = get_news_df()
        
        if news_df.empty:
            logger.error("No news data available for prediction")
            return
        
        logger.info(f"Loaded {len(news_df)} news items")
        
        # Make predictions
        predictions_df = predict_news_batch(news_df, args.n_samples)
        
        # Save results
        output_file = os.path.join('reports', args.output)
        os.makedirs('reports', exist_ok=True)
        predictions_df.to_csv(output_file, index=False)
        
        logger.info(f"Predictions saved to {output_file}")
        
        # Print summary
        successful_predictions = predictions_df['predicted_side_llm'].notna().sum()
        total_predictions = len(predictions_df)
        
        print("\n" + "=" * 60)
        print("BATCH PREDICTION SUMMARY")
        print("=" * 60)
        print(f"Total news items: {total_predictions}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Success rate: {(successful_predictions/total_predictions)*100:.1f}%")
        
        if successful_predictions > 0:
            up_predictions = (predictions_df['predicted_side_llm'] == 'UP').sum()
            down_predictions = (predictions_df['predicted_side_llm'] == 'DOWN').sum()
            avg_confidence = predictions_df['predicted_confidence_llm'].mean()
            avg_time = predictions_df['prediction_time_llm'].mean()
            
            print(f"UP predictions: {up_predictions}")
            print(f"DOWN predictions: {down_predictions}")
            print(f"Average confidence: {avg_confidence:.2f}")
            print(f"Average prediction time: {avg_time:.2f}s")
        
        print("=" * 60)
    
    logger.info("LLM prediction completed")

if __name__ == '__main__':
    main() 
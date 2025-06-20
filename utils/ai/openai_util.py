import os
from openai import OpenAI
from dotenv import load_dotenv
from gptcache import cache
import logging
from utils.ai.language_util import normalize_language_code
import json

load_dotenv()

# Set up OpenAI client with explicit API key
api_key = os.getenv('OPENAI_API_KEY')
model_name = os.getenv('OPENAI_MODEL')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)
cache.init()
cache.set_openai_key()

# Load environment variables
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

 # Changed from "gpt-4o" to "gpt-4"

def tag_news(news, tags):
    prompt = f'Answering with one tag only, pick up the best tag which describes the news "{news}" from the list: {tags}'
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    tag = response.choices[0].message.content
    return tag

def enrich_reason(content, predicted_move):
    system_prompt = """You are a financial analyst providing concise market insights. Your responses should be clear and readable without using any special characters or explicit formatting such as new lines. Use standard punctuation and avoid line breaks within sentences. Separate ideas with periods and commas as needed."""

    if predicted_move is not None:
        direction = "up" if predicted_move > 0 else "down"
        user_prompt = f"""Analyze: "{content}" Asset predicted to move {direction} by {predicted_move:+.2f}%. In less than 40 words: 1. Explain the likely cause of this {direction}ward movement. 2. Briefly discuss potential market implications. 3. Naturally include "predicted {direction}ward move of {predicted_move:+.2f}%". Be concise yet comprehensive. Ensure a complete response with no cut-off sentences."""
    else:
        user_prompt = f'In less than 40 words, summarize the potential market impact of this news. Ensure a complete response with no cut-off sentences: "{content}"'
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=80  # Adjusted for up to 40 words
    )
    reason = response.choices[0].message.content.strip()
    return reason

def extract_ticker(company):
    prompt = f'Extract the company or issuer ticker symbol corresponding to the company name provided. Return only the ticker symbol in uppercase, without any additional text. If you cannot assign a ticker symbol, return "N/A". Company name: "{company}"'
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    ticker = response.choices[0].message.content.strip().upper()
    return ticker if ticker != "N/A" else None

def extract_issuer(news):
    prompt = f'Extract the company or issuer name corresponding to the text provided. Return concise entity name only. If you cannot assign a ticker symbol, return "N/A". News: "{news}"'
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    ticker = response.choices[0].message.content.strip().upper()
    return ticker if ticker != "N/A" else None

def detect_language(title):
    """Detect the language of a news title using GPT."""
    prompt = f'Detect the language of this text and return only the 2-letter ISO language code (e.g. "en" for English, "sv" for Swedish). If you cannot determine the exact ISO code, return the full language name: "{title}"'
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    language = response.choices[0].message.content.strip().lower()
    
    # Use the normalized language code function
    return normalize_language_code(language)

def translate_to_english(text, source_language):
    """Translate text from source language to English using GPT."""
    prompt = f'Translate the following text from {source_language} to English. Return only the translation without any additional text or formatting: "{text}"'
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_prediction_from_openai(content):
    """
    Get price movement prediction from OpenAI.
    Returns a dictionary with 'move' (float) and 'side' (str) predictions.
    """
    system_prompt = """You are a financial market analyst. Analyze news content and predict potential price movements. 
    Return only a JSON object with two fields:
    - 'move': predicted percentage move (float between 0.1 and 10.0)
    - 'side': direction ('up' or 'down')
    For neutral or unclear news, return null for both fields."""

    user_prompt = f'Predict the likely price movement based on this news: "{content}"'
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        prediction = response.choices[0].message.content
        
        # Convert the string response to a Python dictionary
        result = json.loads(prediction)
        
        # Validate the prediction
        if result.get('move') is not None and result.get('side') is not None:
            move = float(result['move'])
            side = result['side'].lower()
            
            # Ensure move is positive and side is valid
            if move > 0 and side in ['up', 'down']:
                # Make move negative if side is down
                if side == 'down':
                    move = -move
                return {'move': move, 'side': side}
        
        return None
        
    except Exception as e:
        logging.error(f"Error getting prediction from OpenAI: {str(e)}")
        return None
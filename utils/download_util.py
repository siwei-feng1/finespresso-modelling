import pandas as pd
from utils.db.news_db_util import map_to_db, add_news_items, remove_duplicates
from utils.enrich_util import enrich_tag_from_url
from tasks.ai.predict import predict
from tasks.enrich.enrich_reason import enrich_reason  # Import directly from openai_util
from utils.logging.log_util import get_logger
from utils.db.instrument_db_util import get_instrument_by_company_name, save_instrument, insert_instrument
from utils.translate_util import translate_dataframe
from utils.yf_util import search_instrument_info

logger = get_logger(__name__)

def process_download(df, publisher, check_uniqueness=True):
    news_items = map_to_db(df, publisher)
    
    if check_uniqueness:
        unique_items, duplicate_count = remove_duplicates(news_items)
        logger.info(f"{publisher.upper()}: Removed {duplicate_count} duplicate items")
        news_items = unique_items
    
    if not news_items:
        logger.info(f"{publisher.upper()}: No items to process after removing duplicates")
        return 0

    logger.info(f"{publisher.upper()}: Processing {len(news_items)} scraped items")

    # Convert news_items to DataFrame for further processing
    df = pd.DataFrame([item.__dict__ for item in news_items])

    # Enrich instrument ID
    logger.info("Starting instrument ID enrichment")
    new_instruments = []
    
    for index, row in df.iterrows():
        company = row['company']
        if not company or str(company).strip() in ['.', '']:
            continue
            
        instrument = get_instrument_by_company_name(company)
        if instrument:
            # Handle both dictionary and object returns
            instrument_id = instrument.get('id') if isinstance(instrument, dict) else getattr(instrument, 'id', None)
            ticker = instrument.get('ticker') if isinstance(instrument, dict) else getattr(instrument, 'ticker', None)
            yf_ticker = instrument.get('yf_ticker') if isinstance(instrument, dict) else getattr(instrument, 'yf_ticker', None)
            url = instrument.get('url') if isinstance(instrument, dict) else getattr(instrument, 'url', None)
            
            if instrument_id:
                df.at[index, 'instrument_id'] = instrument_id
                df.at[index, 'ticker'] = ticker
                df.at[index, 'yf_ticker'] = yf_ticker
                df.at[index, 'ticker_url'] = url
                logger.info(f"Enriched instrument ID for {company}: {instrument_id}")
            else:
                logger.warning(f"Found instrument for {company} but ID was missing")
        else:
            # Try to find instrument info using yfinance
            instrument_info = search_instrument_info(company)
            if instrument_info:
                try:
                    # Create new instrument and get its ID
                    new_instrument = insert_instrument(instrument_info)
                    if new_instrument and new_instrument.id:
                        logger.info(f"Successfully created instrument with ID {new_instrument.id} of type {type(new_instrument.id)}")
                        df.at[index, 'instrument_id'] = int(new_instrument.id)  # Ensure it's an integer
                        df.at[index, 'ticker'] = new_instrument.ticker
                        df.at[index, 'yf_ticker'] = new_instrument.yf_ticker
                        df.at[index, 'ticker_url'] = new_instrument.url
                        new_instruments.append(new_instrument)
                        logger.info(f"Created and enriched new instrument for {company}: {new_instrument.id}")
                    else:
                        logger.warning(f"Failed to create instrument for {company} - no ID returned")
                        df.at[index, 'instrument_id'] = None
                except Exception as e:
                    logger.error(f"Error creating instrument for {company}: {str(e)}")
                    df.at[index, 'instrument_id'] = None
            else:
                logger.info(f"No instrument found and couldn't create one for company: {company}")
    
    if new_instruments:
        logger.info(f"Created {len(new_instruments)} new instruments")
    logger.info("Instrument ID enrichment completed")

    # 1. Enrich event
    try:
        df = enrich_tag_from_url(df)
        logger.info("Event enrichment completed successfully.")
        for index, row in df.iterrows():
            logger.info(f"Event for {row['link']}: {row['event']}")
    except Exception as e:
        logger.error(f"Error during event enrichment: {str(e)}", exc_info=True)
    
    # Log the dataframe state before translation
    logger.info(f"Before translation - Content sample: {df['content'].iloc[0][:200] if not df.empty else 'No content'}")
    logger.info(f"Before translation - Number of rows: {len(df)}")
    
    # 2. Enrich language and translate if needed
    try:
        df = translate_dataframe(df)
        logger.info("Translation completed successfully.")
        logger.info(f"After translation - Content sample: {df['content'].iloc[0][:200] if not df.empty else 'No content'}")
        logger.info(f"After translation - Number of rows: {len(df)}")
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}", exc_info=True)

    # 3. Add predictions using AI model
    try:
        df = predict(df)
        logger.info("Prediction completed successfully.")
        logger.info(f"Predictions added - Number of rows with predictions: {df['predicted_move'].notna().sum()}")
        # Add sample prediction logging
        if not df.empty and df['predicted_move'].notna().any():
            sample_idx = df[df['predicted_move'].notna()].index[0]
            logger.info(f"Sample prediction - Move: {df.at[sample_idx, 'predicted_move']}, "
                       f"Side: {df.at[sample_idx, 'predicted_side']}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)

    # 4. Enrich reason
    try:
        df = enrich_reason(df)
        logger.info("Reason enrichment completed successfully.")
    except Exception as e:
        logger.error(f"Error during reason enrichment: {str(e)}", exc_info=True)
        
        # Log all reasons after enrichment
        logger.info("=== All Reasons After Enrichment ===")
        logger.info(f"Dataframe shape: {df.shape}")
        logger.info(f"Columns in dataframe: {df.columns.tolist()}")
        for idx, row in df.iterrows():
            if pd.notna(row['reason']):
                logger.info(f"Row {idx} reason: {row['reason']}")
            else:
                logger.info(f"Row {idx} has no reason")
        
        reasons_count = df['reason'].notna().sum()
        logger.info(f"Total reasons added: {reasons_count}")
                
    except Exception as e:
        logger.error(f"Error during reason enrichment: {str(e)}", exc_info=True)
    
    # Log before database insertion
    logger.info(f"Before database insertion - Number of rows: {len(df)}")
    logger.info(f"Before database insertion - Columns: {df.columns.tolist()}")
    
    # Map enriched DataFrame back to news items
    enriched_news_items = map_to_db(df, publisher)
    logger.info(f"Created {len(enriched_news_items)} news items for database insertion")

    # Add all items to the database
    added_count, _ = add_news_items(enriched_news_items, check_uniqueness=False)
    logger.info(f"{publisher.upper()}: added {added_count} news items to the database")

    return added_count

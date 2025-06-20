from utils.db_pool import DatabasePool
from sqlalchemy import Column, BigInteger, String, Text, Float, func, or_
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import logging
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the database pool instance
db_pool = DatabasePool()
Base = declarative_base()

class Instrument(Base):
    __tablename__ = 'instrument'

    id = Column(BigInteger, primary_key=True)
    issuer = Column(String(255))
    ticker = Column(String(100))
    yf_ticker = Column(String(100))
    isin = Column(String(100))
    asset_class = Column(String(100))
    sector = Column(String(100))
    exchange = Column(String(100))
    exchange_code = Column(String(100))
    country = Column(String(100))
    url = Column(Text)

def insert_instrument(instrument_data):
    """
    Insert a single instrument and return the created instrument with its ID.
    
    Args:
        instrument_data (dict): Dictionary containing instrument data
    
    Returns:
        tuple: (dict, str) - The created instrument data as a dictionary and a success message
    """
    session = db_pool.get_session()
    try:
        logging.info(f"Inserting new instrument for issuer: {instrument_data.get('issuer')}")
        
        # Create instrument instance
        instrument = Instrument(
            issuer=str(instrument_data.get('issuer', '')),
            ticker=str(instrument_data.get('ticker', '')),
            yf_ticker=str(instrument_data.get('yf_ticker', '')),
            isin=str(instrument_data.get('isin', '')),
            asset_class=str(instrument_data.get('asset_class', '')),
            sector=str(instrument_data.get('sector', '')),
            exchange=str(instrument_data.get('exchange', '')),
            exchange_code=str(instrument_data.get('exchange_code', '')),
            country=str(instrument_data.get('country', '')),
            url=str(instrument_data.get('url', ''))
        )
        
        # Add and flush to get the ID
        session.add(instrument)
        session.flush()
        
        # Convert to dictionary with all fields
        instrument_dict = {
            'id': instrument.id,
            'issuer': instrument.issuer,
            'ticker': instrument.ticker,
            'yf_ticker': instrument.yf_ticker,
            'isin': instrument.isin,
            'asset_class': instrument.asset_class,
            'sector': instrument.sector,
            'exchange': instrument.exchange,
            'exchange_code': instrument.exchange_code,
            'country': instrument.country,
            'url': instrument.url,
            'float_ratio': instrument_data.get('float_ratio', ''),
            'market_cap_class': instrument_data.get('market_cap_class', '')
        }
        
        logging.info(f"Created new instrument with ID: {instrument.id} for issuer: {instrument.issuer}")
        
        # Commit the transaction
        session.commit()
        
        return instrument_dict, f"Successfully inserted instrument for {instrument.issuer}"
        
    except Exception as e:
        session.rollback()
        logging.error(f"Error inserting instrument: {str(e)}")
        return None, f"Error inserting instrument: {str(e)}"
    finally:
        db_pool.return_session(session)

def save_instrument(df):
    """
    Save multiple instruments from a DataFrame.
    Returns list of created instruments.
    """
    created_instruments = []
    logging.info(f"Starting to save {len(df)} instruments")
    
    for _, row in df.iterrows():
        try:
            instrument_data = {
                'issuer': row.get('issuer'),
                'ticker': row.get('ticker'),
                'yf_ticker': row.get('yf_ticker'),
                'isin': row.get('isin'),
                'asset_class': row.get('asset_class'),
                'sector': row.get('sector'),
                'exchange': row.get('exchange'),
                'exchange_code': row.get('exchange_code'),
                'country': row.get('country'),
                'url': row.get('url')
            }
            
            # Use insert_instrument for each row
            instrument, message = insert_instrument(instrument_data)
            created_instruments.append(instrument)
            
        except Exception as e:
            logging.error(f"Error saving instrument row: {str(e)}")
            continue
    
    logging.info(f"Successfully saved {len(created_instruments)} instruments")
    return created_instruments

def get_instrument_by_ticker(ticker):
    """
    Retrieve an instrument by its ticker symbol.
    
    Args:
        ticker (str): The ticker symbol to search for
        
    Returns:
        Instrument: The instrument object if found, None otherwise
    """
    try:
        return db_pool.get_session().query(Instrument).filter(
            func.lower(Instrument.ticker) == func.lower(ticker)
        ).first()
    except Exception as e:
        logging.error(f"Error getting instrument by ticker {ticker}: {str(e)}")
        return None

def get_instrument_by_company_name(company_name):
    logging.info(f"Looking up instrument for company: {company_name}")
    
    session = db_pool.get_session()
    try:
        instruments = session.query(Instrument).filter(
            or_(
                func.lower(Instrument.issuer) == func.lower(company_name),
                func.lower(Instrument.issuer).like(f"%{company_name.lower()}%")
            )
        ).all()
        
        if instruments:
            instrument = instruments[0]
            logging.info(f"Found instrument for {company_name}: {instrument.ticker}")
            if len(instruments) > 1:
                logging.warning(f"Multiple instruments found for {company_name}. Using the first match: {instrument.ticker}")
            
            # Convert to dictionary
            return {
                'id': instrument.id,
                'issuer': instrument.issuer,
                'ticker': instrument.ticker,
                'yf_ticker': instrument.yf_ticker,
                'isin': instrument.isin,
                'asset_class': instrument.asset_class,
                'sector': instrument.sector,
                'exchange': instrument.exchange,
                'exchange_code': instrument.exchange_code,
                'country': instrument.country,
                'url': instrument.url
            }
        else:
            logging.info(f"No instrument found for {company_name}")
            return None
            
    except ProgrammingError as e:
        if 'relation "instrument" does not exist' in str(e):
            logging.error("The 'instrument' table does not exist in the database. Please ensure the table is created and populated.")
        else:
            logging.error(f"An error occurred while querying the database: {str(e)}")
        return None
    finally:
        db_pool.return_session(session)

# Add this at the end of the file to recreate the table if needed
if __name__ == "__main__":
    logging.info("Recreating Instrument table")
    Base.metadata.drop_all(db_pool.engine, tables=[Instrument.__table__])
    Base.metadata.create_all(db_pool.engine, tables=[Instrument.__table__])
    logging.info("Instrument table recreated successfully")

# Add this new function
def create_and_get_instrument(instrument_data):
    """
    Create a new instrument and return its data within a single session.
    
    Args:
        instrument_data (dict): Dictionary containing instrument data
    
    Returns:
        Instrument: The created instrument object with its ID
    """
    session = db_pool.get_session()
    try:
        logging.info(f"Creating new instrument for issuer: {instrument_data.get('issuer')}")
        
        # Create instrument instance
        instrument = Instrument(
            issuer=str(instrument_data.get('issuer', '')),
            ticker=str(instrument_data.get('ticker', '')),
            yf_ticker=str(instrument_data.get('yf_ticker', '')),
            isin=str(instrument_data.get('isin', '')),
            asset_class=str(instrument_data.get('asset_class', '')),
            sector=str(instrument_data.get('sector', '')),
            exchange=str(instrument_data.get('exchange', '')),
            exchange_code=str(instrument_data.get('exchange_code', '')),
            country=str(instrument_data.get('country', '')),
            url=str(instrument_data.get('url', ''))
        )
        
        # Add and commit to get the ID
        session.add(instrument)
        session.commit()
        
        # Refresh the instance to ensure all attributes are loaded
        session.refresh(instrument)
        
        # Create a new instance with all the data
        instrument_copy = Instrument(
            id=instrument.id,
            issuer=instrument.issuer,
            ticker=instrument.ticker,
            yf_ticker=instrument.yf_ticker,
            isin=instrument.isin,
            asset_class=instrument.asset_class,
            sector=instrument.sector,
            exchange=instrument.exchange,
            exchange_code=instrument.exchange_code,
            country=instrument.country,
            url=instrument.url
        )
        
        logging.info(f"Created new instrument with ID: {instrument_copy.id} for issuer: {instrument_copy.issuer}")
        
        return instrument_copy
        
    except Exception as e:
        session.rollback()
        logging.error(f"Error creating instrument: {str(e)}")
        raise
    finally:
        db_pool.return_session(session)

# Add this new function after the other functions
def get_distinct_instrument_fields():
    """
    Get distinct values for various instrument fields to populate select boxes.
    
    Returns:
        dict: Dictionary containing lists of distinct values for each field
    """
    session = db_pool.get_session()
    try:
        logging.info("Retrieving distinct instrument fields")
        
        # Query distinct values for each field
        distinct_fields = {
            'asset_classes': [x[0] for x in session.query(Instrument.asset_class).distinct().all() if x[0]],
            'exchanges': [x[0] for x in session.query(Instrument.exchange).distinct().all() if x[0]],
            'exchange_codes': [x[0] for x in session.query(Instrument.exchange_code).distinct().all() if x[0]],
            'countries': [x[0] for x in session.query(Instrument.country).distinct().all() if x[0]],
            'sectors': [x[0] for x in session.query(Instrument.sector).distinct().all() if x[0]]
        }
        
        # Sort all lists
        for key in distinct_fields:
            distinct_fields[key] = sorted(distinct_fields[key])
            
        logging.info("Successfully retrieved distinct instrument fields")
        return distinct_fields
        
    except Exception as e:
        logging.error(f"Error retrieving distinct instrument fields: {str(e)}")
        # Return empty lists as fallback
        return {
            'asset_classes': [],
            'exchanges': [],
            'exchange_codes': [],
            'countries': [],
            'sectors': []
        }
    finally:
        db_pool.return_session(session)

# Add these new functions after the existing ones

def get_all_instruments():
    """
    Retrieve all instruments from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing all instruments
    """
    session = db_pool.get_session()
    try:
        logging.info("Retrieving all instruments")
        instruments = session.query(Instrument).all()
        
        # Convert to list of dictionaries
        instruments_data = []
        for instrument in instruments:
            instrument_dict = {
                'id': instrument.id,
                'issuer': instrument.issuer,
                'ticker': instrument.ticker,
                'yf_ticker': instrument.yf_ticker,
                'isin': instrument.isin,
                'asset_class': instrument.asset_class,
                'sector': instrument.sector,
                'exchange': instrument.exchange,
                'exchange_code': instrument.exchange_code,
                'country': instrument.country,
                'url': instrument.url
            }
            instruments_data.append(instrument_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(instruments_data)
        logging.info(f"Retrieved {len(df)} instruments")
        return df
        
    except Exception as e:
        logging.error(f"Error retrieving all instruments: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        db_pool.return_session(session)

def delete_instruments(instrument_ids):
    """
    Delete multiple instruments by their IDs.
    
    Args:
        instrument_ids (list): List of instrument IDs to delete
    """
    session = db_pool.get_session()
    try:
        logging.info(f"Deleting instruments with IDs: {instrument_ids}")
        
        # Delete instruments with the given IDs
        deleted = session.query(Instrument).filter(
            Instrument.id.in_(instrument_ids)
        ).delete(synchronize_session=False)
        
        session.commit()
        logging.info(f"Successfully deleted {deleted} instruments")
        
    except Exception as e:
        session.rollback()
        logging.error(f"Error deleting instruments: {str(e)}")
        raise
    finally:
        db_pool.return_session(session)

def get_instrument_by_yf_ticker(yf_ticker):
    """
    Retrieve an instrument by its Yahoo Finance ticker.
    
    Args:
        yf_ticker (str): The Yahoo Finance ticker to search for
        
    Returns:
        Instrument: The instrument object if found, None otherwise
    """
    session = db_pool.get_session()
    try:
        logging.info(f"Looking up instrument for YF ticker: {yf_ticker}")
        
        instrument = session.query(Instrument).filter(
            func.lower(Instrument.yf_ticker) == func.lower(yf_ticker)
        ).first()
        
        if instrument:
            logging.info(f"Found instrument for YF ticker {yf_ticker}")
        else:
            logging.info(f"No instrument found for YF ticker {yf_ticker}")
            
        return instrument
        
    except Exception as e:
        logging.error(f"Error getting instrument by YF ticker {yf_ticker}: {str(e)}")
        return None
    finally:
        db_pool.return_session(session)

# Add this new function after the other functions
def update_instrument(instrument_data):
    """
    Update an existing instrument in the database.
    
    Args:
        instrument_data (dict): Dictionary containing instrument data with ID
        
    Returns:
        dict: Updated instrument data if successful, None if failed
    """
    session = db_pool.get_session()
    try:
        logging.info(f"Updating instrument ID: {instrument_data.get('id')}")
        
        # Get the instrument by ID
        db_instrument = session.query(Instrument).get(instrument_data['id'])
        if db_instrument:
            # Update the fields
            for key, value in instrument_data.items():
                if hasattr(db_instrument, key):
                    setattr(db_instrument, key, value)
            
            # Commit the changes
            session.commit()
            
            # Convert to dictionary for return
            updated_instrument = {
                'id': db_instrument.id,
                'issuer': db_instrument.issuer,
                'ticker': db_instrument.ticker,
                'yf_ticker': db_instrument.yf_ticker,
                'isin': db_instrument.isin,
                'asset_class': db_instrument.asset_class,
                'sector': db_instrument.sector,
                'exchange': db_instrument.exchange,
                'exchange_code': db_instrument.exchange_code,
                'country': db_instrument.country,
                'url': db_instrument.url,
                'float_ratio': instrument_data.get('float_ratio', ''),
                'market_cap_class': instrument_data.get('market_cap_class', '')
            }
            
            logging.info(f"Successfully updated instrument ID: {db_instrument.id}")
            return updated_instrument
        else:
            logging.error(f"No instrument found with ID: {instrument_data.get('id')}")
            return None
            
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating instrument: {str(e)}")
        raise
    finally:
        db_pool.return_session(session)

def _instrument_to_df(instrument):
    """
    Internal helper function to convert an instrument object to DataFrame
    """
    if instrument is None:
        return pd.DataFrame()
        
    if isinstance(instrument, dict):
        return pd.DataFrame([instrument])
        
    return pd.DataFrame([{
        'id': instrument.id,
        'issuer': instrument.issuer,
        'ticker': instrument.ticker,
        'yf_ticker': instrument.yf_ticker,
        'isin': instrument.isin,
        'asset_class': instrument.asset_class,
        'sector': instrument.sector,
        'exchange': instrument.exchange,
        'exchange_code': instrument.exchange_code,
        'country': instrument.country,
        'url': instrument.url
    }])

def get_instrument_by_company_name_df(company_name):
    """
    Retrieve an instrument by company name and return as DataFrame
    
    Args:
        company_name (str): The company name to search for
        
    Returns:
        pd.DataFrame: DataFrame containing the instrument data if found, empty DataFrame otherwise
    """
    instrument = get_instrument_by_company_name(company_name)
    return _instrument_to_df(instrument)

def get_instrument_by_yf_ticker_df(yf_ticker):
    """
    Retrieve an instrument by Yahoo Finance ticker and return as DataFrame
    
    Args:
        yf_ticker (str): The Yahoo Finance ticker to search for
        
    Returns:
        pd.DataFrame: DataFrame containing the instrument data if found, empty DataFrame otherwise
    """
    instrument = get_instrument_by_yf_ticker(yf_ticker)
    return _instrument_to_df(instrument)

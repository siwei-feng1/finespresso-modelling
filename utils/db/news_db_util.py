import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func, and_, select, update, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import TIMESTAMP
import logging
from datetime import datetime
import pandas as pd
import streamlit as st
from sqlalchemy import exists, or_  # Add 'or_' to the imports
from utils.logging.log_util import get_logger
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import text

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Get DATABASE_URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class News(Base):
    __tablename__ = 'news'

    id = Column(Integer, primary_key=True)
    title = Column(Text)
    link = Column(Text)
    company = Column(Text)
    published_date = Column(TIMESTAMP(timezone=True))
    content = Column(Text)
    reason = Column(Text)
    industry = Column(Text)
    publisher_topic = Column(Text)
    event = Column(String(255))
    publisher = Column(String(255))
    downloaded_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    status = Column(String(255))
    instrument_id = Column(BigInteger)
    yf_ticker = Column(String(255))
    ticker = Column(String(16))
    published_date_gmt = Column(TIMESTAMP(timezone=True))
    timezone = Column(String(50))
    publisher_summary = Column(Text)
    ticker_url = Column(String(500))
    predicted_side = Column(String(10))
    predicted_move = Column(Float)
    language = Column(String(50), nullable=True)
    title_en = Column(Text, nullable=True)
    content_en = Column(Text, nullable=True)


def add_news_items(news_items, check_uniqueness=True):
    logger.info(f"Adding {len(news_items)} news items to the database")
    added_count = 0
    duplicate_count = 0

    with Session() as session:
        for item in news_items:
            if check_uniqueness:
                logger.debug(f"Checking item with content length: {len(item.content) if item.content else 0}")
                
                existing_item = session.query(News).filter(
                    News.link == item.link,
                    News.publisher == item.publisher
                ).first()

                if existing_item:
                    duplicate_count += 1
                    logger.debug(f"Duplicate found for link: {item.link}")
                    continue

            session.add(item)
            added_count += 1
            logger.debug(f"Added item with content length: {len(item.content) if item.content else 0}")

        try:
            session.commit()
            logger.info(f"Successfully committed {added_count} items to database")
        except Exception as e:
            logger.error(f"Error committing to database: {str(e)}")
            session.rollback()
            raise

    logger.info(f"Added {added_count} news items to the database, {duplicate_count} duplicates skipped")
    return added_count, duplicate_count

def remove_duplicates(news_items):
    logger.info("Checking for duplicates in the database")
    session = Session()
    try:
        unique_items = []
        duplicate_count = 0
        
        for item in news_items:
            # Check if the item already exists in the database using both link and publisher
            is_duplicate = session.query(exists().where(
                and_(
                    News.link == item.link,
                    News.publisher == item.publisher
                )
            )).scalar()
            
            if not is_duplicate:
                unique_items.append(item)
            else:
                duplicate_count += 1
                logger.debug(f"Found duplicate for link: {item.link} and publisher: {item.publisher}")
        
        logger.info(f"Found {duplicate_count} duplicate items")
        logger.info(f"Keeping {len(unique_items)} unique items")
        
        return unique_items, duplicate_count
    except Exception as e:
        logger.error(f"An error occurred while checking for duplicates: {e}")
        return [], 0
    finally:
        session.close()

def map_to_db(df, source):
    logging.info(f"Mapping dataframe to News objects for source: {source}")
    
    news_items = []
    for _, row in df.iterrows():
        # Handle instrument_id - convert nan or float to proper integer or None
        instrument_id = row.get('instrument_id')
        if pd.isna(instrument_id):
            instrument_id = None
        elif isinstance(instrument_id, float):
            instrument_id = int(instrument_id) if instrument_id.is_integer() else None
            
        news_item = News(
            title=row.get('title', ''),
            link=row.get('link', ''),
            company=row.get('company', ''),
            published_date=row.get('published_date'),
            content=row.get('content', ''),
            reason=row.get('reason', ''),
            industry=row.get('industry', ''),
            publisher_topic=row.get('publisher_topic', ''),
            event=row.get('event', ''),
            publisher=row.get('publisher', source),
            downloaded_at=datetime.utcnow(),
            status=row.get('status', 'raw'),
            instrument_id=instrument_id,  # Use the cleaned instrument_id
            yf_ticker=row.get('yf_ticker', ''),
            ticker=row.get('ticker', ''),
            published_date_gmt=row.get('published_date_gmt'),
            timezone=row.get('timezone', ''),
            publisher_summary=row.get('publisher_summary', ''),
            ticker_url=row.get('ticker_url', ''),
            predicted_side=row.get('predicted_side', None),
            predicted_move=row.get('predicted_move', None),
            language=row.get('language', ''),
            title_en=row.get('title_en'),
            content_en=row.get('content_en')
        )
        news_items.append(news_item)
    
    logging.info(f"Created {len(news_items)} News objects")
    
    return news_items

def remove_duplicate_news_db():
    session = Session()
    try:
        subquery = session.query(News.link, func.min(News.downloaded_at).label('min_downloaded_at')) \
                          .group_by(News.link) \
                          .subquery()
        
        duplicates = session.query(News.id) \
                            .join(subquery, and_(News.link == subquery.c.link,
                                                 News.downloaded_at != subquery.c.min_downloaded_at))
        
        deleted_count = session.query(News).filter(News.id.in_(duplicates)).delete(synchronize_session='fetch')
        
        updated_count = session.query(News).filter(News.status == 'raw').update({News.status: 'clean'}, synchronize_session='fetch')
        
        session.commit()
        logging.info(f"Successfully removed {deleted_count} duplicate news items.")
        logging.info(f"Updated status to 'clean' for {updated_count} news items.")
        
        return deleted_count, updated_count
    except Exception as e:
        logging.error(f"An error occurred while removing duplicates and updating status: {e}")
        session.rollback()
        return 0, 0
    finally:
        session.close()

def get_news_df_date_range(publishers, start_date, end_date):
    session = Session()
    try:
        query = select(News).where(
            News.publisher.in_(publishers),
            News.published_date >= start_date,
            News.published_date <= end_date
        ).order_by(News.published_date.desc())
        
        result = session.execute(query)
        news_items = result.scalars().all()
        
        data = [{
            'news_id': item.id,
            'ticker': item.ticker,
            'ticker_url': item.ticker_url,
            'title': item.title,
            'link': item.link,
            'published_date': item.published_date,
            'company': item.company,
            'event': item.event,
            'reason': item.reason,
            'publisher': item.publisher,
            'industry': item.industry,
            'publisher_topic': item.publisher_topic,
            'instrument_id': item.instrument_id,
            'yf_ticker': item.yf_ticker,
            'published_date_gmt': item.published_date_gmt,
            'timezone': item.timezone,
            'publisher_summary': item.publisher_summary,
            'predicted_side': item.predicted_side,
            'predicted_move': item.predicted_move,
            'event': item.event,
            'language': item.language
        } for item in news_items]
        
        return pd.DataFrame(data)
    finally:
        session.close()

def get_news_without_tickers():
    logging.info("Retrieving news items without tickers from database")
    
    session = Session()
    try:
        query = select(News).where(News.ticker.is_(None))
        result = session.execute(query)
        news_items = result.scalars().all()
        count = len(news_items)
        logging.info(f"Retrieved {count} news items without tickers")
        
        return news_items
    finally:
        session.close()

def update_news_tickers(news_items_with_data):
    logging.info("Updating database with extracted tickers, yf_tickers, and instrument IDs")
    
    session = Session()
    try:
        updated_count = 0
        total_items = len(news_items_with_data)
        for index, (news_id, ticker, yf_ticker, instrument_id, ticker_url) in enumerate(news_items_with_data):
            update_values = {}
            if ticker:
                update_values['ticker'] = ticker
            if yf_ticker:
                update_values['yf_ticker'] = yf_ticker
            if instrument_id:
                update_values['instrument_id'] = instrument_id
            if ticker_url:
                update_values['ticker_url'] = ticker_url
            
            if update_values:
                stmt = update(News).where(News.id == news_id).values(**update_values)
                session.execute(stmt)
                updated_count += 1
            
            if (index + 1) % 10 == 0 or index == total_items - 1:
                session.commit()
                logging.info(f"Processed {index + 1}/{total_items} items")
        
        logging.info(f"Successfully updated {updated_count} news items with tickers, yf_tickers, and instrument IDs")
    except Exception as e:
        logging.error(f"Error updating news items: {str(e)}")
        session.rollback()
    finally:
        session.close()

def update_news_status(news_ids, new_status):
    logging.info(f"Updating status to '{new_status}' for {len(news_ids)} news items")
    
    session = Session()
    try:
        updated_count = session.query(News).filter(News.id.in_(news_ids)).update({News.status: new_status}, synchronize_session='fetch')
        session.commit()
        logging.info(f"Successfully updated status for {updated_count} news items")
        
        return updated_count
    except Exception as e:
        logging.error(f"An error occurred while updating news status: {e}")
        session.rollback()
        return 0
    finally:
        session.close()

def get_news_without_company(publisher):
    logging.info(f"Retrieving news items without company names for publisher: {publisher}")
    
    session = Session()
    try:
        query = select(News).where(
            News.company.is_(None), 
            News.publisher == publisher
        )
        result = session.execute(query)
        news_items = result.scalars().all()
        count = len(news_items)
        logging.info(f"Retrieved {count} news items without company names for {publisher}")
        
        return news_items
    finally:
        session.close()

def update_companies(enriched_df):
    logging.info("Updating database with enriched company names")
    
    session = Session()
    try:
        updated_count = 0
        total_items = len(enriched_df)
        for index, row in enriched_df.iterrows():
            news_item = session.get(News, row['id'])
            if news_item and 'company' in row and row['company']:
                news_item.company = row['company']
                updated_count += 1
            
            if (index + 1) % 10 == 0 or index == total_items - 1:
                logging.info(f"Updated {index + 1}/{total_items} items")
        
        session.commit()
        logging.info(f"Successfully updated {updated_count} news items with company names")
    except Exception as e:
        logging.error(f"Error updating company names: {str(e)}")
        session.rollback()
    finally:
        session.close()

def get_news_by_id(news_id):
    logging.info(f"Retrieving news item with id: {news_id}")
    
    session = Session()
    try:
        query = select(News).where(News.id == news_id)
        result = session.execute(query)
        news_item = result.scalars().first()
        
        if news_item:
            return pd.DataFrame([{
                'news_id': news_item.id,
                'ticker': news_item.ticker,
                'ticker_url': news_item.ticker_url,
                'title': news_item.title,
                'link': news_item.link,
                'published_date': news_item.published_date,
                'company': news_item.company,
                'event': news_item.event,
                'reason': news_item.reason,
                'publisher': news_item.publisher,
                'industry': news_item.industry,
                'publisher_topic': news_item.publisher_topic,
                'instrument_id': news_item.instrument_id,
                'yf_ticker': news_item.yf_ticker,
                'published_date_gmt': news_item.published_date_gmt,
                'timezone': news_item.timezone,
                'publisher_summary': news_item.publisher_summary,
                'predicted_side': news_item.predicted_side,
                'predicted_move': news_item.predicted_move,
                'event': news_item.event,
                'language': news_item.language
            }])
        else:
            logging.warning(f"No news item found with id: {news_id}")
            return pd.DataFrame()
    finally:
        session.close()

def get_news_df(publisher=None):
    logging.info(f"Retrieving all news items ordered by published date{' for publisher: ' + publisher if publisher else ''}")
    
    session = Session()
    try:
        query = select(News).order_by(News.published_date.asc())
        
        if publisher:
            query = query.filter(News.publisher == publisher)
        
        result = session.execute(query)
        news_items = result.scalars().all()
        
        data = [{
            'news_id': item.id,
            'ticker': item.ticker,
            'ticker_url': item.ticker_url,
            'title': item.title,
            'link': item.link,
            'published_date': item.published_date,
            'company': item.company,
            'event': item.event,
            'reason': item.reason,  # Make sure 'reason' is included here
            'publisher': item.publisher,
            'industry': item.industry,
            'publisher_topic': item.publisher_topic,
            'instrument_id': item.instrument_id,
            'yf_ticker': item.yf_ticker,
            'published_date_gmt': item.published_date_gmt,
            'timezone': item.timezone,
            'publisher_summary': item.publisher_summary,
            'predicted_side': item.predicted_side,
            'predicted_move': item.predicted_move,
            'content': item.content,  # Include 'content' as it's used in enrich_reason
            'language': item.language
        } for item in news_items]
        
        df = pd.DataFrame(data)
        logging.info(f"Retrieved {len(df)} news items")
        return df
    finally:
        session.close()

def get_news_latest_df(publisher=None):
    logging.info(f"Retrieving latest 1000 news items ordered by published date{' for publisher: ' + publisher if publisher else ''}")
    
    session = Session()
    try:
        query = select(News).order_by(News.published_date.asc())
        
        if publisher:
            query = query.filter(News.publisher == publisher)
        
        query = query.limit(1000)
        
        result = session.execute(query)
        news_items = result.scalars().all()
        
        data = [{
            'news_id': item.id,
            'ticker': item.ticker,
            'ticker_url': item.ticker_url,
            'title': item.title,
            'link': item.link,
            'published_date': item.published_date,
            'company': item.company,
            'event': item.event,
            'reason': item.reason,
            'publisher': item.publisher,
            'industry': item.industry,
            'publisher_topic': item.publisher_topic,
            'instrument_id': item.instrument_id,
            'yf_ticker': item.yf_ticker,
            'published_date_gmt': item.published_date_gmt,
            'timezone': item.timezone,
            'publisher_summary': item.publisher_summary,
            'predicted_side': item.predicted_side,
            'predicted_move': item.predicted_move,
            'event': item.event,
            'language': item.language
        } for item in news_items]
        
        df = pd.DataFrame(data)
        logging.info(f"Retrieved {len(df)} latest news items")
        return df
    finally:
        session.close()

def update_news_predictions(df):
    logging.info("Updating news table with predictions")
    
    session = Session()
    try:
        updated_count = 0
        total_items = len(df)
        for index, row in df.iterrows():
            news_item = session.get(News, row['news_id'])
            if news_item:
                if pd.notnull(row['predicted_side']):
                    news_item.predicted_side = row['predicted_side']
                    logging.info(f"Updating predicted_side for news_id {row['news_id']}: {row['predicted_side']}")
                else:
                    logging.warning(f"Null predicted_side for news_id {row['news_id']}")
                if pd.notnull(row['predicted_move']):
                    news_item.predicted_move = row['predicted_move']
                    logging.info(f"Updating predicted_move for news_id {row['news_id']}: {row['predicted_move']}")
                else:
                    logging.warning(f"Null predicted_move for news_id {row['news_id']}")
                updated_count += 1
            else:
                logging.warning(f"No news item found for news_id {row['news_id']}")
            
            if (index + 1) % 100 == 0 or index == total_items - 1:
                session.commit()
                logging.info(f"Updated {index + 1}/{total_items} items")
        
        logging.info(f"Successfully updated {updated_count} news items with predictions")
    except Exception as e:
        logging.error(f"Error updating news predictions: {str(e)}")
        session.rollback()
    finally:
        session.close()

def update_records(df):
    logging.info(f"Updating {len(df)} records in the database")
    
    session = Session()
    try:
        updated_count = 0
        for index, row in df.iterrows():
            stmt = update(News).where(News.id == row['news_id'])
            update_values = {}
            for column in row.index:
                if column != 'news_id':
                    value = row[column]
                    if pd.notnull(value):
                        if isinstance(value, pd.Timestamp):
                            value = value.to_pydatetime()
                        update_values[column] = value
            
            if update_values:
                stmt = stmt.values(**update_values)
                result = session.execute(stmt)
                if result.rowcount > 0:
                    updated_count += 1
            
            if (index + 1) % 100 == 0 or index == len(df) - 1:
                session.commit()
                logging.info(f"Updated {updated_count}/{index + 1} records")
        
        logging.info(f"Successfully updated {updated_count} records")
        return updated_count
    except Exception as e:
        logging.error(f"Error updating records: {str(e)}")
        session.rollback()
        return 0
    finally:
        session.close()

def get_instrument_by_company_name(company_name):
    logging.info(f"Looking up instrument for company: {company_name}")
    
    session = Session()
    try:
        # Use both exact match (case-insensitive) and LIKE match on the issuer field
        try:
            instruments = session.query(Instrument).filter(
                or_(
                    func.lower(Instrument.issuer) == func.lower(company_name),
                    func.lower(Instrument.issuer).like(f"%{company_name.lower()}%")
                )
            ).all()
            
            if instruments:
                # Select the first instrument if there are multiple matches
                instrument = instruments[0]
                logging.info(f"Found instrument for {company_name}: {instrument.ticker}")
                if len(instruments) > 1:
                    logging.warning(f"Multiple instruments found for {company_name}. Using the first match: {instrument.ticker}")
            else:
                instrument = None
                logging.info(f"No instrument found for {company_name}")
            
            return instrument
        except ProgrammingError as e:
            if 'relation "instruments" does not exist' in str(e):
                logging.error("The 'instruments' table does not exist in the database. Please ensure the table is created and populated.")
            else:
                logging.error(f"An error occurred while querying the database: {str(e)}")
            return None
    finally:
        session.close()


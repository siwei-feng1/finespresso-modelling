import logging
from datetime import datetime, time
from sqlalchemy import Column, Integer, String, Float, DateTime, text, select, join, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError
import pandas as pd
from utils.db.news_db_util import News
from sqlalchemy import and_
import streamlit as st
from datetime import timedelta
from utils.db_pool import DatabasePool

logger = logging.getLogger(__name__)

# Get the database pool instance
db_pool = DatabasePool()
Base = declarative_base()

class PriceMove(Base):
    __tablename__ = 'price_moves'

    id = Column(Integer, primary_key=True)
    news_id = Column(Integer, nullable=False)
    ticker = Column(String, nullable=False)
    published_date = Column(DateTime, nullable=False)
    begin_price = Column(Float, nullable=False)
    end_price = Column(Float, nullable=False)
    index_begin_price = Column(Float, nullable=False)
    index_end_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
    market = Column(String, nullable=False)
    price_change = Column(Float, nullable=False)
    price_change_percentage = Column(Float, nullable=False)
    index_price_change = Column(Float, nullable=False)
    index_price_change_percentage = Column(Float, nullable=False)
    daily_alpha = Column(Float, nullable=False)
    actual_side = Column(String(10), nullable=False)
    predicted_side = Column(String(10))
    predicted_move = Column(Float)
    downloaded_at = Column(DateTime, nullable=True)
    price_source = Column(String(20), nullable=False, default='yfinance')
    runid = Column(Integer, nullable=True)  # BIGINT equivalent in SQLAlchemy

    def __init__(self, news_id, ticker, published_date, begin_price, end_price, index_begin_price, index_end_price,
                 volume, market, price_change, price_change_percentage, index_price_change, index_price_change_percentage,
                 daily_alpha, actual_side, predicted_side=None, predicted_move=None, price_source='yfinance', runid=None):
        self.news_id = news_id
        self.ticker = ticker
        self.published_date = published_date
        self.begin_price = begin_price
        self.end_price = end_price
        self.index_begin_price = index_begin_price
        self.index_end_price = index_end_price
        self.volume = volume
        self.market = market
        self.price_change = price_change
        self.price_change_percentage = price_change_percentage
        self.index_price_change = index_price_change
        self.index_price_change_percentage = index_price_change_percentage
        self.daily_alpha = daily_alpha
        self.actual_side = actual_side
        self.predicted_side = predicted_side
        self.predicted_move = predicted_move
        self.price_source = price_source
        self.runid = runid

def store_price_move(price_move: PriceMove) -> bool:
    """Store a price move in the database"""
    session = db_pool.get_session()
    try:
        # If price_move is a pandas Series, convert it to a PriceMove object
        if isinstance(price_move, pd.Series):
            price_move = PriceMove(
                news_id=price_move['news_id'],
                ticker=price_move['yf_ticker'],
                published_date=price_move['published_date'],
                begin_price=price_move['begin_price'],
                end_price=price_move['end_price'],
                index_begin_price=price_move['index_begin_price'],
                index_end_price=price_move['index_end_price'],
                volume=price_move.get('Volume'),
                market=price_move['market'],
                price_change=price_move['price_change'],
                price_change_percentage=price_move['price_change_percentage'],
                index_price_change=price_move['index_price_change'],
                index_price_change_percentage=price_move['index_price_change_percentage'],
                daily_alpha=price_move['daily_alpha'],
                actual_side=price_move['actual_side'],
                price_source='yfinance',  # Default to yfinance for Series objects
                runid=price_move.get('runid')  # Get runid from Series if available
            )

        # Check if a price move already exists for this news_id and price_source
        existing = session.query(PriceMove).filter(
            and_(
                PriceMove.news_id == price_move.news_id,
                PriceMove.price_source == price_move.price_source
            )
        ).first()
        
        if existing:
            # Update existing record
            for key, value in price_move.__dict__.items():
                if key != '_sa_instance_state':  # Skip SQLAlchemy internal attribute
                    setattr(existing, key, value)
            logger.info(f"Updated existing price move for news_id: {price_move.news_id}, ticker: {price_move.ticker}")
        else:
            # Add new record
            session.add(price_move)
            logger.info(f"Added new price move for news_id: {price_move.news_id}, ticker: {price_move.ticker}")
        
        session.commit()
        
        # Verify storage
        stored = session.query(PriceMove).filter(
            and_(
                PriceMove.news_id == price_move.news_id,
                PriceMove.price_source == price_move.price_source
            )
        ).first()
        
        if stored:
            logger.info(f"Verified: Price move for news_id {price_move.news_id} is in the database")
            return True
        else:
            logger.error(f"Verification failed: Price move for news_id {price_move.news_id} not found in the database")
            return False
            
    except Exception as e:
        logger.error(f"Error storing price move: {e}")
        logger.exception("Detailed traceback:")
        session.rollback()
        return False
    finally:
        db_pool.return_session(session)

def get_price_moves(runid=None):
    """
    Get all price moves joined with news data
    
    Args:
        runid: Optional runid to filter by specific run
    """
    session = db_pool.get_session()
    try:
        # Base query joining News and PriceMove
        query = select(
            News.id, News.content, News.title, News.content_en, News.title_en, 
            News.event, News.predicted_side, News.predicted_move, News.publisher,
            News.published_date, News.ticker, News.company, News.reason, News.link,
            News.ticker_url, PriceMove.actual_side, PriceMove.price_change_percentage, 
            PriceMove.daily_alpha, PriceMove.runid
        ).select_from(
            join(News, PriceMove, News.id == PriceMove.news_id)
        )
        
        # Add runid filter if specified
        if runid is not None:
            query = query.where(PriceMove.runid == runid)
        
        query = query.order_by(News.published_date.desc())
        
        # Execute query and get results
        result = session.execute(query)
        rows = result.fetchall()
        
        # Create DataFrame with named columns
        df = pd.DataFrame(rows, columns=[
            'id', 'content', 'title', 'content_en', 'title_en', 'event',
            'predicted_side', 'predicted_move', 'publisher', 'published_date',
            'ticker', 'company', 'reason', 'link', 'ticker_url', 'actual_side',
            'price_change_percentage', 'daily_alpha', 'runid'
        ])
        
        logging.info(f"Retrieved {len(df)} price moves with news data")
        return df
    except Exception as e:
        logging.error(f"Error retrieving price moves: {str(e)}")
        logging.exception("Full traceback:")
        return pd.DataFrame()
    finally:
        db_pool.return_session(session)

def get_news_price_moves():
    """
    Get all price moves joined with news data. Alias for get_price_moves() for backwards compatibility.
    """
    return get_price_moves()

def get_price_moves_by_runid(runid):
    """
    Get price moves for a specific runid
    
    Args:
        runid: The runid to filter by
        
    Returns:
        DataFrame with price moves for the specified runid
    """
    return get_price_moves(runid=runid)

def get_available_runids():
    """
    Get list of available runids in the database
    
    Returns:
        List of runids
    """
    session = db_pool.get_session()
    try:
        query = select(PriceMove.runid).distinct().where(PriceMove.runid.isnot(None)).order_by(PriceMove.runid)
        result = session.execute(query)
        runids = [row[0] for row in result.fetchall()]
        logger.info(f"Found {len(runids)} available runids: {runids}")
        return runids
    except Exception as e:
        logger.error(f"Error retrieving runids: {str(e)}")
        return []
    finally:
        db_pool.return_session(session)

# Create tables
Base.metadata.create_all(db_pool.engine)

def add_market_column():
    with db_pool.engine.connect() as connection:
        connection.execute(text("ALTER TABLE price_moves ADD COLUMN market VARCHAR(255) NOT NULL DEFAULT 'market_open';"))
        connection.commit()

def get_price_moves_date_range(start_date, end_date, publisher=None, runid=None):
    session = db_pool.get_session()
    try:
        # Convert dates to UTC datetime if they aren't already
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.tz_localize('UTC')
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.tz_localize('UTC')
            
        logging.info(f"Querying with date range: {start_date} to {end_date}")
        st.write(f"Querying with date range: {start_date} to {end_date}")
        
        # First check if we have any news in this date range
        news_check = select(func.count(News.id)).where(
            News.published_date.between(start_date, end_date)
        )
        news_count = session.execute(news_check).scalar()
        st.write(f"Found {news_count} news items in date range")
        
        # Print the news check query
        news_check_sql = news_check.compile(
            dialect=session.bind.dialect,
            compile_kwargs={"literal_binds": True}
        )
        st.write("News check query:")
        st.code(str(news_check_sql), language="sql")
        
        # Base query joining News and PriceMove
        query = select(
            News.id, News.content, News.title, News.content_en, News.title_en, 
            News.event, News.predicted_side, News.predicted_move, News.publisher,
            News.published_date, News.ticker, News.company, News.reason, News.link,
            News.ticker_url, PriceMove.actual_side, PriceMove.price_change_percentage, 
            PriceMove.daily_alpha, PriceMove.runid
        ).select_from(
            join(News, PriceMove, News.id == PriceMove.news_id)
        ).where(
            News.published_date.between(start_date, end_date)
        )
        
        # Add publisher filter if specified
        if publisher:
            query = query.where(News.publisher == publisher)
            st.write(f"Filtering for publisher: {publisher}")
        
        # Add runid filter if specified
        if runid is not None:
            query = query.where(PriceMove.runid == runid)
            st.write(f"Filtering for runid: {runid}")
        
        query = query.order_by(News.published_date.desc())
        
        # Print the main query with actual values
        compiled = query.compile(
            dialect=session.bind.dialect,
            compile_kwargs={"literal_binds": True}
        )
        st.write("Main query:")
        st.code(str(compiled), language="sql")
        
        # Execute query and get results
        result = session.execute(query)
        rows = result.fetchall()
        st.write(f"Query returned {len(rows)} rows")
        
        # If no results, check a simple join without date filter
        if len(rows) == 0:
            basic_join = select(func.count()).select_from(
                join(News, PriceMove, News.id == PriceMove.news_id)
            )
            join_count = session.execute(basic_join).scalar()
            st.write(f"Total records in join (without date filter): {join_count}")
            
            # Print the basic join query
            basic_join_sql = basic_join.compile(
                dialect=session.bind.dialect,
                compile_kwargs={"literal_binds": True}
            )
            st.write("Basic join query:")
            st.code(str(basic_join_sql), language="sql")
        
        # Create DataFrame with named columns
        df = pd.DataFrame(rows, columns=[
            'id', 'content', 'title', 'content_en', 'title_en', 'event',
            'predicted_side', 'predicted_move', 'publisher', 'published_date',
            'ticker', 'company', 'reason', 'link', 'ticker_url', 'actual_side',
            'price_change_percentage', 'daily_alpha', 'runid'
        ])
        
        return df
    except Exception as e:
        logging.error(f"Error retrieving news and price moves: {str(e)}")
        logging.exception("Full traceback:")
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()
    finally:
        db_pool.return_session(session)

def add_price_source_column():
    """Add price_source column to price_moves table if it doesn't exist"""
    with db_pool.engine.connect() as connection:
        # Check if column exists
        result = connection.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'price_moves' 
            AND column_name = 'price_source'
        """))
        
        if not result.fetchone():
            # Add column if it doesn't exist
            connection.execute(text("""
                ALTER TABLE price_moves 
                ADD COLUMN price_source VARCHAR(20) NOT NULL DEFAULT 'yfinance'
            """))
            connection.commit()
            logger.info("Added price_source column to price_moves table")
        else:
            logger.info("price_source column already exists in price_moves table")

def add_runid_column():
    """Add runid column to price_moves table if it doesn't exist"""
    with db_pool.engine.connect() as connection:
        # Check if column exists
        result = connection.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'price_moves' 
            AND column_name = 'runid'
        """))
        
        if not result.fetchone():
            # Add column if it doesn't exist
            connection.execute(text("""
                ALTER TABLE price_moves 
                ADD COLUMN runid BIGINT NULL
            """))
            
            # Add index for better query performance
            connection.execute(text("""
                CREATE INDEX idx_price_moves_runid ON price_moves(runid)
            """))
            
            connection.commit()
            logger.info("Added runid column and index to price_moves table")
        else:
            logger.info("runid column already exists in price_moves table")

def remove_volume_not_null_constraint():
    """Remove NOT NULL constraint from volume column in price_moves table"""
    with db_pool.engine.connect() as connection:
        try:
            # Remove NOT NULL constraint from volume column
            connection.execute(text("""
                ALTER TABLE price_moves 
                ALTER COLUMN volume DROP NOT NULL
            """))
            
            connection.commit()
            logger.info("Successfully removed NOT NULL constraint from volume column in price_moves table")
            
        except Exception as e:
            logger.error(f"Error removing volume NOT NULL constraint: {e}")
            connection.rollback()
            raise

def get_raw_price_moves(runid=None):
    """
    Get raw price moves data from the price_moves table without joining with news data
    
    Args:
        runid: Optional runid to filter by specific run
        
    Returns:
        DataFrame with raw price moves data
    """
    session = db_pool.get_session()
    try:
        # Base query for raw price moves
        query = select(PriceMove).order_by(PriceMove.published_date.desc())
        
        # Add runid filter if specified
        if runid is not None:
            query = query.where(PriceMove.runid == runid)
        
        # Execute query and get results
        result = session.execute(query)
        rows = result.fetchall()
        
        # Create DataFrame with all PriceMove columns
        df = pd.DataFrame([row[0].__dict__ for row in rows])
        
        # Remove SQLAlchemy internal attribute
        if '_sa_instance_state' in df.columns:
            df = df.drop('_sa_instance_state', axis=1)
        
        logging.info(f"Retrieved {len(df)} raw price moves")
        return df
    except Exception as e:
        logging.error(f"Error retrieving raw price moves: {str(e)}")
        logging.exception("Full traceback:")
        return pd.DataFrame()
    finally:
        db_pool.return_session(session)

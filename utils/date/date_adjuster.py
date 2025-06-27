import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from utils.db.news_db_util import News, engine
import pytz
from datetime import datetime, timedelta
import logging
from holidays import US as US_holidays

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_previous_trading_day(date):
    us_holidays = US_holidays()
    current_date = date
    while True:
        current_date -= timedelta(days=1)
        if current_date.weekday() < 5 and current_date not in us_holidays:
            return current_date

def get_next_trading_day(date):
    us_holidays = US_holidays()
    current_date = date
    while True:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5 and current_date not in us_holidays:
            return current_date

def adjust_published_date(publisher: str, target_timezone: str):
    """
    Adjust the published_date for all news items of a given publisher to a target timezone.

    Args:
    publisher (str): The publisher of the news items to adjust.
    target_timezone (str): The target timezone (e.g., 'US/Eastern', 'US/Pacific').

    Returns:
    int: The number of records updated.
    """
    logging.info(f"Adjusting published dates for {publisher} to {target_timezone}")

    session = Session(engine)
    try:
        # Fetch all news items for the given publisher
        query = select(News).where(News.publisher == publisher)
        result = session.execute(query)
        news_items = result.scalars().all()

        target_tz = pytz.timezone(target_timezone)
        updated_count = 0

        for item in news_items:
            # Handle both naive and aware datetime objects
            if item.published_date_gmt.tzinfo is None:
                published_date_gmt = pytz.utc.localize(item.published_date_gmt)
            else:
                published_date_gmt = item.published_date_gmt.astimezone(pytz.utc)

            # Calculate the offset
            offset = target_tz.utcoffset(published_date_gmt)
            
            # Apply the offset directly
            adjusted_date = published_date_gmt + offset

            # Update the database
            stmt = update(News).where(News.id == item.id).values(
                published_date=adjusted_date.replace(tzinfo=None),  # Store as naive datetime
                timezone=target_timezone
            )
            result = session.execute(stmt)
            updated_count += result.rowcount

        session.commit()
        logging.info(f"Successfully updated {updated_count} news items")
        return updated_count

    except Exception as e:
        logging.error(f"An error occurred while adjusting dates: {e}")
        session.rollback()
        return 0
    finally:
        session.close()

if __name__ == "__main__":
    # Example usage
    publisher = "globenewswire_biotech"
    target_timezone = "US/Eastern"  # Use IANA timezone names
    updated_count = adjust_published_date(publisher, target_timezone)
    print(f"Updated {updated_count} records for {publisher} to {target_timezone}")

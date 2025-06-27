import pandas as pd
import pandas_market_calendars as mcal
from datetime import timedelta
import pytz

def adjust_dates_for_weekends_and_holidays(today_date, exchange='NYSE'):
    # Use the specified exchange calendar
    calendar = mcal.get_calendar(exchange)
    
    # Convert today_date to datetime if it's not already
    if not isinstance(today_date, pd.Timestamp):
        today_date = pd.Timestamp(today_date)
    
    # Get the previous and next valid business days
    prev_business_day = calendar.date_range(end_date=today_date, periods=1)[0].date()
    next_business_day = calendar.date_range(start_date=today_date + pd.Timedelta(days=1), periods=1)[0].date()
    
    return prev_business_day, next_business_day

def get_business_days_between(start_date, end_date, exchange='NYSE'):
    calendar = mcal.get_calendar(exchange)
    business_days = calendar.valid_days(start_date=start_date, end_date=end_date)
    return business_days.tolist()

def is_business_day(date, exchange='NYSE'):
    calendar = mcal.get_calendar(exchange)
    return date.date() in calendar.valid_days(start_date=date.date(), end_date=date.date())

def adjust_date_to_est(date):
    """
    Adjusts the given date to EST, accounting for daylight saving time.
    """
    if date.tzinfo is None:
        date = pytz.utc.localize(date)
    else:
        date = date.astimezone(pytz.utc)

    est_tz = pytz.timezone('US/Eastern')
    est_date = date.astimezone(est_tz)
    
    # Calculate the difference between UTC and EST
    utc_offset = est_date.utcoffset().total_seconds() / 3600
    
    # Adjust the date
    adjusted_date = date - timedelta(hours=abs(utc_offset))
    
    return adjusted_date.replace(tzinfo=None)  # Return as naive datetime

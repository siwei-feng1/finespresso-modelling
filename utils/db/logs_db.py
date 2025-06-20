from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import logging
from utils.db_pool import DatabasePool

# Get the database pool instance
db_pool = DatabasePool()
Base = declarative_base()

class Logs(Base):
    __tablename__ = 'fe_logs'

    id = Column(Integer, primary_key=True)
    message = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    status = Column(String, nullable=False)

def create_logs_table():
    Base.metadata.create_all(db_pool.engine)

def save_log(message, status):
    session = db_pool.get_session()
    try:
        new_log = Logs(message=message, status=status)
        session.add(new_log)
        session.commit()
        logging.info(f"Successfully added log: {message}")
        return True
    except Exception as e:
        logging.error(f"An error occurred while saving log: {e}")
        session.rollback()
        return False
    finally:
        db_pool.return_session(session)

def get_logs(limit=100):
    session = db_pool.get_session()
    try:
        logs = session.query(Logs).order_by(Logs.timestamp.desc()).limit(limit).all()
        return logs
    except Exception as e:
        logging.error(f"An error occurred while retrieving logs: {e}")
        return []
    finally:
        db_pool.return_session(session)

# Create the logs table when this module is imported
create_logs_table()

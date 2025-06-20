from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
import os
import logging

class DatabasePool:
    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the database connection pool"""
        load_dotenv()
        
        DATABASE_URL = os.getenv('DATABASE_URL')
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set")

        # Configure the engine with pooling
        self._engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,  # Maximum number of connections in the pool
            max_overflow=10,  # Maximum number of connections that can be created beyond pool_size
            pool_timeout=30,  # Timeout for getting a connection from the pool
            pool_recycle=1800,  # Recycle connections after 30 minutes
            pool_pre_ping=True  # Enable connection health checks
        )

        # Create a thread-safe session factory
        self._session_factory = scoped_session(sessionmaker(bind=self._engine))

    @property
    def engine(self):
        """Get the SQLAlchemy engine"""
        return self._engine

    @property
    def session_factory(self):
        """Get the session factory"""
        return self._session_factory

    def get_session(self):
        """Get a new session from the pool"""
        return self._session_factory()

    def return_session(self, session):
        """Return a session to the pool"""
        if session:
            session.close()
            self._session_factory.remove()

    def dispose(self):
        """Dispose of the engine and all pooled connections"""
        if self._engine:
            self._engine.dispose() 
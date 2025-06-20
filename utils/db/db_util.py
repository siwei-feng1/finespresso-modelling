from utils.db_pool import DatabasePool

# Get the database pool instance
db_pool = DatabasePool()

# Export the engine for other modules that need it
engine = db_pool.engine 
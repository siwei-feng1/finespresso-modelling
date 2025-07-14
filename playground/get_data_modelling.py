import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the database URL from .env
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL not found in .env file")

# Create a database connection
engine = create_engine(database_url)

# Define the SQL query to merge tables and select columns
query = """
SELECT 
    n.id, n.content, n.title, n.content_en, n.title_en, 
    n.event, n.predicted_side, n.predicted_move, n.publisher, 
    n.published_date, n.ticker, n.company, n.reason, n.link, 
    n.ticker_url, p.actual_side, p.price_change_percentage, p.daily_alpha
FROM news n
INNER JOIN price_moves p
    ON n.id = p.id
WHERE p.actual_side IS NOT NULL 
    AND p.daily_alpha IS NOT NULL
"""

# Execute the query and load into a DataFrame
df = pd.read_sql(query, engine)

# Save the result to a CSV file
df.to_csv("data/modeling_data.csv", index=False)

print("Data has been merged, filtered, and saved to 'modeling_data.csv'.")
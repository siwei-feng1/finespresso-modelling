from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

table_name = "news"

query = text(f"SELECT * FROM {table_name} limit 2")

with engine.connect() as conn:
    result = conn.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
print(df)
   



from sqlalchemy import create_engine, inspect
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def infer_schema():
    # Read database URL from environment variable
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # Create SQLAlchemy engine
    engine = create_engine(database_url)
    inspector = inspect(engine)

    # Dictionary to store schema information
    schema_info = {}

    # Get all table names
    tables = inspector.get_table_names()

    for table_name in tables:
        # Get column information for each table
        columns = inspector.get_columns(table_name)
        
        # Store column details
        column_info = []
        for column in columns:
            column_info.append({
                'name': column['name'],
                'type': str(column['type']),
                'nullable': column['nullable'],
                'default': str(column['default']) if column['default'] else None,
                'primary_key': column.get('primary_key', False)
            })

        # Get foreign key information
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        # Get primary key information
        primary_keys = inspector.get_pk_constraint(table_name)
        
        # Get index information
        indexes = inspector.get_indexes(table_name)

        # Store all table information
        schema_info[table_name] = {
            'columns': column_info,
            'foreign_keys': foreign_keys,
            'primary_keys': primary_keys,
            'indexes': indexes
        }

    return schema_info

def save_schema():
    # Create data directory if it doesn't exist
    os.makedirs('sql', exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'sql/schema_{timestamp}.json'
    
    # Infer schema and save to file
    schema = infer_schema()
    
    with open(filename, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"Schema saved to {filename}")

if __name__ == "__main__":
    save_schema()

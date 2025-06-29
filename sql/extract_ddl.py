from sqlalchemy import create_engine, MetaData, inspect
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def ensure_sql_directory():
    """Ensure the sql directory exists"""
    sql_dir = os.path.join(os.path.dirname(__file__), "..", "sql")
    if not os.path.exists(sql_dir):
        os.makedirs(sql_dir)
    return sql_dir

def should_skip_table(table_name):
    """Check if table should be skipped"""
    skip_prefixes = ('directus_', 'cms_')
    return any(table_name.startswith(prefix) for prefix in skip_prefixes)

def get_postgresql_type(column_type):
    """Convert SQLAlchemy type to PostgreSQL type string"""
    type_str = str(column_type)
    
    # Handle common type mappings
    if 'VARCHAR' in type_str:
        if '(' in type_str:
            return type_str
        return 'VARCHAR'
    elif 'INTEGER' in type_str:
        return 'INTEGER'
    elif 'BIGINT' in type_str:
        return 'BIGINT'
    elif 'DOUBLE PRECISION' in type_str:
        return 'DOUBLE PRECISION'
    elif 'TIMESTAMP' in type_str:
        return 'TIMESTAMP'
    elif 'UUID' in type_str:
        return 'UUID'
    elif 'TEXT' in type_str:
        return 'TEXT'
    elif 'BOOLEAN' in type_str:
        return 'BOOLEAN'
    elif 'JSON' in type_str:
        return 'JSON'
    else:
        return type_str

def extract_ddl():
    """Extract DDL schema from the source database"""
    # Read database URL from environment variable
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # Create engine
    engine = create_engine(database_url)
    
    # Get the inspector
    inspector = inspect(engine)
    
    # Prepare the output file
    sql_dir = ensure_sql_directory()
    output_file = os.path.join(sql_dir, "finespresso-db.ddl")
    
    with open(output_file, 'w') as f:
        # Write header comment
        f.write("-- Database schema extracted from source database\n")
        f.write("-- Generated automatically by extract_ddl.py\n")
        f.write("-- Database: Finespresso\n\n")
        
        # Get all tables
        tables = inspector.get_table_names()
        
        for table_name in tables:
            if should_skip_table(table_name):
                continue
                
            # Get column information
            columns = inspector.get_columns(table_name)
            
            # Get primary key information
            primary_keys = inspector.get_pk_constraint(table_name)
            pk_columns = primary_keys.get('constrained_columns', [])
            
            # Write table creation SQL
            f.write(f"-- Table: {table_name}\n")
            f.write(f"CREATE TABLE {table_name} (\n")
            
            column_definitions = []
            for column in columns:
                col_name = column['name']
                col_type = get_postgresql_type(column['type'])
                nullable = "" if column['nullable'] else " NOT NULL"
                default = ""
                
                if column['default'] is not None:
                    default_value = str(column['default'])
                    # Handle sequence defaults
                    if 'nextval' in default_value:
                        default = f" DEFAULT {default_value}"
                    elif 'gen_random_uuid' in default_value:
                        default = " DEFAULT gen_random_uuid()"
                    elif 'CURRENT_TIMESTAMP' in default_value:
                        default = " DEFAULT CURRENT_TIMESTAMP"
                    else:
                        default = f" DEFAULT {default_value}"
                
                column_definitions.append(f"    {col_name} {col_type}{nullable}{default}")
            
            # Add primary key constraint if it exists
            if pk_columns:
                pk_constraint = f"    PRIMARY KEY ({', '.join(pk_columns)})"
                column_definitions.append(pk_constraint)
            
            f.write(",\n".join(column_definitions))
            f.write("\n);\n\n")
            
            # Get and write indexes
            indexes = inspector.get_indexes(table_name)
            for index in indexes:
                if not index.get('unique', False):  # Skip primary key indexes
                    columns = ', '.join(index['column_names'])
                    f.write(f"CREATE INDEX IF NOT EXISTS {index['name']} ON {table_name} ({columns});\n")
            
            # Get and write foreign keys
            foreign_keys = inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                if not should_skip_table(fk['referred_table']):
                    constrained_cols = ', '.join(fk['constrained_columns'])
                    referred_cols = ', '.join(fk['referred_columns'])
                    f.write(f"-- Foreign Key: {fk['name']}\n")
                    f.write(f"-- {table_name}({constrained_cols}) REFERENCES {fk['referred_table']}({referred_cols})\n")
            
            f.write("\n")
    
    print(f"DDL schema has been extracted to {output_file}")

if __name__ == "__main__":
    extract_ddl() 
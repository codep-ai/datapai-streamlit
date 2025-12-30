import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine

# Load environment variables from file
from dotenv import load_dotenv
load_dotenv('.env')

# SQLite3
SQLITE3_DB_PATH = os.getenv("SQLITE3_DB_PATH")

# Redshift
REDSHIFT_DBNAME = os.getenv("REDSHIFT_DBNAME")
REDSHIFT_SCHEMA = os.getenv("REDSHIFT_SCHEMA")
REDSHIFT_USER = os.getenv("REDSHIFT_USER")
REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")
REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
REDSHIFT_PORT = os.getenv("REDSHIFT_PORT")

# Snowflake
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")

def clean_data(data):
    # Clean data here, for example, removing single quotes to avoid SQL errors
    cleaned_data = []
    for row in data:
        cleaned_row = [str(val).replace("'", "") for val in row]
        cleaned_data.append(cleaned_row)
    return cleaned_data

def copy_data_to_redshift(tables):
    # Create engine for Redshift
    redshift_engine = create_engine(f'postgresql://{REDSHIFT_USER}:{REDSHIFT_PASSWORD}@{REDSHIFT_HOST}:{REDSHIFT_PORT}/{REDSHIFT_DBNAME}')
    
    # Connect to SQLite3
    sqlite_conn = sqlite3.connect(SQLITE3_DB_PATH)
    
    # Copy tables and data to Redshift
    for table_name in tables:
        # Read data from SQLite3
        cursor = sqlite_conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()
        
        # Clean the data
        cleaned_rows = clean_data(rows)
        
        # Insert cleaned data into Redshift
        with redshift_engine.connect() as conn:
            for row in cleaned_rows:
                values = ', '.join([f"'{value}'" for value in row])
                conn.execute(f"INSERT INTO {REDSHIFT_SCHEMA}.{table_name} VALUES ({values});")
    
    # Close SQLite3 connection
    sqlite_conn.close()
    print("Data copied to Redshift.")


if __name__ == "__main__":
    # Connect to SQLite3
    sqlite_conn = sqlite3.connect(SQLITE3_DB_PATH)
    
    # Get list of tables from SQLite3 database
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", sqlite_conn)
    table_names = tables['name'].tolist()
    
    # Copy data to Redshift
    copy_data_to_redshift(table_names)
    
    
    # Close SQLite3 connection
    sqlite_conn.close()


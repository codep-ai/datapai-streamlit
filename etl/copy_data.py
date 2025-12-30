import os
import sqlite3
import pandas as pd
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

# Connect to SQLite3
sqlite_conn = sqlite3.connect(SQLITE3_DB_PATH)

# Get list of tables from SQLite3 database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", sqlite_conn)

# Create engine for Redshift
redshift_engine = create_engine(f'postgresql://{REDSHIFT_USER}:{REDSHIFT_PASSWORD}@{REDSHIFT_HOST}:{REDSHIFT_PORT}/{REDSHIFT_DBNAME}')

# Create engine for Snowflake
snowflake_engine = create_engine(f'snowflake://{SNOWFLAKE_USER}:{SNOWFLAKE_PASSWORD}@{SNOWFLAKE_ACCOUNT}/{SNOWFLAKE_DATABASE}/{SNOWFLAKE_SCHEMA}?warehouse={SNOWFLAKE_WAREHOUSE}&role={SNOWFLAKE_ROLE}')

# Copy tables and data to Redshift and Snowflake
for _, table_row in tables.iterrows():
    table_name = table_row['name']
    
    # Read table structure and data into DataFrame
    table_structure = pd.read_sql_query(f"PRAGMA table_info('{table_name}');", sqlite_conn)
    table_data = pd.read_sql_query(f"SELECT * FROM {table_name};", sqlite_conn)
    # Write table structure and data to Redshift
    table_structure.to_sql(table_name, redshift_engine, schema=REDSHIFT_SCHEMA, if_exists='replace', index=False, dtype={col: 'VARCHAR' for col in table_structure['name']})
    table_data.to_sql(table_name, redshift_engine, schema=REDSHIFT_SCHEMA, if_exists='append', index=False)

    # Write table structure and data to Snowflake
    table_structure.to_sql(table_name, snowflake_engine, schema=SNOWFLAKE_SCHEMA, if_exists='replace', index=False, dtype={col: 'VARCHAR' for col in table_structure['name']})
    table_data.to_sql(table_name, snowflake_engine, schema=SNOWFLAKE_SCHEMA, if_exists='append', index=False)

    

# Close connections
sqlite_conn.close()

print("Tables and data copied successfully!")


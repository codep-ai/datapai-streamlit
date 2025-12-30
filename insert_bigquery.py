import pandas as pd
import os
import sqlite3
from google.cloud import bigquery
import toml

# Load environment variables from file
from dotenv import load_dotenv

# Load TOML file for environment variables
def load_toml_config():
    toml_file = os.path.join(os.path.expanduser("."), ".streamlit", "secrets.toml")
    with open(toml_file, "r") as f:
        config = toml.load(f)
    return config

# Initialize BigQuery client
def initialize_bigquery_client(config):
    return bigquery.Client(project=config['bigquery']['project_id'])

# Clean data function
def clean_data(data):
    cleaned_data = []
    for row in data:
        cleaned_row = [str(val).replace("'", "") for val in row]
        cleaned_data.append(cleaned_row)
    return cleaned_data

# Function to load all tables from SQLite to BigQuery
def load_all_tables_to_bigquery(sqlite_file, client):
    # Connect to SQLite database
    conn = sqlite3.connect(sqlite_file)
    
    # Get list of tables in the SQLite database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    
    for table_name in table_names:
        # Read SQLite table into Pandas DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Define BigQuery dataset and table references
        dataset_id = client.project + "." + config['bigquery']['dataset_id']
        table_ref = client.dataset(config['bigquery']['dataset_id']).table(table_name)

        # Convert DataFrame to BigQuery format and upload
        job_config = bigquery.LoadJobConfig(schema=df.dtypes.to_dict())
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Waits for the job to complete.

        print(f"Loaded {len(df)} rows into {table_name} in BigQuery dataset {config['bigquery']['dataset_id']}")

    conn.close()

# Main function
if __name__ == "__main__":
    # Load TOML config file
    config = load_toml_config()
    
    # Initialize BigQuery client
    client = initialize_bigquery_client(config)
    
    # SQLite3 database path
    #sqlite_file = os.path.join(os.path.expanduser("~"), ".streamlit", config['sqlite']['db_path'])
    sqlite_file = "/home/ec2-user/git/vanna-streamlit//home/ec2-user/git/vanna-streamlit"
    
    # Load all tables to BigQuery
    load_all_tables_to_bigquery(sqlite_file, client)


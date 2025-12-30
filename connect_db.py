# connect_db.py
import streamlit as st
import snowflake.connector
import psycopg2
import sqlite3
import duckdb
from databricks import sql
from google.oauth2 import service_account
from google.cloud import bigquery

# Function to connect to Google BigQuery
def connect_to_bigquery():
    credentials_dict = {
        "type": st.secrets["gcp_service_account"]["type"],
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"],
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"]["client_id"],
        "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
        "token_uri": st.secrets["gcp_service_account"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
    }

    # Create credentials object
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)

    # Initialize BigQuery client
    conn = bigquery.Client(credentials=credentials, project=credentials.project_id)

    st.write("Connected to Google BigQuery")
    return conn



# Function to connect to Snowflake
def connect_to_snowflake():
    conn = snowflake.connector.connect(
        user=st.secrets["SNOWFLAKE_USER"],
        password=st.secrets["SNOWFLAKE_PASSWORD"],
        account=st.secrets["SNOWFLAKE_ACCOUNT"],
        warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
        database=st.secrets["SNOWFLAKE_DATABASE"],
        role=st.secrets["SNOWFLAKE_ROLE"],
        schema=st.secrets["SNOWFLAKE_SCHEMA"]
    )
    st.write("Connected to Snowflake")
    return conn

# Function to connect to Redshift
def connect_to_redshift():
    conn = psycopg2.connect(
        dbname=st.secrets["REDSHIFT_DBNAME"],
        user=st.secrets["REDSHIFT_USER"],
        password=st.secrets["REDSHIFT_PASSWORD"],
        host=st.secrets["REDSHIFT_HOST"],
        port=st.secrets["REDSHIFT_PORT"]
    )

    st.write("Connected to Redshift")
    return conn


def connect_to_databricks():
    conn = sql.connect(
        server_hostname=st.secrets["DATABRICKS_HOSTNAME"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_ACCESS_TOKEN"]
    )

    st.write("Connected to Databricks")
    return conn


# Function to connect to SQLite3
def connect_to_sqlite():
    conn = sqlite3.connect(st.secrets["SQLITE3_DB_PATH"])
    st.write("Connected to SQLite3")
    return conn

# Function to connect to DuckDB
def connect_to_duckdb():
    conn = duckdb.connect(database=st.secrets["DUCKDB_DB_PATH"])
    st.write("Connected to DuckDB")
    return conn

# Common function to connect to a database based on db_type
def connect_to_db(db_type):
    #db_type=st.session_state["selected_db"]
    if db_type == "Snowflake":
    #    st.write("Connected to Snowflake")
        return connect_to_snowflake()
    elif db_type == "Redshift":
    #    st.write("Connected to Redshift")
        return connect_to_redshift()
    elif db_type == "SQLite3":
    #    st.write("Connected to SQLlite3")
        return connect_to_sqlite()
    elif db_type == "DuckDB":
    #    st.write("Connected to DuckDB")
        return connect_to_duckdb()
    elif db_type == "Bigquery":
    #    st.write("Connected to Bigquery")
        return connect_to_bigquery()
    else:
    #    st.write("Unsupported database type")
        return None


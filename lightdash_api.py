import requests

# Lightdash API URL
LIGHTDASH_API_URL = "https://platform.datap.ai/bi/api/v1/graphql"
LIGHTDASH_API_TOKEN = "c545395c7abe5d532a23edc67a7f1f3d"
#  c545395c7abe5d532a23edc67a7f1f3d  is the cli key
#   lightdash login http://localhost:8080 --token c545395c7abe5d532a23edc67a7f1f3d

# Define which databases Lightdash supports
SUPPORTED_LIGHTDASH_DBS = ["Snowflake", "Redshift", "Bigquery"]

def execute_sql_in_lightdash(sql_query, database_type="Snowflake"):
    """
    Executes a SQL query in Lightdash only if the database type is supported.

    :param sql_query: The SQL query to run in Lightdash
    :param database_type: The selected database type (default: Snowflake)
    :return: API response from Lightdash
    """
    if database_type not in SUPPORTED_LIGHTDASH_DBS:
        return {"error": f"Database '{database_type}' is not supported in Lightdash"}

    headers = {
        "Authorization": f"Bearer {LIGHTDASH_API_TOKEN}",
        "Content-Type": "application/json"
    }

    graphql_query = {
        "query": """
        mutation RunSql($sql: String!) {
            runSql(sql: $sql) {
                result
            }
        }
        """,
        "variables": {"sql": sql_query}
    }

    response = requests.post(LIGHTDASH_API_URL, json=graphql_query, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

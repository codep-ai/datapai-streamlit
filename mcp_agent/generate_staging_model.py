# mcp_agent/generate_staging_model.py

import os
from mcp.utils import snake_case
from mcp.constants import STAGING_FOLDER
from dbt_helpers.columns import get_table_columns_from_dbt
from genai.rag import retrieve_rag_context
from openai import ChatOpenAI


def generate_staging_model_files(table_name: str, schema: str):
    columns = get_table_columns_from_dbt(schema, table_name)

    snake_columns = [(col[0], snake_case(col[0])) for col in columns]
    select_stmt = ",\n    ".join(
        [f"{src} as {alias}" if src != alias else src for src, alias in snake_columns]
    )

    sql_path = os.path.join(STAGING_FOLDER, f"stg_{table_name}.sql")
    with open(sql_path, "w") as f:
        f.write(f"""{{% materialization 'view', adapter='default' %}}
select
    {select_stmt}
from {{% source('{schema}', '{table_name}') %}}
{{% endmaterialization %}}""")
    print(f"✅ Generated staging SQL: {sql_path}")

    context = retrieve_rag_context(f"business rules for table {table_name}", top_k=5)
    llm = ChatOpenAI(temperature=0)
    col_names = [snake_case(col[0]) for col in columns]
    col_prompt = "\n".join([f"- {name}" for name in col_names])
    prompt = f"""
Use the following context to add descriptions and dbt tests:
    
{context}

Columns:
{col_prompt} 
        
Respond with a YAML block describing a dbt model for stg_{table_name}.
Include description, columns, tests, and meta.source_doc if relevant.
"""             
    response = llm([{"role": "user", "content": prompt}]).content.strip()
                        
    yml_path = os.path.join(STAGING_FOLDER, f"stg_{table_name}.yml")
    with open(yml_path, "w") as f:
        f.write(response)
    print(f"✅ Generated staging YAML: {yml_path}")


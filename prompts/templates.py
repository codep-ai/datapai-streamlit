# prompts/

# This file contains example prompts or templates used for AI agents (e.g., dbt, RAG, ingestion)

RAG_CONTEXT_PROMPT = """
Use the following business rules and metadata to answer questions or generate dbt model documentation.

{context}

Question: {question}
"""

DBT_MODEL_YAML_PROMPT_TEMPLATE = """
Use the following context to add descriptions and dbt tests:

{context}

Columns:
{columns}
        
Respond with a YAML block describing a dbt model for {model_name}.
Include description, columns, tests, and meta.source_doc if relevant.
"""

DBT_DOCS_GENERATION_PROMPT = """
You are an expert data engineer.

Given the column list below, write a dbt YAML documentation block that includes:
- A clear description of the model
- Descriptions for each column
- Recommended dbt tests like `not_null`, `unique`, or `accepted_values`

Columns:
{columns}
"""


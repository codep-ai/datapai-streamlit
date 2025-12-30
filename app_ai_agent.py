# app_ai_agent.py
from __future__ import annotations
import io, os, json
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from connect_db import connect_to_db

def _run_sql_via_connector(db_type: str, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        conn = connect_to_db(db_type)
        if conn is None:
            return None, f"Unsupported database type: {db_type}"
        if db_type in ("Redshift", "SQLite3"):
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return df, None
        if db_type == "DuckDB":
            df = conn.execute(sql).df()
            conn.close()
            return df, None
        if db_type == "Bigquery":
            df = conn.query(sql).result().to_dataframe()
            return df, None
        if db_type == "Snowflake":
            cur = conn.cursor()
            try:
                cur.execute(sql)
                df = pd.DataFrame(cur.fetchall(), columns=[col[0] for col in cur.description])
            finally:
                cur.close()
                conn.close()
            return df, None
        return None, f"Unsupported DB type: {db_type}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ------------------------------
# Exported entry point
# ------------------------------

def run_ai_agents():
    st.title("🤖 Data P AI Agent Hub")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Run SQL",
        "dbt Agent",
        "Knowledge Ingest Agent",
        "Ingest File",
        "DLT Ingest",
        "Airbyte Sync"
    ])

    with tab1:
        st.header("Run SQL")
        st.caption("Paste SQL and choose destination DB.")
        sql = st.text_area("SQL", height=150)
        db = st.selectbox("Database", ["Snowflake", "Redshift", "Bigquery", "SQLite3", "DuckDB"])
        if st.button("Execute"):
            df, err = _run_sql_via_connector(db, sql)
            if err:
                st.error(err)
            elif df is not None:
                st.dataframe(df)
            else:
                st.info("No rows returned")

    with tab2:
        st.header("dbt Agent")
        schema = st.text_input("Schema Name", "src_file")
        tables = st.text_input("Table Names (comma-separated, optional)", "")
        table_list = [t.strip() for t in tables.split(",") if t.strip()] if tables else None
        if st.button("Run dbt Agent"):
            from agents.dbt_agent import run_dbt_agent
            results = run_dbt_agent(schema=schema, table_names=table_list)
            st.success(f"Processed: {', '.join(results)}")

    with tab3:
        st.header("Knowledge Ingest Agent")
        uploads = st.file_uploader("Upload documents/images", accept_multiple_files=True, type=["pdf", "docx", "txt", "md", "markdown", "png", "jpg", "jpeg"])
        if uploads and st.button("Ingest to Vector Store"):
            temp_paths = []
            for f in uploads:
                path = f"/tmp/{f.name}"
                with open(path, "wb") as out:
                    out.write(f.read())
                temp_paths.append(path)
            from agents.knowledge_ingest_agent import run_knowledge_ingest_agent
            run_knowledge_ingest_agent(temp_paths)
            st.success("Knowledge ingested to LanceDB")
        
        from agents.knowledge_query_agent import answer_with_ollama

        st.subheader("Ask questions about ingested knowledge")

        question = st.text_area("Question", height=80, key="knowledge_question")
        collections = st.multiselect(
            "Collections to search",
            options=["documents", "pdfs", "images"],
            default=["pdfs", "documents"],
            key="knowledge_collections",
        )
        k = st.slider("Top K results", 1, 20, 5, key="knowledge_k")

        if st.button("Ask", key="knowledge_ask_button"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching LanceDB + querying Ollama..."):
                    try:
                        answer = answer_with_ollama(
                            question=question,
                            collections=collections,
                            k=k,
                        )
                        st.markdown("### Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"RAG query failed: {e}")



    with tab4:
        st.header("Ingest File")
        dest_db = st.selectbox("Destination", ["Snowflake", "Redshift", "Bigquery", "Sqlite3", "Duckdb"], key="file_dest")
        file_format = st.selectbox("File format", ["csv", "parquet", "iceberg"], index=0)
        table_name = st.text_input("Destination table name", value="ingested_data")

        up = st.file_uploader(f"Upload {file_format.upper()}", type=[file_format])
        source_path = st.text_input("...or S3/local path", value="")

        source_obj = io.BytesIO(up.read()) if up else None
        source = source_obj if source_obj else source_path.strip()

        if st.button("Ingest file"):
            if not source:
                st.warning("Please provide a file upload or a valid path.")
            else:
                try:
                    from agents.file_ingest_agent import ingest_file

                    # Add a fake filename for in-memory files so extension detection works
                    if isinstance(source, io.BytesIO):
                        source.name = f"upload.{file_format}"

                    with st.spinner("Processing file..."):
                        ingest_file(source, dest_db, table_name)

                    st.success(f"✅ File ingested to {dest_db}: table `{table_name}`")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    with tab5:
        st.header("DLT ingestion")
        source_uri = st.text_input("Source URI", value="s3://my-bucket/path/file.csv")
        destination = st.selectbox("Destination", ["Bigquery", "Snowflake"])
        ds_or_schema = st.text_input("Dataset (BigQuery) or Schema (Snowflake)", value="datapai")
        table_name = st.text_input("Table name", value="dlt_ingested")

        if st.button("Run DLT load"):
            from agents.dlt_ingest_agent import ingest_with_dlt
            ok, msg = ingest_with_dlt(
                source_uri=source_uri,
                destination=destination,
                dataset_or_schema=ds_or_schema,
                table_name=table_name,
                pipeline_name="dlt_csv_pipeline",
            )
            st.success(msg) if ok else st.error(msg)

    with tab6:
        st.header("Airbyte sync")
        connection_id = st.text_input("Airbyte Connection ID (UUID)", value=os.environ.get("AIRBYTE_CONNECTION_ID", ""))
        poll = st.checkbox("Poll until completion", value=True)

        if st.button("Trigger sync"):
            if not connection_id:
                st.warning("Provide a connection id.")
            else:
                from agents.airbyte_ingest_agent import ingest_with_airbyte
                ok, msg = ingest_with_airbyte(connection_id=connection_id, poll=poll)
                st.success(msg) if ok else st.error(msg)


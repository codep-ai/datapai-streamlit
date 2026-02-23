import streamlit as st

#from vanna.remote import VannaDefault
from remote import VannaDefault



@st.cache_resource(ttl=3600)
def setup_vanna():
    APIKEY=st.secrets.get("VANNA_API_KEY")
    selected_db=st.session_state["selected_db"]
    vn = VannaDefault(api_key=APIKEY, model='chinook')
    #vn.connect_db_2()
    vn.connect_to_sqlite("https://vanna.ai/Chinook.sqlite")
    return vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner=False)
def generate_sql_cached(question: str, selected_db: str):
    vn = setup_vanna()

    assumption = ""
    schema_context = ""

    if selected_db == "Snowflake":
        database = st.secrets["SNOWFLAKE_DATABASE"]
        schema = st.secrets["SNOWFLAKE_SCHEMA"]
        assumption = (
            f" Use Snowflake SQL. Assume database is {database} and schema is {schema}. "
            f'Note: "_" may be used to split fields in names.'
        )

    elif selected_db == "dbt":
        database = st.secrets["SNOWFLAKE_DATABASE"]
        schema = st.secrets["SNOWFLAKE_SCHEMA"]
        assumption = (
            f" Generate dbt SQL (dbt model style) for Snowflake. "
            f"Assume database is {database} and schema is {schema}."
        )

    elif selected_db == "Redshift":
        database = st.secrets["REDSHIFT_DBNAME"]
        schema = st.secrets["REDSHIFT_SCHEMA"]
        assumption = (
            f" Use Amazon Redshift SQL. "
            f"Assume database is {database} and schema is {schema}."
        )

    elif selected_db in ("Athena", "Athena (Iceberg)"):
        # ---- Dialect + namespace assumptions (ALWAYS safe) ----
        databases = st.secrets.get("ATHENA_DATABASES")
        default_db = st.secrets.get("ATHENA_DATABASE", None)

        if databases:
            db_hint = f"Available databases include: {', '.join(databases)}."
        elif default_db:
            db_hint = f"Assume default database is {default_db}."

        # ---- OPTIONAL: Glue schema + Iceberg-aware context ----
        try:
            from athena_metadata import build_schema_context

            region = st.secrets.get("AWS_REGION")
            databases = st.secrets.get("ATHENA_DATABASES")
            if not databases:
                databases = [default_db] if default_db else []

            schema_context = build_schema_context(
                databases=databases,
                region_name=region,
                max_tables_per_db=st.secrets.get("ATHENA_MAX_TABLES_PER_DB", 25),
                max_columns_per_table=st.secrets.get("ATHENA_MAX_COLS_PER_TABLE", 30),
            )
        except Exception:
            # Do not block SQL generation if Glue access is unavailable
            schema_context = ""
        assumption = (
            "Use Amazon Athena dialect (Presto/Trino-style), limit 20 rows. "
            f"{schema_context} "
            "Prefer marts or semantic, save cost, use partition filters and LIMIT when possible."
        )

    # Build final question
    else:
        assumption=''
        pass
    question_db = question + assumption
    return vn.generate_sql(question=question_db, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str, selected_db: str):
    vn = setup_vanna()
    #return vn.is_sql_valid(sql=sql, selected_db=selected_db)
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Generating dbt code ...")
def generate_dbt_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_dbt_code(question=question, sql=sql, df=df)
    return code

@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)



def maybe_train_warehouse_context(vn, selected_db: str):
    """
    Optional context injection. Only runs when Athena is selected.
    Does NOT assume Iceberg; Iceberg hints appear only if detected in Glue.
    """
    if selected_db not in ("Athena", "Athena (Iceberg)"):
        return

    # Lazy import so users who don't use Athena never load boto3 / glue logic
    try:
        from athena_metadata import build_schema_context
    except Exception as e:
        # Don't hard-fail the app. Just skip extra context.
        st.warning(f"Athena schema context not available: {e}")
        return

    # You can decide which databases to scan
    databases = st.secrets.get("ATHENA_DATABASES", [])
    if not databases:
        # fallback: a single schema, or simply skip if not provided
        default_db = st.secrets.get("ATHENA_SCHEMA", "default")
        databases = [default_db]

    region = st.secrets.get("AWS_REGION", None)

    schema_context = build_schema_context(
        databases=databases,
        region_name=region,
        max_tables_per_db=st.secrets.get("ATHENA_MAX_TABLES_PER_DB", 25),
        max_columns_per_table=st.secrets.get("ATHENA_MAX_COLS_PER_TABLE", 30),
    )

    # This "documentation" training is optional; if you prefer prompt injection instead, see below.
    vn.train(documentation=schema_context)

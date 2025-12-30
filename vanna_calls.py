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


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str, selected_db: str):
    vn = setup_vanna()
    if selected_db =='Snowflake':
        database=st.secrets["SNOWFLAKE_DATABASE"]
        schema=st.secrets["SNOWFLAKE_SCHEMA"]
        assumption = ' in ' + selected_db + f', "_" is used to split field, assuming database name is {database} and schema name is {schema}.'
    elif selected_db =='dbt':
        database=st.secrets["SNOWFLAKE_DATABASE"]
        schema=st.secrets["SNOWFLAKE_SCHEMA"]
        assumption = ' in dbt for snowflake'  + f' ,please generate sql like dbt code instead, using dbt syntax, assuming database name is {database} and schema name is {schema} '
    elif selected_db =='Redshift':
        database=st.secrets["REDSHIFT_DBNAME"]
        schema=st.secrets["REDSHIFT_SCHEMA"]
        assumption = ' in ' + selected_db + f' , assuming database name is {database} and schema name is {schema} '
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

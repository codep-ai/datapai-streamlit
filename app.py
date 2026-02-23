import time
import streamlit as st
from lightdash_api import execute_sql_in_lightdash
from code_editor import code_editor
from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_dbt_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached,
    setup_vanna
)
# app.py
from connect_db import connect_to_db

avatar_url = "https://www.datap.ai/images/datapai-logo.png"

#st.set_page_config(layout="wide")

# Streamlit app layout
st.title("DataPAI(DATA + AI) - Enterprise Data with AI ")
st.title("Generate SQL for these databases by AI ")

# Adding checkboxes for database options
db_options = ["Snowflake", "Redshift","Athena", "SQLite3","Bigquery", "DuckDB"]
default_db_index = db_options.index("SQLite3")  # Set SQLite3 as the default

# Initialize the global variable for selected database
if 'selected_db' not in st.session_state:
    st.session_state.selected_db = db_options[default_db_index]

# Create a selectbox for selecting a database and update the global variable
st.session_state.selected_db = st.selectbox("Select a Database", db_options, index=db_options.index(st.session_state.selected_db))
selected_db=st.session_state.selected_db

# Show selected option and connect to the database
if st.session_state.selected_db:
    st.write(f"Selected Database: {st.session_state.selected_db}")

    # Assuming connect_to_db is a function that connects to the database
    st.write(f"Connecting to {selected_db}...")

    # Use the selected database in the connect_to_db function
    conn = connect_to_db(st.session_state.selected_db)

st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL22", value=True, key="show_sql")
st.sidebar.checkbox("Show DBT", value=True, key="show_dbt_code")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True)

#debug only
st.sidebar.write(st.session_state)

my_question = st.session_state.get("my_question", default=None)
def set_question(question):
    st.session_state["my_question"] = question

assistant_message_suggested = st.chat_message(
    "assistant", avatar=avatar_url
)
if assistant_message_suggested.button("Click to show suggested questions", key="show_suggested"):
    st.session_state["my_question"] = None
    questions = generate_questions_cached()
    for i, question in enumerate(questions):
        time.sleep(0.05)
        button = st.button(
            question,
            on_click=set_question,
            args=(question,),
            key=f"suggested_question_{i}"  # Added unique key
        )

#my_question = st.session_state.get("my_question", default=None)

if my_question is None:
    my_question = st.chat_input(
        "Ask me a question about your data",
    )

if my_question:
    st.session_state["my_question"] = my_question
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    # 🔹 [MODIFIED] Retrieve dbt metadata with error handling
    try:
        metadata_context = get_dbt_metadata(my_question)
    except Exception as e:
        metadata_context = None
        st.error(f"⚠️ Failed to retrieve dbt metadata: {e}")
        
    # 🔹 [ADDED at ~Line 130] AI-powered Q&A for dbt metadata
    if metadata_context:
        prompt = f"Context:\n{metadata_context}\n\nUser Query: {my_question}\n\nAnswer:"
        dbt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in dbt, SQL, and analytics."},
                {"role": "user", "content": prompt}
            ],
            api_key="your-api-key"
        )
        assistant_message_dbt = st.chat_message("assistant", avatar=avatar_url)
        assistant_message_dbt.write(dbt_response["choices"][0]["message"]["content"])
        sql_query_prompt = f"{metadata_context} \n\n {my_question}" if metadata_context else my_question
    else:
        sql_query_prompt =""

    sql = generate_sql_cached(question=sql_query_prompt, selected_db=selected_db)    
    # add end
   # sql = generate_sql_cached(question=my_question,selected_db=selected_db)

    if sql:
        if is_sql_valid_cached(sql=sql, selected_db=selected_db):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message(
                    "assistant", avatar=avatar_url
                )
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
        else:
            assistant_message = st.chat_message(
                "assistant", avatar=avatar_url
            )
            assistant_message.write(sql)
            st.stop()

        df = run_sql_cached(sql=sql)

        if selected_db == "Athena":
            st.info("Using Amazon Athena on S3. Iceberg, Parquet, and CSV tables are supported.")
 
        # 🔵 Run SQL in Lightdash if Snowflake is selected
        lightdash_response = None
        if selected_db in ["Snowflake","Redshift","Bigquery"] :
            lightdash_response = execute_sql_in_lightdash(sql, selected_db)
            if "error" in lightdash_response:
                st.error("Failed to execute SQL in Lightdash: " + lightdash_response["error"])
            else:
                st.success("SQL successfully executed in Lightdash!")
                lightdash_dashboard_url = "https://platform.datap.ai/bi/projects/snowflake_datapai/dashboards/YOUR_DASHBOARD_ID"
                st.markdown(f"[View results in Lightdash]({lightdash_dashboard_url})")        

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                assistant_message_table = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                if len(df) > 20:
                    assistant_message_table.text("First 20 rows of data")
                    assistant_message_table.dataframe(df.head(20))
                else:
                    assistant_message_table.dataframe(df)


            if st.session_state.get("show_dbt_code", False):
                assistant_message_dbt_code = st.chat_message(
                    "dbt assistant",
                    avatar=avatar_url,
                )
                assistant_message_dbt_code.code(
                    code, language="python", line_numbers=True
                )


            if should_generate_chart_cached(question=my_question, sql=sql, df=df):

                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    assistant_message_plotly_code = st.chat_message(
                        "assistant",
                        avatar=avatar_url,
                    )
                    assistant_message_plotly_code.code(
                        code, language="python", line_numbers=True
                    )

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        assistant_message_chart = st.chat_message(
                            "assistant",
                            avatar=avatar_url,
                        )
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            assistant_message_chart.plotly_chart(fig)
                        else:
                            assistant_message_chart.error("I couldn't generate a chart")

            if st.session_state.get("show_summary", True):
                assistant_message_summary = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    assistant_message_summary.text(summary)

            if st.session_state.get("show_followup", True):
                assistant_message_followup = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                followup_questions = generate_followup_cached(
                    question=my_question, sql=sql, df=df
                )
                st.session_state["df"] = None

                if len(followup_questions) > 0:
                    assistant_message_followup.text(
                        "Here are some possible follow-up questions"
                    )
                    # Print the first 5 follow-up questions
                    for i, question in enumerate(followup_questions[:5]):
                        assistant_message_followup.button(
                            question, on_click=set_question, args=(question,), key=f"followup_question_{i}"  # Added unique key
                        )

    else:
        assistant_message_error = st.chat_message(
            "assistant", avatar=avatar_url
        )
        assistant_message_error.error("I wasn't able to generate SQL for that question")


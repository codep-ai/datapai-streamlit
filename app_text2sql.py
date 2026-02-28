import time
import streamlit as st
from code_editor import code_editor
from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_dbt_code_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached,
    setup_vanna
)
from connect_db import connect_to_db

def run_text2sql():
    avatar_url = "https://www.datap.ai/images/datapai-logo.png"

    st.title("DataPAI(DATA + AI) - Enterprise Data with AI ")
    st.title("Generate SQL for these databases by AI ")

    db_options = ["Snowflake", "Redshift", "Athena", "SQLite3", "Bigquery", "DuckDB", "dbt"]
    default_db_index = db_options.index("SQLite3")

    if 'selected_db' not in st.session_state:
        st.session_state.selected_db = db_options[default_db_index]

    st.session_state.selected_db = st.selectbox("Select a Database", db_options, index=db_options.index(st.session_state.selected_db))
    selected_db = st.session_state.selected_db

    if st.session_state.selected_db:
        st.write(f"Selected Database: {st.session_state.selected_db}")
        st.write(f"Connecting to {selected_db}...")
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

    st.sidebar.write(st.session_state)

    my_question = st.session_state.get("my_question", default=None)

    def set_question(question):
        st.session_state["my_question"] = question

    assistant_message_suggested = st.chat_message("assistant", avatar=avatar_url)
    if assistant_message_suggested.button("Click to show suggested questions", key="show_suggested"):
        st.session_state["my_question"] = None
        questions = generate_questions_cached()
        for i, question in enumerate(questions):
            time.sleep(0.05)
            st.button(question, on_click=set_question, args=(question,), key=f"suggested_question_{i}")

    if my_question is None:
        my_question = st.chat_input("Ask me a question about your data")

    if my_question:
        st.session_state["my_question"] = my_question
        user_message = st.chat_message("user")
        user_message.write(f"{my_question}")

        sql = generate_sql_cached(question=my_question, selected_db=selected_db)

        if sql:
            if is_sql_valid_cached(sql=sql, selected_db=selected_db):
                if st.session_state.get("show_sql", True):
                    assistant_message_sql = st.chat_message("assistant", avatar=avatar_url)
                    assistant_message_sql.code(sql, language="sql", line_numbers=True)
            else:
                assistant_message = st.chat_message("assistant", avatar=avatar_url)
                assistant_message.write(sql)
                st.stop()

            df = run_sql_cached(sql=sql)
            if df is not None:
                st.session_state["df"] = df

            if st.session_state.get("df") is not None:
                if st.session_state.get("show_table", True):
                    df = st.session_state.get("df")
                    assistant_message_table = st.chat_message("assistant", avatar=avatar_url)
                    assistant_message_table.dataframe(df.head(20) if len(df) > 20 else df)

                if st.session_state.get("show_dbt_code", False):
                    assistant_message_sql = st.chat_message("dbt assistant", avatar=avatar_url)
                    assistant_message_sql.code(sql, language="sql", line_numbers=True)

                if should_generate_chart_cached(question=my_question, sql=sql, df=df):
                    code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)
                    if st.session_state.get("show_plotly_code", False):
                        assistant_message_plotly_code = st.chat_message("assistant", avatar=avatar_url)
                        assistant_message_plotly_code.code(code, language="python", line_numbers=True)

                    if code:
                        if st.session_state.get("show_chart", True):
                            assistant_message_chart = st.chat_message("assistant", avatar=avatar_url)
                            fig = generate_plot_cached(code=code, df=df)
                            if fig is not None:
                                assistant_message_chart.plotly_chart(fig)
                            else:
                                assistant_message_chart.error("I couldn't generate a chart")

                if st.session_state.get("show_summary", True):
                    assistant_message_summary = st.chat_message("assistant", avatar=avatar_url)
                    summary = generate_summary_cached(question=my_question, df=df)
                    if summary:
                        assistant_message_summary.text(summary)

                if st.session_state.get("show_followup", True):
                    assistant_message_followup = st.chat_message("assistant", avatar=avatar_url)
                    followup_questions = generate_followup_cached(question=my_question, sql=sql, df=df)
                    st.session_state["df"] = None

                    if followup_questions:
                        assistant_message_followup.text("Here are some possible follow-up questions")
                        for i, question in enumerate(followup_questions[:5]):
                            assistant_message_followup.button(question, on_click=set_question, args=(question,), key=f"followup_question_{i}")

        else:
            assistant_message_error = st.chat_message("assistant", avatar=avatar_url)
            assistant_message_error.error("I wasn't able to generate SQL for that question")

if __name__ == "__main__":
    run_text2sql()


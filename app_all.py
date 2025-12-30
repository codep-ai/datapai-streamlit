import streamlit as st
from app_text2sql import run_text2sql
from app_ai_agent import run_ai_agents

st.set_page_config(page_title="DataPAI: AI-Driven Data Platform", layout="wide")

st.title("DataPAI: Unified AI Services")

page = st.sidebar.radio(
    "Choose a service",
    ["Text2SQL", "AI agent for digital transformation"]
)

if page == "Text2SQL":
    run_text2sql()
elif page == "AI agent for digital transformation":
    run_ai_agents()


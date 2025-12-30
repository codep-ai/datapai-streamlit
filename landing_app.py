import streamlit as st
import os

st.set_page_config(page_title="AI Data Platform", layout="centered")

st.title("🚀 AI Data Platform")
st.markdown("Choose a module to get started:")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Text-to-SQL"):
        os.system("streamlit run app_all.py")

with col2:
    if st.button("⚙️ AI Agent ETL"):
        os.system("streamlit run app_all_ai.py")


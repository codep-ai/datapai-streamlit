import streamlit as st
from agents.agent_registry import AgentRegistry
from utils.mcp_client import MCPClient

# Load available agent names
registry = AgentRegistry()
agent_names = registry.get_agent_names()

# UI Sidebar
st.sidebar.title("AI Agent Config")
selected_agent = st.sidebar.selectbox("Select Agent", agent_names)
source_config = {}

# Optional: Read MCP endpoints from registry.yaml
mcp_client = MCPClient()

# CSV Agent UI
if selected_agent == "csv_ingest":
    st.title("CSV Ingestion Agent")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    destination = st.selectbox("Select destination", ["bigquery", "postgres"])  # can be expanded
    table_name = st.text_input("Target table name")

    if st.button("Run Agent") and uploaded_file and table_name:
        source_config = {
            "uploaded_file": uploaded_file,
            "destination": destination,
            "table_name": table_name
        }
        agent = registry.get_agent(selected_agent)
        mcp_metadata = agent.run(source_config)
        st.success("Ingestion completed")
        st.json(mcp_metadata)

# Airbyte Agent UI
elif selected_agent == "airbyte_ingest":
    st.title("Airbyte Ingestion Agent")
    connection_id = st.text_input("Airbyte Connection ID")
    if st.button("Run Agent") and connection_id:
        source_config = {"connection_id": connection_id}
        agent = registry.get_agent(selected_agent)
        mcp_metadata = agent.run(source_config)
        st.success("Airbyte sync completed")
        st.json(mcp_metadata)

# DLT Agent UI
elif selected_agent == "dlt_ingest":
    st.title("DLT Ingestion Agent")
    source_name = st.text_input("Source module name (e.g., github)")
    dataset_name = st.text_input("Dataset name")
    destination = st.selectbox("Destination", ["bigquery", "postgres", "duckdb"])
    if st.button("Run Agent") and source_name and dataset_name:
        source_config = {
            "source_name": source_name,
            "destination": destination,
            "dataset_name": dataset_name
        }
        agent = registry.get_agent(selected_agent)
        mcp_metadata = agent.run(source_config)
        st.success("DLT pipeline completed")
        st.json(mcp_metadata)

# Terraform Agent UI (AWS or Azure)
elif selected_agent in ["terraform_aws", "terraform_azure"]:
    st.title(f"Terraform Agent ({selected_agent})")
    env = st.selectbox("Environment", ["dev", "prod"])
    action = st.selectbox("Action", ["init", "plan", "apply"])
    confirm = st.checkbox("Confirm execution")

    if st.button("Run Terraform") and confirm:
        source_config = {
            "env": env,
            "action": action
        }
        agent = registry.get_agent(selected_agent)
        mcp_metadata = agent.run(source_config)
        st.success(f"Terraform {action} completed")
        st.json(mcp_metadata)

else:
    st.warning("Agent UI not implemented yet.")


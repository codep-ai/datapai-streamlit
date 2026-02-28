def load_all_tools():
    # Import modules that register @tool functions.
    # IMPORTANT: do NOT import agent_base here.
    from .. import file_ingest_agent  # noqa
    from .. import dbt_agent          # noqa
    from .. import snowflake_ingest_agent  # noqa
    # add more as needed

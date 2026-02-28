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
            return None, f"Unsupported DB type: {db_type}"

        cur = conn.cursor()
        cur.execute(sql)

        # Try fetch
        try:
            df = pd.DataFrame(cur.fetchall(), columns=[col[0] for col in cur.description])
        except Exception:
            df = None

        # Commit for DML
        try:
            conn.commit()
        except Exception:
            pass

        cur.close()
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


# ------------------------------
# Exported entry point
# ------------------------------

def _render_budget_sidebar() -> None:
    """Show today's LLM API spend vs daily budget in the sidebar."""
    try:
        from agents.cost_guard import CostGuard
        s = CostGuard().status()
    except Exception:
        return   # don't crash the app if cost_guard can't load

    if not s.get("enabled", True):
        st.sidebar.caption("💰 Cost guard disabled")
        return

    spent     = s["spent_today"]
    budget    = s["budget_usd"]
    remaining = s["remaining_usd"]
    pct       = s["pct_used"]
    calls     = s["calls_today"]

    st.sidebar.markdown("### 💰 Daily LLM Budget")
    st.sidebar.progress(min(pct / 100, 1.0))

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Spent",     f"${spent:.4f}")
    col2.metric("Remaining", f"${remaining:.4f}")

    status_colour = "🟢" if pct < 75 else ("🟡" if pct < 95 else "🔴")
    st.sidebar.caption(
        f"{status_colour} {pct}% of ${budget:.2f} daily budget used "
        f"· {calls} call{'s' if calls != 1 else ''} today"
    )
    if remaining <= 0:
        st.sidebar.error(
            "⛔ Daily budget exhausted — paid LLM calls are blocked. "
            "Increase `DAILY_LLM_BUDGET_USD` or wait until midnight."
        )


def run_ai_agents():
    st.title("🤖 Data P AI Agent Hub")

    _render_budget_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Run SQL",
        "dbt Agent",
        "Knowledge Ingest Agent",
        "Ingest File",
        "DLT Ingest",
        "Airbyte Sync",
        "Workflow (Ingest → dbt)",
        "ASX Announcements",
    ])

    with tab1:
        st.header("Run SQL")
        st.caption("Paste SQL and choose destination DB.")
        sql = st.text_area("SQL", height=150)
        db = st.selectbox("Database", ["Snowflake", "Redshift", "Bigquery", "SQLite3", "DuckDB"])
        if st.button("Execute"):
            if not sql.strip():
                st.warning("Please enter SQL.")
            else:
                df, err = _run_sql_via_connector(db, sql)
                if err:
                    st.error(err)
                else:
                    if df is not None and not df.empty:
                        st.dataframe(df)
                    else:
                        st.success("Query executed (no rows returned).")

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
        st.header("📚 Knowledge Base — Ingest & Chat")

        # ── Session state initialisation ───────────────────────────────────
        if "rag_chat_history" not in st.session_state:
            st.session_state.rag_chat_history = []   # [{role, content}]
        if "rag_docs_ingested" not in st.session_state:
            st.session_state.rag_docs_ingested = []  # list of ingested filenames

        # ── Layout: sidebar panel (ingest) + main area (chat) ─────────────
        ingest_col, chat_col = st.columns([1, 2])

        # ─── LEFT: Ingest panel ───────────────────────────────────────────
        with ingest_col:
            st.subheader("📥 Ingest Documents")

            uploads = st.file_uploader(
                "Upload PDF, TXT, MD, CSV, or images",
                accept_multiple_files=True,
                type=["pdf", "txt", "md", "csv", "png", "jpg", "jpeg"],
                key="rag_uploader",
            )

            model_choice = st.text_input(
                "Ollama model",
                value=os.environ.get("RAG_LLM_MODEL", "llama3.2"),
                key="rag_model",
                help="Model name available in your local Ollama instance",
            )

            top_k = st.slider("Documents to retrieve (k)", 1, 20, 5, key="rag_k")

            if uploads and st.button("📤 Ingest to Vector Store", key="rag_ingest_btn"):
                temp_paths = []
                for f in uploads:
                    path = f"/tmp/{f.name}"
                    with open(path, "wb") as out:
                        out.write(f.read())
                    temp_paths.append(path)

                with st.spinner("Ingesting into LanceDB…"):
                    from agents.knowledge_ingest_agent import run_knowledge_ingest_agent
                    run_knowledge_ingest_agent(temp_paths)

                for f in uploads:
                    if f.name not in st.session_state.rag_docs_ingested:
                        st.session_state.rag_docs_ingested.append(f.name)
                st.success(f"✅ Ingested {len(uploads)} document(s)")

            # Show ingested docs
            if st.session_state.rag_docs_ingested:
                st.caption("**Ingested this session:**")
                for doc in st.session_state.rag_docs_ingested:
                    st.caption(f"• {doc}")

            st.divider()

            # Clear chat
            if st.button("🗑 Clear chat history", key="rag_clear"):
                st.session_state.rag_chat_history = []
                st.rerun()

        # ─── RIGHT: Chat panel ────────────────────────────────────────────
        with chat_col:
            st.subheader("💬 Ask about your documents")

            # Render existing chat history
            for msg in st.session_state.rag_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("📎 Sources", expanded=False):
                            for src in msg["sources"]:
                                st.caption(
                                    f"**{src.get('filename','?')}** "
                                    f"[{src.get('collection','?')}]  "
                                    f"`{src.get('source_uri','')}`"
                                )

            # Chat input — renders at bottom, triggers on Enter
            user_input = st.chat_input(
                "Ask a question about your ingested documents…",
                key="rag_chat_input",
            )

            if user_input and user_input.strip():
                # Add user message to history and render immediately
                st.session_state.rag_chat_history.append({
                    "role": "user",
                    "content": user_input,
                })
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Build history for RAG (exclude the current turn — it's the question)
                history_for_rag = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.rag_chat_history[:-1]
                ]

                # Call RAG + Ollama with full history
                with st.chat_message("assistant"):
                    with st.spinner("Searching knowledge base and thinking…"):
                        try:
                            from agents.knowledge_query_agent import (
                                search_lancedb,
                                build_context_from_results,
                            )
                            import requests as _req

                            ollama_host = os.environ.get(
                                "OLLAMA_HOST",
                                os.environ.get("STREAMLIT_OLLAMA_HOST", "http://localhost:11434"),
                            )
                            resolved_model = model_choice or os.environ.get("RAG_LLM_MODEL", "llama3.2")

                            # Retrieve relevant docs from LanceDB
                            df = search_lancedb(user_input, k=top_k)
                            context = build_context_from_results(df, max_chars=6000)

                            # Build sources for display
                            sources = []
                            if not df.empty:
                                for _, row in df.iterrows():
                                    sources.append({
                                        "filename":   str(row.get("filename", "")),
                                        "collection": str(row.get("collection", "")),
                                        "source_uri": str(row.get("source_uri", "")),
                                    })

                            # Build full message list with history
                            messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a helpful data and documentation assistant. "
                                        "Use ONLY the provided context to answer. "
                                        "If the answer is not in the context, say you don't know. "
                                        "Do NOT invent facts."
                                    ),
                                }
                            ]
                            # Inject previous conversation turns
                            for turn in history_for_rag:
                                messages.append({"role": turn["role"], "content": turn["content"]})

                            # Augment current question with RAG context
                            augmented = (
                                f"Context from knowledge base:\n"
                                f"{'─'*50}\n{context}\n{'─'*50}\n\n"
                                f"Question: {user_input}"
                            )
                            messages.append({"role": "user", "content": augmented})

                            # Call Ollama
                            resp = _req.post(
                                f"{ollama_host}/api/chat",
                                json={"model": resolved_model, "messages": messages, "stream": False},
                                timeout=300,
                            )
                            resp.raise_for_status()
                            answer = resp.json().get("message", {}).get("content", "No answer returned.")

                        except Exception as exc:
                            answer = f"⚠️ Error: {exc}"
                            sources = []

                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 Sources", expanded=False):
                            for src in sources:
                                st.caption(
                                    f"**{src.get('filename','?')}** "
                                    f"[{src.get('collection','?')}]  "
                                    f"`{src.get('source_uri','')}`"
                                )

                # Store assistant reply in history
                st.session_state.rag_chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

    with tab4:
        st.header("Ingest File")
        st.caption("Upload CSV/Parquet and load into a target database table.")
        up = st.file_uploader("Upload a file", type=["csv", "parquet"])
        destination = st.selectbox("Destination", ["snowflake", "redshift", "bigquery", "duckdb", "sqlite3"])
        schema_name = st.text_input("Schema/Dataset", value="src_file")
        table_name = st.text_input("Table name", value="src_uploaded_file")
        overwrite = st.checkbox("Overwrite table", value=True)

        if st.button("Run File ingest"):
            if not up:
                st.warning("Please upload a file.")
            else:
                # Save temp file
                tmp_path = f"/tmp/{up.name}"
                with open(tmp_path, "wb") as out:
                    out.write(up.read())

                from agents.file_ingest_agent import run_file_ingest_agent
                ok, msg = run_file_ingest_agent(
                    file_path=tmp_path,
                    destination=destination,
                    schema_or_dataset=schema_name,
                    table_name=table_name,
                    overwrite=overwrite,
                )
                st.success(msg) if ok else st.error(msg)

    with tab5:
        st.header("DLT Ingest")
        st.caption("Run a simple DLT ingest from a source URI (e.g., S3 CSV) into a destination.")
        source_uri = st.text_input("Source URI (e.g., s3://bucket/path/file.csv)", value="")
        destination = st.selectbox("Destination (DLT)", ["duckdb", "bigquery", "snowflake", "redshift"], key="dlt_dest")
        ds_or_schema = st.text_input("Dataset/Schema", value="src_file", key="dlt_schema")
        table_name = st.text_input("Table name", value="src_dlt", key="dlt_table")

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
        connection_id = st.text_input(
            "Airbyte Connection ID (UUID)",
            value=os.environ.get("AIRBYTE_CONNECTION_ID", "")
        )
        poll = st.checkbox("Poll until completion", value=True)

        if st.button("Trigger sync"):
            if not connection_id:
                st.warning("Provide a connection id.")
            else:
                from agents.airbyte_ingest_agent import ingest_with_airbyte
                ok, msg = ingest_with_airbyte(connection_id=connection_id, poll=poll)
                st.success(msg) if ok else st.error(msg)

    # NEW: Tab 7 - Canonical Workflow (Phase 5 UI)
    with tab7:  # NEW
        st.header("Canonical Workflow: Ingest → dbt")
        st.caption("Deterministic runner (Phase 3) with optional Supervisor planning (Phase 4).")

        source_type = st.selectbox("Source type", ["s3_csv", "s3_parquet", "local_file"], key="wf_source_type")
        source = st.text_input("Source (S3 URI or local path)", value="", key="wf_source")

        target = st.selectbox("Target", ["snowflake", "redshift", "bigquery", "duckdb", "sqlite3"], key="wf_target")
        target_schema = st.text_input("Target schema", value="src_file", key="wf_target_schema")
        target_table = st.text_input("Target table", value="stg_example", key="wf_target_table")
        mode = st.selectbox("Load mode", ["upsert", "overwrite"], key="wf_mode")
        pk_text = st.text_input("Primary key columns (comma separated, optional)", value="", key="wf_pk")
        pk = [c.strip() for c in pk_text.split(",") if c.strip()] or None

        use_supervisor = st.checkbox("Use Supervisor planning (LLM)", value=True, key="wf_use_supervisor")
        dry_run = st.checkbox("Dry run (plan only)", value=False, key="wf_dry_run")

        user_request = {
            "workflow": "ingest_to_dbt",
            "source_type": source_type,
            "source": source,
            "target": target,
            "target_schema": target_schema,
            "target_table": target_table,
            "mode": mode,
            "pk": pk,
        }

        if st.button("Run workflow", key="wf_run"):
            if not source:
                st.warning("Please provide a source path/URI.")
                st.stop()

            plan_obj = None
            if use_supervisor:
                try:
                    # Phase 4 expected interfaces:
                    # - SupervisorAgent.plan(user_request) -> plan object (dataclass or dict)
                    # - etl.workflow_service.execute_plan(plan_obj) -> RunContext
                    from agents.llm_client import RouterChatClient
                    from agents.supervisor_agent import SupervisorAgent

                    llm = RouterChatClient()
                    supervisor = SupervisorAgent(llm)

                    with st.spinner("Planning with Supervisor (LLM)..."):
                        plan_obj = supervisor.plan(user_request)

                    st.subheader("Plan")
                    st.json(plan_obj.__dict__ if hasattr(plan_obj, "__dict__") else plan_obj)

                except Exception as e:
                    st.warning(
                        f"Supervisor planning unavailable; running deterministically without plan. Details: {e}"
                    )

            if dry_run:
                st.success("Dry run complete (plan only).")
                st.stop()

            try:
                # Preferred path (Phase 4): execute Supervisor plan -> deterministic runner
                if plan_obj is not None:
                    from etl.workflow_service import execute_plan
                    with st.spinner("Executing workflow..."):
                        ctx = execute_plan(plan_obj)
                else:
                    # Fallback path (Phase 3): call deterministic runner directly
                    from etl.contracts import WorkflowRequest
                    from etl.workflow_runner import run_ingest_to_dbt

                    req = WorkflowRequest(
                        source_type=source_type,
                        source=source,
                        target=target,
                        target_schema=target_schema,
                        target_table=target_table,
                        mode=mode,
                        pk=pk,
                        options={},
                    )
                    with st.spinner("Executing deterministic workflow..."):
                        ctx = run_ingest_to_dbt(req)

                st.subheader("Run Result")
                st.write("Run ID:", getattr(ctx, "run_id", "n/a"))
                st.json({
                    "artifacts": getattr(ctx, "artifacts", {}),
                    "metrics": getattr(ctx, "metrics", {}),
                    "logs_tail": (getattr(ctx, "logs", [])[-50:] if getattr(ctx, "logs", None) else []),
                })
                st.success("Workflow completed.")

            except Exception as e:
                st.error(f"Workflow failed: {e}")

    # ── Tab 8: ASX Market Announcements ────────────────────────────────────────
    with tab8:
        st.header("📈 ASX Market Announcements")
        st.caption(
            "Fetch and interpret ASX announcements in real-time — no manual PDF upload needed. "
            "Use **Quick Interpret** for instant LLM analysis, or **Ingest** to add to the knowledge base for RAG."
        )

        # ── Session state init ──────────────────────────────────────────────
        for _key, _default in {
            "asx_announcements":  [],
            "asx_pdf_cache":      {},   # {url: pdf_text}
            "asx_interpretation": "",
            "asx_selected_idx":   0,
            "asx_chat_history":   [],
            "asx_signal":         "",   # latest trading signal markdown
            "asx_signal_idx":     0,    # which announcement the signal is for
        }.items():
            if _key not in st.session_state:
                st.session_state[_key] = _default

        # ── Controls ────────────────────────────────────────────────────────
        ctrl_col, _ = st.columns([2, 1])
        with ctrl_col:
            tickers_input = st.text_input(
                "ASX Ticker(s)",
                placeholder="e.g.  BHP, CBA, RIO",
                help="Enter one or more ASX ticker symbols separated by commas.",
                key="asx_ticker_input",
            )
            ann_count = st.slider(
                "Number of announcements per ticker", 5, 50, 20, key="asx_count"
            )
            ms_only = st.checkbox(
                "Market-sensitive only", key="asx_ms_only",
                help="Tick to return only announcements flagged as market-sensitive."
            )

        fetch_btn = st.button("🔍 Fetch Announcements", key="asx_fetch")

        if fetch_btn and tickers_input.strip():
            
            from agents.asx_announcement_agent import fetch_asx_announcements as _fetch

            tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
            all_anns: list = []
            for ticker in tickers:
                with st.spinner(f"Fetching announcements for {ticker}…"):
                    try:
                        fetched = _fetch(ticker, count=ann_count, market_sensitive_only=ms_only)
                        all_anns.extend(fetched)
                        st.success(f"✅ {ticker}: {len(fetched)} announcement(s) fetched")
                    except Exception as exc:
                        st.error(f"❌ {ticker}: {exc}")

            st.session_state.asx_announcements  = all_anns
            st.session_state.asx_interpretation = ""
            st.session_state.asx_chat_history   = []
            st.session_state.asx_selected_idx   = 0
            

        # ── Announcements table + actions ───────────────────────────────────
        if st.session_state.asx_announcements:
            anns = st.session_state.asx_announcements

            df_display = pd.DataFrame([{
                "Ticker":    a["ticker"],
                "Date":      (a["document_date"] or "")[:10],
                "Headline":  a["headline"],
                "Type":      a["doc_type"],
                "Pages":     a["number_of_pages"],
                "Size KB":   a["size_kb"],
                "Sensitive": "🔴" if a["market_sensitive"] else "",
            } for a in anns])

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.divider()

            # Announcement selector
            headline_options = [
                f"{a['ticker']} | {(a['document_date'] or '')[:10]} | {a['headline'][:70]}"
                for a in anns
            ]
            selected_label = st.selectbox(
                "Select announcement to act on",
                headline_options,
                index=min(st.session_state.asx_selected_idx, len(headline_options) - 1),
                key="asx_select",
            )
            selected_idx = headline_options.index(selected_label)
            selected_ann = anns[selected_idx]

            # Per-announcement action buttons
            act_col1, act_col2, act_col3, act_col4 = st.columns(4)

            with act_col1:
                if st.button("⚡ Quick Interpret (direct LLM)", key="asx_interpret"):
                    from agents.asx_announcement_agent import (
                        download_pdf_bytes,
                        extract_text_from_pdf_bytes,
                        interpret_announcement,
                    )
                    pdf_url = selected_ann.get("url", "")
                    with st.spinner("Downloading PDF and interpreting…"):
                        try:
                            if pdf_url not in st.session_state.asx_pdf_cache:
                                pdf_bytes = download_pdf_bytes(pdf_url)
                                pdf_text  = extract_text_from_pdf_bytes(pdf_bytes)
                                st.session_state.asx_pdf_cache[pdf_url] = pdf_text
                            else:
                                pdf_text = st.session_state.asx_pdf_cache[pdf_url]

                            interpretation = interpret_announcement(selected_ann, pdf_text)
                            st.session_state.asx_interpretation = interpretation
                            st.session_state.asx_chat_history   = []
                            st.session_state.asx_selected_idx   = selected_idx
                        except Exception as exc:
                            st.error(f"Interpretation failed: {exc}")

            with act_col2:
                if st.button("📥 Ingest to Knowledge Base", key="asx_ingest_one"):
                    from agents.asx_announcement_agent import (
                        download_pdf_bytes,
                        extract_text_from_pdf_bytes,
                        ingest_announcement_to_lancedb,
                    )
                    pdf_url = selected_ann.get("url", "")
                    db_uri  = os.environ.get("LANCEDB_URI", "s3://codepais3/lancedb_data/")
                    with st.spinner("Downloading and ingesting to LanceDB…"):
                        try:
                            if pdf_url not in st.session_state.asx_pdf_cache:
                                pdf_bytes = download_pdf_bytes(pdf_url)
                                pdf_text  = extract_text_from_pdf_bytes(pdf_bytes)
                                st.session_state.asx_pdf_cache[pdf_url] = pdf_text
                            else:
                                pdf_text = st.session_state.asx_pdf_cache[pdf_url]

                            result = ingest_announcement_to_lancedb(
                                selected_ann, pdf_text, db_uri=db_uri
                            )
                            if result["status"] == "ingested":
                                st.success(f"✅ Ingested: {result['filename']}")
                            else:
                                st.info(f"ℹ️ {result.get('reason', 'Already ingested')}")
                        except Exception as exc:
                            st.error(f"Ingest failed: {exc}")

            with act_col3:
                if st.button("📦 Ingest ALL to Knowledge Base", key="asx_ingest_all"):
                    from agents.asx_announcement_agent import (
                        download_pdf_bytes,
                        extract_text_from_pdf_bytes,
                        ingest_announcement_to_lancedb,
                    )
                    db_uri  = os.environ.get("LANCEDB_URI", "s3://codepais3/lancedb_data/")
                    results = []
                    prog    = st.progress(0, "Ingesting…")
                    for i, ann in enumerate(anns):
                        pdf_url = ann.get("url", "")
                        try:
                            if pdf_url not in st.session_state.asx_pdf_cache:
                                pdf_bytes = download_pdf_bytes(pdf_url)
                                pdf_text  = extract_text_from_pdf_bytes(pdf_bytes)
                                st.session_state.asx_pdf_cache[pdf_url] = pdf_text
                            else:
                                pdf_text = st.session_state.asx_pdf_cache[pdf_url]

                            res = ingest_announcement_to_lancedb(ann, pdf_text, db_uri=db_uri)
                            results.append(res)
                        except Exception as exc:
                            results.append({
                                "status":   "error",
                                "filename": pdf_url,
                                "reason":   str(exc),
                            })
                        prog.progress(
                            (i + 1) / len(anns),
                            f"Processing {i + 1}/{len(anns)}: {ann.get('headline', '')[:50]}"
                        )

                    ingested = sum(1 for r in results if r["status"] == "ingested")
                    skipped  = sum(1 for r in results if r["status"] == "skipped")
                    errors   = sum(1 for r in results if r["status"] == "error")
                    st.success(
                        f"Done — ✅ ingested: {ingested}, "
                        f"⏭ skipped: {skipped}, "
                        f"❌ errors: {errors}"
                    )

            with act_col4:
                if st.button("🎯 Trading Signal", key="asx_signal_btn",
                             help="AI trading signal: buy/sell/hold + price targets per timeframe. NOT financial advice."):
                    from agents.asx_announcement_agent import (
                        download_pdf_bytes,
                        extract_text_from_pdf_bytes,
                    )
                    from agents.asx_trading_signal import (
                        fetch_all_timeframes,
                        generate_trading_signal,
                    )
                    pdf_url = selected_ann.get("url", "")
                    with st.spinner(
                        "Fetching live price data + generating AI trading signal… (~20–40 s)"
                    ):
                        try:
                            # Reuse cached PDF text
                            if pdf_url not in st.session_state.asx_pdf_cache:
                                pdf_bytes = download_pdf_bytes(pdf_url)
                                pdf_text  = extract_text_from_pdf_bytes(pdf_bytes)
                                st.session_state.asx_pdf_cache[pdf_url] = pdf_text
                            else:
                                pdf_text = st.session_state.asx_pdf_cache[pdf_url]

                            # Fetch multi-timeframe OHLCV + indicators
                            ticker_sym       = selected_ann.get("ticker", "")
                            indicators_by_tf = fetch_all_timeframes(ticker_sym)
                            available        = [tf for tf, v in indicators_by_tf.items() if v]
                            if not available:
                                st.warning(
                                    f"⚠️ Live price data unavailable for {ticker_sym}.AX — "
                                    "signal will be based on announcement text only. "
                                    "Yahoo Finance may be rate-limiting or the ticker is lightly traded."
                                )
                            else:
                                st.info(f"📊 Price data loaded for timeframes: {', '.join(available)}")

                            signal_md = generate_trading_signal(
                                selected_ann, pdf_text, indicators_by_tf
                            )
                            st.session_state.asx_signal     = signal_md
                            st.session_state.asx_signal_idx = selected_idx
                        except Exception as exc:
                            st.error(f"❌ Signal generation failed: {exc}")

            # ── Trading signal display ───────────────────────────────────────
            if st.session_state.asx_signal:
                sig_ann = anns[st.session_state.asx_signal_idx]
                st.divider()
                st.subheader(
                    f"🎯 AI Trading Signal — {sig_ann['ticker']} "
                    f"({(sig_ann['document_date'] or '')[:10]})"
                )
                st.error(
                    "⚠️ **NOT FINANCIAL ADVICE** — AI-generated signal for informational "
                    "and educational purposes only. Always consult a licensed financial "
                    "adviser before making any investment decision. DataPAI accepts no "
                    "responsibility for trading decisions based on this output.",
                    icon="⚠️",
                )
                st.markdown(st.session_state.asx_signal)
                st.caption(
                    f"Signal generated from: {sig_ann['headline'][:80]}  |  "
                    "LLM chain: Gemini flash-lite → GPT-5.1 reviewer"
                )

            # ── Interpretation display + follow-up chat ─────────────────────
            if st.session_state.asx_interpretation:
                st.divider()
                ann = anns[st.session_state.asx_selected_idx]
                st.subheader(
                    f"📊 {ann['ticker']} — {(ann['document_date'] or '')[:10]}"
                )
                st.caption(ann["headline"])
                st.markdown(st.session_state.asx_interpretation)

                st.divider()
                st.subheader("💬 Follow-up Questions")

                for msg in st.session_state.asx_chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                follow_up = st.chat_input(
                    "Ask a follow-up about this announcement…",
                    key="asx_chat_input",
                )
                if follow_up and follow_up.strip():
                    from agents.asx_announcement_agent import interpret_announcement

                    st.session_state.asx_chat_history.append(
                        {"role": "user", "content": follow_up}
                    )
                    with st.chat_message("user"):
                        st.markdown(follow_up)

                    ref_ann  = anns[st.session_state.asx_selected_idx]
                    pdf_url  = ref_ann.get("url", "")
                    pdf_text = st.session_state.asx_pdf_cache.get(pdf_url, "")

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking…"):
                            try:
                                answer = interpret_announcement(
                                    ref_ann, pdf_text, question=follow_up
                                )
                            except Exception as exc:
                                answer = f"⚠️ Error: {exc}"
                        st.markdown(answer)

                    st.session_state.asx_chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

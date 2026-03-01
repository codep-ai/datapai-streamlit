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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Run SQL",
        "dbt Agent",
        "Knowledge Ingest Agent",
        "Ingest File",
        "DLT Ingest",
        "Airbyte Sync",
        "Workflow (Ingest → dbt)",
        "📢 Market Announcements",
        "📊 Technical Analysis",
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

    # ── Tab 8: Market Announcements (ASX + US SEC EDGAR) ──────────────────────
    with tab8:
        st.header("📢 Market Announcements")

        # ── Market selector ─────────────────────────────────────────────────
        market_choice = st.radio(
            "Select market",
            ["🇦🇺 ASX (Australian Securities Exchange)",
             "🇺🇸 US (SEC EDGAR — NYSE / NASDAQ)"],
            horizontal=True,
            key="market_choice",
        )
        _is_us = market_choice.startswith("🇺🇸")

        # ── Session state init — ASX ────────────────────────────────────────
        for _key, _default in {
            "asx_announcements":  [],
            "asx_pdf_cache":      {},
            "asx_interpretation": "",
            "asx_selected_idx":   0,
            "asx_chat_history":   [],
            "asx_signal":         "",
            "asx_signal_idx":     0,
        }.items():
            if _key not in st.session_state:
                st.session_state[_key] = _default

        # ── Session state init — US ─────────────────────────────────────────
        for _key, _default in {
            "us_filings":         [],
            "us_text_cache":      {},   # {url: text}
            "us_interpretation":  "",
            "us_selected_idx":    0,
            "us_chat_history":    [],
            "us_signal":          "",
            "us_signal_idx":      0,
        }.items():
            if _key not in st.session_state:
                st.session_state[_key] = _default

        st.divider()

        # ════════════════════════════════════════════════════════════════════
        # ASX PATH
        # ════════════════════════════════════════════════════════════════════
        if not _is_us:
            st.subheader("📈 ASX Market Announcements")
            st.caption(
                "Fetch and interpret ASX announcements in real-time — no manual PDF upload needed. "
                "Use **Quick Interpret** for instant LLM analysis, or **Ingest** to add to the knowledge base for RAG."
            )

        # ── ASX Controls ────────────────────────────────────────────────────
        if not _is_us:
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

                # ── RAG cross-search: find similar past ASX announcements ────
                with st.expander("🔍 Find Similar Past Announcements in Knowledge Base", expanded=False):
                    st.caption(
                        "Semantic search across ingested ASX announcements in the "
                        "LanceDB vector store — find historical filings with similar "
                        "content to the current signal."
                    )
                    rag_query_asx = (
                        f"{sig_ann.get('ticker', '')} "
                        f"{sig_ann.get('headline', '')} "
                        f"{sig_ann.get('document_type', '')}"
                    ).strip()
                    if st.button(
                        "🔍 Search Knowledge Base",
                        key="asx_rag_search_btn",
                        help="Uses vector similarity to find related historical announcements",
                    ):
                        with st.spinner("Searching vector store for similar announcements…"):
                            try:
                                from agents.knowledge_query_agent import search_lancedb
                                rag_df = search_lancedb(
                                    rag_query_asx,
                                    collections=["asx_announcements"],
                                    k=5,
                                )
                                if rag_df is not None and not rag_df.empty:
                                    st.success(f"Found {len(rag_df)} similar past announcement(s):")
                                    for _, row in rag_df.iterrows():
                                        title   = str(row.get("title",    row.get("headline", "Untitled")))
                                        snippet = str(row.get("text",     row.get("content",  "")))[:300]
                                        source  = str(row.get("source",   row.get("url", "")))
                                        date    = str(row.get("date",     row.get("document_date", "")))[:10]
                                        with st.container(border=True):
                                            st.markdown(f"**{title}** — {date}")
                                            if snippet:
                                                st.caption(snippet + "…")
                                            if source:
                                                st.caption(f"Source: {source}")
                                else:
                                    st.info(
                                        "No similar announcements found in the knowledge base. "
                                        "Ingest ASX announcements via the RAG tab to build the index."
                                    )
                            except Exception as exc:
                                st.warning(
                                    f"Knowledge base search unavailable: {exc}. "
                                    "Ensure LanceDB is configured and the asx_announcements "
                                    "collection has been ingested."
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

        # ════════════════════════════════════════════════════════════════════
        # US PATH — SEC EDGAR
        # ════════════════════════════════════════════════════════════════════
        if _is_us:
            st.subheader("🇺🇸 US SEC EDGAR Filings")
            st.caption(
                "Fetch and interpret SEC filings (8-K, 10-Q, 10-K) for NYSE / NASDAQ stocks. "
                "Uses the free SEC EDGAR API — no API key needed. "
                "Set **SEC_CONTACT_EMAIL** env var to comply with SEC User-Agent policy."
            )
            st.error(
                "⚠️ **NOT FINANCIAL ADVICE** — AI-generated analysis for informational "
                "and educational purposes only.",
                icon="⚠️",
            )

            # ── US Controls ─────────────────────────────────────────────────
            us_col1, us_col2, us_col3 = st.columns([2, 2, 1])
            with us_col1:
                us_ticker = st.text_input(
                    "US Ticker",
                    placeholder="e.g. AAPL, NVDA, MSFT",
                    key="us_ticker_input",
                    help="NYSE / NASDAQ ticker symbol (no suffix needed).",
                ).strip().upper()

            with us_col2:
                us_form_types = st.multiselect(
                    "Form types",
                    options=["8-K", "10-Q", "10-K", "DEF 14A", "S-1"],
                    default=["8-K", "10-Q", "10-K"],
                    key="us_form_types",
                )

            with us_col3:
                us_count = st.number_input(
                    "Max filings", min_value=5, max_value=50, value=20, key="us_count"
                )

            us_fetch_btn = st.button("🔍 Fetch Filings", key="us_fetch")

            if us_fetch_btn and us_ticker:
                from agents.sec_filing_agent import fetch_sec_filings as _us_fetch
                with st.spinner(f"Fetching SEC filings for {us_ticker}…"):
                    try:
                        filings = _us_fetch(
                            us_ticker,
                            count=us_count,
                            form_types=tuple(us_form_types or ["8-K", "10-Q", "10-K"]),
                        )
                        st.session_state.us_filings        = filings
                        st.session_state.us_interpretation = ""
                        st.session_state.us_chat_history   = []
                        st.session_state.us_selected_idx   = 0
                        st.session_state.us_signal         = ""
                        if filings:
                            st.success(f"✅ {len(filings)} filing(s) found for {us_ticker}")
                        else:
                            st.warning(
                                f"No filings found for {us_ticker} with the selected form types. "
                                "Try including 8-K or broadening form type selection."
                            )
                    except Exception as exc:
                        st.error(f"❌ {exc}")

            # ── Filings table ────────────────────────────────────────────────
            if st.session_state.us_filings:
                us_filings = st.session_state.us_filings

                us_df = pd.DataFrame([{
                    "Ticker":    f["ticker"],
                    "Form":      f["form_type"],
                    "Filed":     f["filed_date"],
                    "Headline":  f["headline"],
                    "Sensitive": "🔴" if f["market_sensitive"] else "",
                } for f in us_filings])
                st.dataframe(us_df, use_container_width=True, hide_index=True)

                st.divider()

                us_options = [
                    f"{f['form_type']} | {f['filed_date']} | {f['headline'][:65]}"
                    for f in us_filings
                ]
                us_label = st.selectbox(
                    "Select filing to act on",
                    us_options,
                    index=min(st.session_state.us_selected_idx, len(us_options) - 1),
                    key="us_select",
                )
                us_idx     = us_options.index(us_label)
                us_filing  = us_filings[us_idx]

                # ── Action buttons ───────────────────────────────────────────
                us_a1, us_a2, us_a3 = st.columns(3)

                with us_a1:
                    if st.button("⚡ Interpret Filing", key="us_interpret_btn"):
                        from agents.sec_filing_agent import (
                            download_filing_text,
                            interpret_filing,
                        )
                        us_url = us_filing.get("url", "")
                        with st.spinner("Downloading filing and interpreting…"):
                            try:
                                if us_url not in st.session_state.us_text_cache:
                                    text = download_filing_text(
                                        us_filing["cik"],
                                        us_filing["accession"],
                                        us_filing["primary_doc"],
                                    )
                                    st.session_state.us_text_cache[us_url] = text
                                else:
                                    text = st.session_state.us_text_cache[us_url]

                                interp = interpret_filing(us_filing, text, use_grounding=True)
                                st.session_state.us_interpretation = interp
                                st.session_state.us_chat_history   = []
                                st.session_state.us_selected_idx   = us_idx
                            except Exception as exc:
                                st.error(f"❌ {exc}")

                with us_a2:
                    if st.button("🎯 Trading Signal", key="us_signal_btn",
                                 help="AI signal: buy/sell/hold + price targets. NOT financial advice."):
                        from agents.sec_filing_agent import (
                            download_filing_text,
                            generate_us_trading_signal,
                        )
                        from agents.technical_analysis import fetch_all_timeframes
                        us_url = us_filing.get("url", "")
                        with st.spinner("Fetching live price data + generating US trading signal… (~20–40 s)"):
                            try:
                                if us_url not in st.session_state.us_text_cache:
                                    text = download_filing_text(
                                        us_filing["cik"],
                                        us_filing["accession"],
                                        us_filing["primary_doc"],
                                    )
                                    st.session_state.us_text_cache[us_url] = text
                                else:
                                    text = st.session_state.us_text_cache[us_url]

                                # US stocks: no suffix for NYSE/NASDAQ
                                indicators_by_tf = fetch_all_timeframes(
                                    us_filing["ticker"], suffix="", source="yahoo"
                                )
                                available = [tf for tf, v in indicators_by_tf.items() if v]
                                if available:
                                    st.info(f"📊 Price data loaded for: {', '.join(available)}")
                                else:
                                    st.warning(
                                        "⚠️ Live price data unavailable — signal based on filing text only."
                                    )

                                signal_md = generate_us_trading_signal(
                                    us_filing, text,
                                    indicators_by_tf=indicators_by_tf,
                                    ticker_suffix="",
                                    use_grounding=True,
                                )
                                st.session_state.us_signal     = signal_md
                                st.session_state.us_signal_idx = us_idx
                            except Exception as exc:
                                st.error(f"❌ Signal generation failed: {exc}")

                with us_a3:
                    st.link_button(
                        "📄 View on SEC EDGAR",
                        us_filing.get("url", "https://www.sec.gov/cgi-bin/browse-edgar"),
                        help="Open the original filing on SEC.gov",
                    )

                # ── Trading signal display ───────────────────────────────────
                if st.session_state.us_signal:
                    sig_f = us_filings[st.session_state.us_signal_idx]
                    st.divider()
                    st.subheader(
                        f"🎯 AI Trading Signal — {sig_f['ticker']} "
                        f"({sig_f['form_type']} filed {sig_f['filed_date']})"
                    )
                    st.error(
                        "⚠️ **NOT FINANCIAL ADVICE** — AI-generated signal for informational "
                        "and educational purposes only. Always consult a licensed financial "
                        "adviser before making any investment decision.",
                        icon="⚠️",
                    )
                    st.markdown(st.session_state.us_signal)
                    st.caption(
                        f"Signal from: {sig_f['headline']}  |  "
                        "LLM chain: Gemini (grounded) → GPT reviewer"
                    )

                    # ── RAG cross-search: find similar past SEC filings ───────
                    with st.expander("🔍 Find Similar Past Filings in Knowledge Base", expanded=False):
                        st.caption(
                            "Semantic search across ingested documents in the LanceDB "
                            "vector store — find historical filings or announcements "
                            "with similar content to this SEC filing."
                        )
                        rag_query_us = (
                            f"{sig_f.get('ticker', '')} "
                            f"{sig_f.get('form_type', '')} "
                            f"{sig_f.get('headline', '')}"
                        ).strip()
                        if st.button(
                            "🔍 Search Knowledge Base",
                            key="us_rag_search_btn",
                            help="Vector similarity search across ingested documents",
                        ):
                            with st.spinner("Searching vector store for similar filings…"):
                                try:
                                    from agents.knowledge_query_agent import search_lancedb
                                    rag_df_us = search_lancedb(
                                        rag_query_us,
                                        collections=["asx_announcements", "documents", "pdfs"],
                                        k=5,
                                    )
                                    if rag_df_us is not None and not rag_df_us.empty:
                                        st.success(f"Found {len(rag_df_us)} similar document(s):")
                                        for _, row in rag_df_us.iterrows():
                                            title   = str(row.get("title",    row.get("headline", "Untitled")))
                                            snippet = str(row.get("text",     row.get("content",  "")))[:300]
                                            source  = str(row.get("source",   row.get("url", "")))
                                            coll    = str(row.get("collection", ""))
                                            with st.container(border=True):
                                                st.markdown(f"**{title}**" + (f" [{coll}]" if coll else ""))
                                                if snippet:
                                                    st.caption(snippet + "…")
                                                if source:
                                                    st.caption(f"Source: {source}")
                                    else:
                                        st.info(
                                            "No similar filings found in the knowledge base. "
                                            "Ingest SEC filings or ASX announcements via the "
                                            "RAG tab to build the index."
                                        )
                                except Exception as exc:
                                    st.warning(
                                        f"Knowledge base search unavailable: {exc}. "
                                        "Ensure LanceDB is configured and documents have been ingested."
                                    )

                # ── Interpretation display ───────────────────────────────────
                if st.session_state.us_interpretation:
                    st.divider()
                    ref_f = us_filings[st.session_state.us_selected_idx]
                    st.subheader(
                        f"📋 {ref_f['ticker']} — {ref_f['form_type']} ({ref_f['filed_date']})"
                    )
                    st.markdown(st.session_state.us_interpretation)

                    st.divider()
                    st.subheader("💬 Ask Follow-up Questions")

                    # Chat history
                    for msg in st.session_state.us_chat_history:
                        role_icon = "🧑" if msg["role"] == "user" else "🤖"
                        st.markdown(f"**{role_icon}** {msg['content']}")

                    with st.form("us_chat_form", clear_on_submit=True):
                        us_question = st.text_input(
                            "Your question",
                            placeholder="e.g. What was revenue growth YoY? Any guidance changes?",
                            key="us_question_input",
                        )
                        us_submit = st.form_submit_button("Ask")

                    if us_submit and us_question.strip():
                        from agents.sec_filing_agent import (
                            download_filing_text,
                            answer_filing_question,
                        )
                        us_url = ref_f.get("url", "")
                        if us_url not in st.session_state.us_text_cache:
                            filing_text = download_filing_text(
                                ref_f["cik"], ref_f["accession"], ref_f["primary_doc"]
                            )
                            st.session_state.us_text_cache[us_url] = filing_text
                        else:
                            filing_text = st.session_state.us_text_cache[us_url]

                        st.session_state.us_chat_history.append(
                            {"role": "user", "content": us_question}
                        )
                        with st.spinner("Thinking…"):
                            try:
                                answer = answer_filing_question(
                                    ref_f,
                                    filing_text,
                                    us_question,
                                    chat_history=st.session_state.us_chat_history[:-1],
                                )
                            except Exception as exc:
                                answer = f"⚠️ Error: {exc}"
                        st.markdown(answer)
                        st.session_state.us_chat_history.append(
                            {"role": "assistant", "content": answer}
                        )

    # ── Tab 9: Technical Analysis (Single Stock + Watchlist Scanner) ─────────────
    with tab9:
        st.header("📊 Technical Analysis")
        st.error(
            "⚠️ **NOT FINANCIAL ADVICE** — AI-generated analysis for informational "
            "and educational purposes only. Always consult a licensed financial adviser "
            "before making any investment decision.",
            icon="⚠️",
        )

        # ── Mode selector ─────────────────────────────────────────────────────
        ta_mode = st.radio(
            "Mode",
            ["🔍 Single Stock", "📋 Watchlist Scanner"],
            horizontal=True,
            key="ta_mode",
        )

        # ── Shared: exchange suffix selector ─────────────────────────────────
        suffix_options = {
            "ASX (.AX)":       ".AX",
            "NYSE / NASDAQ":   "",
            "London (.L)":     ".L",
            "Toronto (.TO)":   ".TO",
            "Hong Kong (.HK)": ".HK",
            "Custom…":         "__custom__",
        }
        tf_labels = {"5m": "5-Minute", "30m": "30-Minute", "1h": "1-Hour", "1d": "Daily"}

        # ══════════════════════════════════════════════════════════════════════
        # SINGLE STOCK MODE
        # ══════════════════════════════════════════════════════════════════════
        if ta_mode == "🔍 Single Stock":
            st.caption(
                "Live multi-timeframe price analysis powered by Yahoo Finance + "
                "optional sector/macro context enrichment. Works without any announcement."
            )

            col_ticker, col_suffix, col_btn = st.columns([3, 2, 1])

            with col_ticker:
                ta_ticker = st.text_input(
                    "Ticker symbol",
                    value="BHP",
                    placeholder="e.g. BHP, CBA, AAPL, BP",
                    key="ta_ticker",
                    help="Enter a bare ticker. The exchange suffix is set separately.",
                ).strip().upper()

            with col_suffix:
                suffix_choice = st.selectbox(
                    "Exchange",
                    list(suffix_options.keys()),
                    index=0,
                    key="ta_suffix_choice",
                )
                if suffix_choice == "Custom…":
                    ta_suffix = st.text_input(
                        "Custom suffix (e.g. .SI, .DE)",
                        value=".AX",
                        key="ta_suffix_custom",
                    ).strip()
                else:
                    ta_suffix = suffix_options[suffix_choice]

            with col_btn:
                st.write("")
                st.write("")
                analyse_btn = st.button(
                    "🔍 Analyse", key="ta_analyse_btn", use_container_width=True
                )

            ta_question = st.text_input(
                "Optional: specific question for the AI signal",
                value="",
                placeholder="e.g. Is a breakout forming? What are the key support levels?",
                key="ta_question",
            ).strip() or None

            ta_timeframes = st.multiselect(
                "Timeframes to include",
                options=list(tf_labels.keys()),
                default=["5m", "30m", "1h", "1d"],
                format_func=lambda x: tf_labels[x],
                key="ta_timeframes",
            )
            if not ta_timeframes:
                ta_timeframes = ["1d"]

            # Enhancement toggles
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                use_macro = st.checkbox(
                    "🌐 Include Sector & Macro Context",
                    value=False,
                    key="ta_use_macro",
                    help=(
                        "Fetches sector ETF performance, commodity prices, and FX rates "
                        "relevant to this stock and injects them into the LLM prompt. "
                        "Adds ~5 seconds for data fetch."
                    ),
                )
            with col_opt2:
                use_vision = st.checkbox(
                    "📷 Chart Vision Analysis",
                    value=False,
                    key="ta_use_vision",
                    help=(
                        "Renders a 3-panel technical chart (Price+BB+EMAs / RSI / MACD) "
                        "and sends it to Gemini Vision for visual pattern recognition. "
                        "Identifies chart patterns (H&S, wedges, triangles) and divergences."
                    ),
                )

            # ── Run analysis ─────────────────────────────────────────────────
            if analyse_btn and ta_ticker:
                from agents.technical_analysis import (
                    fetch_all_timeframes,
                    fetch_ohlcv,
                    build_technical_context,
                    generate_technical_signal,
                )

                yf_symbol = f"{ta_ticker}{ta_suffix}"

                with st.spinner(
                    f"Fetching price data for {yf_symbol} across "
                    f"{len(ta_timeframes)} timeframe(s)…"
                ):
                    indicators_by_tf = fetch_all_timeframes(
                        ta_ticker,
                        suffix=ta_suffix,
                        timeframes=tuple(ta_timeframes),
                    )

                available = [tf for tf, v in indicators_by_tf.items() if v is not None]

                if not available:
                    st.warning(
                        f"⚠️ No price data found for **{yf_symbol}**. "
                        "Check the ticker symbol and exchange suffix. "
                        "Yahoo Finance may be rate-limiting — wait 30 s and retry."
                    )
                    for k in ("ta_indicators", "ta_context", "ta_signal",
                              "ta_chart_bytes", "ta_vision", "ta_macro_ctx"):
                        st.session_state[k] = None
                    st.session_state["ta_symbol"] = yf_symbol
                else:
                    st.success(f"✅ Price data loaded for: **{', '.join(available)}**")

                    ctx = build_technical_context(
                        ta_ticker, indicators_by_tf, suffix=ta_suffix
                    )
                    st.session_state["ta_indicators"] = indicators_by_tf
                    st.session_state["ta_context"]    = ctx
                    st.session_state["ta_symbol"]     = yf_symbol
                    st.session_state["ta_suffix"]     = ta_suffix

                    # Optional: macro/sector context
                    macro_ctx = ""
                    if use_macro:
                        with st.spinner("Fetching sector/macro context…"):
                            try:
                                from agents.market_context import fetch_sector_context
                                macro_ctx = fetch_sector_context(ta_ticker, ta_suffix)
                                st.session_state["ta_macro_ctx"] = macro_ctx
                                with st.expander("🌐 Sector & Macro Context", expanded=False):
                                    st.code(macro_ctx, language=None)
                            except Exception as exc:
                                st.warning(f"Macro context unavailable: {exc}")
                                st.session_state["ta_macro_ctx"] = ""
                    else:
                        st.session_state["ta_macro_ctx"] = ""

                    # Optional: chart rendering for Vision
                    chart_bytes = None
                    if use_vision:
                        with st.spinner("Rendering chart for Vision analysis…"):
                            try:
                                from agents.technical_analysis import fetch_ohlcv
                                from agents.chart_vision import render_chart
                                chart_df = fetch_ohlcv(ta_ticker, "1d", ta_suffix)
                                if chart_df is not None:
                                    ind_1d = indicators_by_tf.get("1d")
                                    chart_bytes = render_chart(
                                        ta_ticker, chart_df, ind_1d,
                                        suffix=ta_suffix, timeframe="1d",
                                    )
                                    st.session_state["ta_chart_bytes"] = chart_bytes
                                else:
                                    st.warning("Could not fetch 1d data for chart rendering.")
                                    st.session_state["ta_chart_bytes"] = None
                            except Exception as exc:
                                st.warning(f"Chart render failed: {exc}")
                                st.session_state["ta_chart_bytes"] = None

                    with st.spinner("Generating AI technical signal… (~20–40 s)"):
                        signal_md = generate_technical_signal(
                            ta_ticker,
                            suffix=ta_suffix,
                            question=ta_question,
                            timeframes=tuple(ta_timeframes),
                            indicators_by_tf=indicators_by_tf,
                            macro_context=macro_ctx,
                        )
                    st.session_state["ta_signal"]  = signal_md
                    st.session_state["ta_vision"]  = None  # reset vision until button pressed

            # ── Display: indicator snapshot ───────────────────────────────────
            if st.session_state.get("ta_context"):
                st.divider()
                sym = st.session_state.get("ta_symbol", "")
                st.subheader(f"📈 Indicator Summary — {sym}")
                with st.expander("Raw indicator data (all timeframes)", expanded=False):
                    st.code(st.session_state["ta_context"], language=None)

                inds = st.session_state.get("ta_indicators", {})
                snap_rows = []
                for tf in ["5m", "30m", "1h", "1d"]:
                    ind = inds.get(tf)
                    if ind is None:
                        continue
                    snap_rows.append({
                        "Timeframe": tf_labels.get(tf, tf),
                        "Price":     f"${ind['current_price']:.4f}",
                        "Change":    (
                            f"+{ind['change_pct']}%"
                            if ind.get("change_pct") is not None and ind["change_pct"] >= 0
                            else (f"{ind['change_pct']}%" if ind.get("change_pct") is not None else "N/A")
                        ),
                        "Trend":     ind.get("trend", "N/A"),
                        "RSI(14)":   (
                            f"{ind['rsi']} ({ind['rsi_label']})"
                            if ind.get("rsi") else "N/A"
                        ),
                        "MACD":      ind.get("macd_label", "N/A"),
                        "BB%":       (
                            f"{ind['bb_pct']:.2f} — {ind['bb_label']}"
                            if ind.get("bb_pct") is not None else "N/A"
                        ),
                        "Vol Ratio": f"{ind['vol_ratio']}×" if ind.get("vol_ratio") else "N/A",
                    })

                if snap_rows:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(snap_rows).set_index("Timeframe"),
                        use_container_width=True,
                    )

            # ── Display: AI signal ────────────────────────────────────────────
            if st.session_state.get("ta_signal"):
                st.divider()
                sym = st.session_state.get("ta_symbol", "")
                st.subheader(f"🎯 AI Technical Signal — {sym}")
                st.markdown(st.session_state["ta_signal"])
                macro_note = (
                    "  |  🌐 Sector/macro context injected"
                    if st.session_state.get("ta_macro_ctx") else ""
                )
                st.caption(
                    "LLM chain: Gemini (draft + grounding) → GPT reviewer"
                    f"{macro_note}  |  Data: Yahoo Finance"
                )

            # ── Display: Chart Vision analysis ────────────────────────────────
            chart_bytes_cached = st.session_state.get("ta_chart_bytes")
            if chart_bytes_cached:
                st.divider()
                st.subheader("📷 Chart Vision Analysis")
                col_chart, col_vis = st.columns([2, 1])
                with col_chart:
                    st.image(chart_bytes_cached, caption="Technical Chart (1D)", use_container_width=True)
                with col_vis:
                    if st.button(
                        "🔭 Analyse Chart with Gemini Vision",
                        key="ta_vision_btn",
                        use_container_width=True,
                        help="Sends the chart to Gemini Vision for visual pattern recognition",
                    ):
                        with st.spinner("Gemini Vision analysing chart patterns…"):
                            try:
                                from agents.chart_vision import analyse_chart_with_gemini
                                sym_state = st.session_state.get("ta_symbol", "")
                                suf_state = st.session_state.get("ta_suffix", ".AX")
                                ind_1d    = (
                                    st.session_state.get("ta_indicators", {}).get("1d")
                                )
                                vision_md = analyse_chart_with_gemini(
                                    sym_state.replace(suf_state, ""),
                                    chart_bytes_cached,
                                    indicators=ind_1d,
                                    suffix=suf_state,
                                    timeframe="1d",
                                )
                                st.session_state["ta_vision"] = vision_md
                            except Exception as exc:
                                st.error(f"Vision analysis failed: {exc}")

                if st.session_state.get("ta_vision"):
                    st.markdown(st.session_state["ta_vision"])

        # ══════════════════════════════════════════════════════════════════════
        # WATCHLIST SCANNER MODE
        # ══════════════════════════════════════════════════════════════════════
        else:
            st.caption(
                "Scan multiple tickers in parallel. "
                "Fetches live price data for each ticker and builds a summary table. "
                "Select any ticker from the results to generate a full AI signal."
            )

            wl_col1, wl_col2 = st.columns([3, 1])

            with wl_col1:
                wl_raw = st.text_area(
                    "Tickers (one per line or comma-separated)",
                    value="BHP\nCBA\nCSL\nWBC\nRIO",
                    height=120,
                    key="wl_tickers_input",
                    placeholder="BHP\nRIO\nCBA\nAAPL\nNVDA",
                )

            with wl_col2:
                suffix_choice_wl = st.selectbox(
                    "Exchange",
                    list(suffix_options.keys()),
                    index=0,
                    key="wl_suffix_choice",
                )
                if suffix_choice_wl == "Custom…":
                    wl_suffix = st.text_input(
                        "Custom suffix", value=".AX", key="wl_suffix_custom"
                    ).strip()
                else:
                    wl_suffix = suffix_options[suffix_choice_wl]

                scan_btn = st.button(
                    "🚀 Scan Watchlist",
                    key="wl_scan_btn",
                    use_container_width=True,
                    type="primary",
                )

            # Parse tickers
            wl_tickers = []
            for raw in wl_raw.replace(",", "\n").splitlines():
                t = raw.strip().upper()
                if t:
                    wl_tickers.append(t)
            wl_tickers = list(dict.fromkeys(wl_tickers))  # deduplicate, preserve order

            if scan_btn and wl_tickers:
                from agents.technical_analysis import fetch_all_timeframes
                import concurrent.futures

                def _scan_one(ticker: str):
                    """Fetch 1d indicators for a single ticker — designed for ThreadPool."""
                    try:
                        inds = fetch_all_timeframes(
                            ticker,
                            suffix=wl_suffix,
                            timeframes=("1d",),
                        )
                        return ticker, inds.get("1d")
                    except Exception as exc:
                        return ticker, None

                progress = st.progress(0, text=f"Scanning {len(wl_tickers)} tickers…")
                results = {}

                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                    futures = {pool.submit(_scan_one, t): t for t in wl_tickers}
                    done = 0
                    for fut in concurrent.futures.as_completed(futures):
                        ticker, ind = fut.result()
                        results[ticker] = ind
                        done += 1
                        progress.progress(
                            done / len(wl_tickers),
                            text=f"Scanned {done}/{len(wl_tickers)}: {ticker}",
                        )

                progress.empty()
                st.session_state["wl_results"] = results
                st.session_state["wl_suffix"]  = wl_suffix

            # ── Watchlist summary table ───────────────────────────────────────
            wl_results = st.session_state.get("wl_results", {})
            if wl_results:
                import pandas as pd

                rows = []
                for ticker, ind in wl_results.items():
                    if ind is None:
                        rows.append({
                            "Ticker":    ticker,
                            "Price":     "N/A",
                            "1d %":      "N/A",
                            "Trend":     "—",
                            "RSI(14)":   "N/A",
                            "MACD":      "—",
                            "BB%":       "N/A",
                            "Vol Ratio": "N/A",
                            "⚡ Signal":  "NO DATA",
                        })
                        continue

                    p   = ind.get("current_price", 0)
                    chg = ind.get("change_pct")
                    chg_str = (
                        f"+{chg}%" if chg is not None and chg >= 0
                        else (f"{chg}%" if chg is not None else "N/A")
                    )

                    rsi = ind.get("rsi")
                    rsi_lbl = ind.get("rsi_label", "")
                    macd = ind.get("macd_label", "—")
                    bp   = ind.get("bb_pct")

                    # Compute a quick composite signal (no LLM — deterministic)
                    bullish = 0
                    bearish = 0
                    if rsi and rsi < 40: bullish += 1
                    if rsi and rsi > 65: bearish += 1
                    if macd == "BULLISH": bullish += 1
                    if macd == "BEARISH": bearish += 1
                    trend = ind.get("trend", "")
                    if trend == "UPTREND":   bullish += 1
                    if trend == "DOWNTREND": bearish += 1
                    if bp is not None and bp < 0.20: bullish += 1
                    if bp is not None and bp > 0.80: bearish += 1

                    if   bullish >= 3: quick_sig = "🟢 STRONG BUY"
                    elif bullish == 2: quick_sig = "🟩 BUY"
                    elif bearish >= 3: quick_sig = "🔴 STRONG SELL"
                    elif bearish == 2: quick_sig = "🟥 SELL"
                    else:              quick_sig = "⬜ HOLD/NEUTRAL"

                    rows.append({
                        "Ticker":    ticker,
                        "Price":     f"${p:.4f}",
                        "1d %":      chg_str,
                        "Trend":     trend or "—",
                        "RSI(14)":   f"{rsi} ({rsi_lbl})" if rsi else "N/A",
                        "MACD":      macd or "—",
                        "BB%":       f"{bp:.2f} — {ind.get('bb_label','')}" if bp is not None else "N/A",
                        "Vol Ratio": f"{ind['vol_ratio']}×" if ind.get("vol_ratio") else "N/A",
                        "⚡ Signal":  quick_sig,
                    })

                st.divider()
                wl_suf = st.session_state.get("wl_suffix", "")
                st.subheader(f"📋 Watchlist Scan Results — {len(rows)} tickers ({wl_suf or 'US'})")
                df_wl = pd.DataFrame(rows).set_index("Ticker")
                st.dataframe(df_wl, use_container_width=True)

                # ── Full AI signal for a selected ticker ──────────────────────
                st.divider()
                st.subheader("🎯 Generate Full AI Signal")
                available_tickers = [
                    t for t, ind in wl_results.items() if ind is not None
                ]
                if available_tickers:
                    sel_col, btn_col = st.columns([3, 1])
                    with sel_col:
                        wl_sel_ticker = st.selectbox(
                            "Select ticker for full signal",
                            available_tickers,
                            key="wl_sel_ticker",
                        )
                    with btn_col:
                        st.write("")
                        wl_signal_btn = st.button(
                            "Generate Signal",
                            key="wl_signal_btn",
                            use_container_width=True,
                            type="primary",
                        )

                    if wl_signal_btn and wl_sel_ticker:
                        from agents.technical_analysis import (
                            fetch_all_timeframes,
                            build_technical_context,
                            generate_technical_signal,
                        )
                        wl_suf_now = st.session_state.get("wl_suffix", ".AX")

                        with st.spinner(
                            f"Fetching all timeframes + generating signal for "
                            f"{wl_sel_ticker}{wl_suf_now}…"
                        ):
                            full_inds = fetch_all_timeframes(
                                wl_sel_ticker,
                                suffix=wl_suf_now,
                                timeframes=("5m", "30m", "1h", "1d"),
                            )
                            wl_signal_md = generate_technical_signal(
                                wl_sel_ticker,
                                suffix=wl_suf_now,
                                indicators_by_tf=full_inds,
                            )

                        st.session_state["wl_full_signal"]        = wl_signal_md
                        st.session_state["wl_full_signal_ticker"] = (
                            f"{wl_sel_ticker}{wl_suf_now}"
                        )

                    if st.session_state.get("wl_full_signal"):
                        st.error(
                            "⚠️ **NOT FINANCIAL ADVICE** — AI-generated signal for "
                            "informational and educational purposes only.",
                            icon="⚠️",
                        )
                        st.subheader(
                            f"🎯 {st.session_state.get('wl_full_signal_ticker', '')} — Full Signal"
                        )
                        st.markdown(st.session_state["wl_full_signal"])
                else:
                    st.info("No tickers with valid price data available for signal generation.")

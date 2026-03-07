"""
tests/test_traceability.py

Critical-path tests for the Datap.ai Trace Ledger.

Tests cover:
  - TraceEvent creation and serialisation
  - Redaction helpers (secrets, PII, SQL)
  - SQLite backend (in-memory) — append, fetch, search
  - TraceLedger convenience methods
  - Multi-tenant isolation (no cross-tenant bleed)
  - Trace replay and timeline reconstruction
  - Append-only invariant (no mutations)
  - NullBackend no-op behaviour

Run with:
    pytest tests/test_traceability.py -v
"""

from __future__ import annotations

import pytest
import threading

from traceability.models import (
    EventType,
    ActorType,
    TraceStatus,
    IdentityContext,
    TraceEvent,
    _sha256,
)
from traceability.redaction import (
    mask_secrets,
    summarise,
    hash_payload,
    safe_sql_summary,
    redact_dict,
)
from traceability.backends import NullTraceLedgerBackend
from traceability.backends.sqlite_backend import SQLiteTraceLedgerBackend
from traceability.ledger import TraceLedger, reset_ledger
from traceability.replay import TraceReplayer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def alice_identity():
    return IdentityContext(
        tenant_id    = "acme",
        workspace_id = "analytics",
        user_id      = "alice",
        session_id   = "sess-alice-001",
    )

@pytest.fixture
def bob_identity():
    """Different tenant — must never see Alice's events."""
    return IdentityContext(
        tenant_id    = "rival-corp",
        workspace_id = "finance",
        user_id      = "bob",
        session_id   = "sess-bob-001",
    )

@pytest.fixture
def sqlite_backend(tmp_path):
    db = SQLiteTraceLedgerBackend(db_path=str(tmp_path / "test_traces.db"))
    db.initialise()
    yield db
    db.close()

@pytest.fixture
def ledger(sqlite_backend):
    return TraceLedger(sqlite_backend)


# ── Model tests ───────────────────────────────────────────────────────────────

class TestTraceEventModel:

    def test_new_generates_uuid(self, alice_identity):
        event = TraceEvent.new(
            identity   = alice_identity,
            event_type = EventType.REQUEST_RECEIVED,
        )
        assert len(event.trace_id) == 36   # UUID4 format
        assert event.trace_id != ""

    def test_new_sets_identity(self, alice_identity):
        event = TraceEvent.new(
            identity   = alice_identity,
            event_type = EventType.REQUEST_RECEIVED,
        )
        assert event.tenant_id    == "acme"
        assert event.workspace_id == "analytics"
        assert event.user_id      == "alice"
        assert event.session_id   == "sess-alice-001"

    def test_sql_is_hashed_not_stored(self, alice_identity):
        sql = "SELECT * FROM orders WHERE user_id = 123"
        event = TraceEvent.new(
            identity   = alice_identity,
            event_type = EventType.SQL_GENERATED,
            sql_text   = sql,
        )
        assert event.sql_hash == _sha256(sql)
        # The raw SQL must NOT be stored anywhere in the event
        d = event.to_dict()
        assert sql not in str(d.values())

    def test_prompt_is_hashed_not_stored(self, alice_identity):
        prompt = "You are a SQL expert. Generate SQL for: revenue by region"
        event = TraceEvent.new(
            identity    = alice_identity,
            event_type  = EventType.MODEL_INVOKED,
            prompt_text = prompt,
        )
        assert event.prompt_hash == _sha256(prompt)
        d = event.to_dict()
        assert prompt not in str(d.values())

    def test_to_dict_has_all_required_fields(self, alice_identity):
        event = TraceEvent.new(
            identity   = alice_identity,
            event_type = EventType.REQUEST_RECEIVED,
        )
        d = event.to_dict()
        required = [
            "trace_id", "tenant_id", "workspace_id", "user_id",
            "session_id", "request_id", "event_type", "event_timestamp",
            "actor_type", "actor_id", "status",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_identity_from_env(self, monkeypatch):
        monkeypatch.setenv("DATAPAI_TENANT_ID", "env-tenant")
        monkeypatch.setenv("DATAPAI_USER_ID", "env-user")
        identity = IdentityContext.from_env()
        assert identity.tenant_id == "env-tenant"
        assert identity.user_id   == "env-user"


# ── Redaction tests ───────────────────────────────────────────────────────────

class TestRedaction:

    def test_mask_api_key(self):
        text = 'config = {"api_key": "sk-abcdef1234567890"}'
        result = mask_secrets(text)
        assert "sk-abcdef" not in result
        assert "[REDACTED]" in result

    def test_mask_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test.sig"
        result = mask_secrets(text)
        assert "eyJ" not in result

    def test_summarise_truncates(self):
        long_text = "x" * 1000
        result = summarise(long_text, max_len=100)
        assert len(result) <= 101   # 100 + "…"
        assert result.endswith("…")

    def test_summarise_none(self):
        assert summarise(None) is None

    def test_summarise_masks_secrets(self):
        text = "secret: sk-verysecretkey123456"
        result = summarise(text)
        assert "sk-verysecretkey" not in result

    def test_safe_sql_summary_removes_literals(self):
        sql = "SELECT * FROM orders WHERE name = 'Alice' AND amount > 1000"
        result = safe_sql_summary(sql)
        assert "Alice" not in result
        assert "1000" not in (result or "")

    def test_hash_payload(self):
        h1 = hash_payload("hello")
        h2 = hash_payload("hello")
        h3 = hash_payload("world")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64   # SHA-256 hex

    def test_redact_dict_sensitive_keys(self):
        d = {"email": "alice@example.com", "name": "Alice", "score": 42}
        result = redact_dict(d)
        assert result["email"] == "[REDACTED]"
        assert result["name"]  == "[REDACTED]"
        assert result["score"] == 42


# ── SQLite backend tests ──────────────────────────────────────────────────────

class TestSQLiteBackend:

    def test_initialise_creates_table(self, sqlite_backend):
        # If table exists, count should return 0 (not raise)
        count = sqlite_backend.count("acme")
        assert count == 0

    def test_append_and_fetch_by_trace_id(self, sqlite_backend, alice_identity):
        event = TraceEvent.new(
            identity   = alice_identity,
            event_type = EventType.REQUEST_RECEIVED,
        )
        sqlite_backend.append(event.to_dict())
        rows = sqlite_backend.fetch_by_trace_id(event.trace_id)
        assert len(rows) == 1
        assert rows[0]["trace_id"] == event.trace_id

    def test_fetch_by_session(self, sqlite_backend, alice_identity):
        for etype in [EventType.REQUEST_RECEIVED, EventType.SQL_GENERATED, EventType.RESPONSE_RETURNED]:
            event = TraceEvent.new(identity=alice_identity, event_type=etype)
            sqlite_backend.append(event.to_dict())

        rows = sqlite_backend.fetch_by_session("acme", "sess-alice-001")
        assert len(rows) == 3

    def test_search_by_event_type(self, sqlite_backend, alice_identity):
        for etype in [EventType.REQUEST_RECEIVED, EventType.SQL_GENERATED, EventType.SQL_BLOCKED]:
            event = TraceEvent.new(identity=alice_identity, event_type=etype)
            sqlite_backend.append(event.to_dict())

        rows = sqlite_backend.search(
            tenant_id  = "acme",
            event_type = "sql_generated",
        )
        assert len(rows) == 1
        assert rows[0]["event_type"] == "sql_generated"

    def test_search_by_status_blocked(self, sqlite_backend, alice_identity):
        blocked = TraceEvent.new(
            identity   = alice_identity,
            event_type = EventType.SQL_BLOCKED,
            status     = TraceStatus.BLOCKED,
        )
        ok = TraceEvent.new(
            identity   = alice_identity,
            event_type = EventType.SQL_EXECUTED,
            status     = TraceStatus.OK,
        )
        sqlite_backend.append(blocked.to_dict())
        sqlite_backend.append(ok.to_dict())

        rows = sqlite_backend.search(tenant_id="acme", status="blocked")
        assert len(rows) == 1
        assert rows[0]["status"] == "blocked"

    def test_idempotent_append_on_duplicate_trace_id(self, sqlite_backend, alice_identity):
        event = TraceEvent.new(identity=alice_identity, event_type=EventType.REQUEST_RECEIVED)
        sqlite_backend.append(event.to_dict())
        sqlite_backend.append(event.to_dict())   # duplicate — should be silently ignored
        rows = sqlite_backend.fetch_by_trace_id(event.trace_id)
        assert len(rows) == 1   # only one row, not two

    def test_count(self, sqlite_backend, alice_identity, bob_identity):
        for _ in range(3):
            e = TraceEvent.new(identity=alice_identity, event_type=EventType.REQUEST_RECEIVED)
            sqlite_backend.append(e.to_dict())
        e2 = TraceEvent.new(identity=bob_identity, event_type=EventType.REQUEST_RECEIVED)
        sqlite_backend.append(e2.to_dict())

        assert sqlite_backend.count("acme")       == 3
        assert sqlite_backend.count("rival-corp")  == 1


# ── Multi-tenant isolation tests ──────────────────────────────────────────────

class TestMultiTenantIsolation:

    def test_session_fetch_is_tenant_scoped(self, sqlite_backend, alice_identity, bob_identity):
        # Both use the same session_id to test isolation by tenant_id
        shared_session = "sess-shared"
        alice = IdentityContext(
            tenant_id    = alice_identity.tenant_id,
            workspace_id = alice_identity.workspace_id,
            user_id      = alice_identity.user_id,
            session_id   = shared_session,
        )
        bob = IdentityContext(
            tenant_id    = bob_identity.tenant_id,
            workspace_id = bob_identity.workspace_id,
            user_id      = bob_identity.user_id,
            session_id   = shared_session,
        )
        for identity in [alice, bob]:
            e = TraceEvent.new(identity=identity, event_type=EventType.REQUEST_RECEIVED)
            sqlite_backend.append(e.to_dict())

        alice_rows = sqlite_backend.fetch_by_session("acme", shared_session)
        bob_rows   = sqlite_backend.fetch_by_session("rival-corp", shared_session)

        assert all(r["tenant_id"] == "acme"       for r in alice_rows)
        assert all(r["tenant_id"] == "rival-corp"  for r in bob_rows)

    def test_search_cannot_see_other_tenants(self, sqlite_backend, alice_identity, bob_identity):
        for identity in [alice_identity, bob_identity]:
            e = TraceEvent.new(identity=identity, event_type=EventType.SQL_GENERATED)
            sqlite_backend.append(e.to_dict())

        alice_results = sqlite_backend.search(tenant_id="acme")
        assert all(r["tenant_id"] == "acme" for r in alice_results)


# ── TraceLedger tests ─────────────────────────────────────────────────────────

class TestTraceLedger:

    def test_emit_returns_trace_id(self, ledger, alice_identity):
        trace_id = ledger.emit(
            identity   = alice_identity,
            event_type = EventType.REQUEST_RECEIVED,
            input_text = "Show me revenue by region",
        )
        assert len(trace_id) == 36

    def test_emit_request_received(self, ledger, alice_identity):
        trace_id = ledger.emit_request_received(
            identity   = alice_identity,
            request_id = "req-001",
            input_text = "How many orders last week?",
        )
        event = ledger.get_event(trace_id)
        assert event is not None
        assert event["event_type"] == "request_received"
        assert event["user_id"]    == "alice"

    def test_emit_sql_generated_hashes_sql(self, ledger, alice_identity):
        sql = "SELECT COUNT(*) FROM orders WHERE created_at > CURRENT_DATE - 7"
        trace_id = ledger.emit_sql_generated(
            identity        = alice_identity,
            request_id      = "req-002",
            sql_text        = sql,
            model_name      = "claude-sonnet-4-6",
            datasource_name = "snowflake_prod",
        )
        event = ledger.get_event(trace_id)
        assert event["sql_hash"] == _sha256(sql)
        assert event["event_type"] == "sql_generated"

    def test_emit_sql_blocked(self, ledger, alice_identity):
        trace_id = ledger.emit_sql_validated(
            identity       = alice_identity,
            request_id     = "req-003",
            sql_text       = "DROP TABLE orders",
            policy_result  = "BLOCKED",
            status         = TraceStatus.BLOCKED,
            error_message  = "DDL statements not allowed",
        )
        event = ledger.get_event(trace_id)
        assert event["event_type"] == "sql_blocked"
        assert event["status"]     == "blocked"

    def test_emit_policy_check_passed(self, ledger, alice_identity):
        trace_id = ledger.emit_policy_check(
            identity      = alice_identity,
            request_id    = "req-004",
            policy_result = "PASSED",
            passed        = True,
        )
        event = ledger.get_event(trace_id)
        assert event["event_type"] == "policy_check_passed"
        assert event["status"]     == "ok"

    def test_emit_policy_check_failed(self, ledger, alice_identity):
        trace_id = ledger.emit_policy_check(
            identity      = alice_identity,
            request_id    = "req-005",
            policy_result = "BLOCKED",
            passed        = False,
            reason        = "User does not have access to PII schema",
        )
        event = ledger.get_event(trace_id)
        assert event["event_type"] == "policy_check_failed"
        assert event["status"]     == "blocked"

    def test_emit_never_raises(self, ledger, alice_identity):
        """Ledger must not propagate errors to the caller."""
        # Simulate a backend failure by closing the backend first
        ledger._backend.close()
        # Should not raise
        trace_id = ledger.emit(
            identity   = alice_identity,
            event_type = EventType.REQUEST_RECEIVED,
        )
        assert isinstance(trace_id, str)

    def test_get_session_timeline_ordered(self, ledger, alice_identity):
        request_id = "req-timeline"
        for etype in [
            EventType.REQUEST_RECEIVED,
            EventType.MEMORY_RETRIEVED,
            EventType.MODEL_INVOKED,
            EventType.SQL_GENERATED,
            EventType.RESPONSE_RETURNED,
        ]:
            ledger.emit(
                identity   = alice_identity,
                event_type = etype,
                request_id = request_id,
            )

        timeline = ledger.get_session_timeline("acme", "sess-alice-001")
        assert len(timeline) == 5
        timestamps = [e["event_timestamp"] for e in timeline]
        assert timestamps == sorted(timestamps)


# ── Replay tests ──────────────────────────────────────────────────────────────

class TestTraceReplayer:

    def test_get_request_timeline(self, ledger, alice_identity):
        request_id = "req-replay"
        trace_ids = []
        for etype in [
            EventType.REQUEST_RECEIVED,
            EventType.SQL_GENERATED,
            EventType.SQL_VALIDATED,
            EventType.SQL_EXECUTED,
            EventType.RESPONSE_RETURNED,
        ]:
            tid = ledger.emit(
                identity   = alice_identity,
                event_type = etype,
                request_id = request_id,
            )
            trace_ids.append(tid)

        replayer = TraceReplayer(ledger)
        timeline = replayer.get_request_timeline("acme", request_id)
        assert len(timeline) == 5

    def test_get_sql_chain(self, ledger, alice_identity):
        request_id = "req-sql-chain"
        for etype in [
            EventType.REQUEST_RECEIVED,
            EventType.SQL_GENERATED,
            EventType.SQL_VALIDATED,
            EventType.SQL_EXECUTED,
            EventType.RESPONSE_RETURNED,
        ]:
            ledger.emit(
                identity   = alice_identity,
                event_type = etype,
                request_id = request_id,
            )

        replayer = TraceReplayer(ledger)
        timeline = replayer.get_request_timeline("acme", request_id)
        sql_chain = replayer.get_sql_chain(timeline)
        assert len(sql_chain) == 3
        assert all(e.is_sql_event for e in sql_chain)

    def test_format_timeline_not_empty(self, ledger, alice_identity):
        request_id = "req-format"
        ledger.emit(
            identity   = alice_identity,
            event_type = EventType.REQUEST_RECEIVED,
            request_id = request_id,
        )
        replayer = TraceReplayer(ledger)
        timeline = replayer.get_request_timeline("acme", request_id)
        text = replayer.format_timeline(timeline)
        assert "TRACE TIMELINE" in text
        assert "request_received" in text

    def test_to_streamlit_cards(self, ledger, alice_identity):
        request_id = "req-cards"
        ledger.emit(
            identity   = alice_identity,
            event_type = EventType.SQL_BLOCKED,
            request_id = request_id,
            status     = TraceStatus.BLOCKED,
        )
        replayer = TraceReplayer(ledger)
        timeline = replayer.get_request_timeline("acme", request_id)
        cards = replayer.to_streamlit_cards(timeline)
        assert len(cards) == 1
        assert cards[0]["is_error"] is True


# ── Null backend tests ────────────────────────────────────────────────────────

class TestNullBackend:

    def test_null_backend_swallows_all(self, alice_identity):
        backend = NullTraceLedgerBackend()
        backend.initialise()
        event = TraceEvent.new(identity=alice_identity, event_type=EventType.REQUEST_RECEIVED)
        backend.append(event.to_dict())   # no-op
        assert backend.fetch_by_trace_id(event.trace_id) == []
        assert backend.fetch_by_session("acme", "sess") == []
        assert backend.search(tenant_id="acme") == []
        assert backend.count("acme") == 0
        backend.close()


# ── Append-only invariant test ────────────────────────────────────────────────

class TestAppendOnlyInvariant:

    def test_duplicate_trace_id_does_not_mutate(self, sqlite_backend, alice_identity):
        """
        If an event with the same trace_id is appended twice (e.g. on retry),
        the second write MUST be silently ignored — not update the existing row.
        """
        event = TraceEvent.new(
            identity      = alice_identity,
            event_type    = EventType.REQUEST_RECEIVED,
            input_summary = "original summary",
        )
        sqlite_backend.append(event.to_dict())

        # Attempt to overwrite with a different summary
        tampered = event.to_dict()
        tampered["input_summary"] = "TAMPERED"
        sqlite_backend.append(tampered)

        rows = sqlite_backend.fetch_by_trace_id(event.trace_id)
        assert len(rows) == 1
        assert rows[0]["input_summary"] == "original summary"


# ── Thread-safety test ────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_appends(self, sqlite_backend, alice_identity):
        """Multiple threads can append simultaneously without data corruption."""
        errors = []

        def append_events():
            try:
                for _ in range(10):
                    e = TraceEvent.new(
                        identity   = alice_identity,
                        event_type = EventType.REQUEST_RECEIVED,
                    )
                    sqlite_backend.append(e.to_dict())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=append_events) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        count = sqlite_backend.count("acme")
        assert count == 50   # 5 threads × 10 events each

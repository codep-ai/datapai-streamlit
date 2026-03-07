"""
traceability/redaction.py

Redaction and summarisation helpers for the Datap.ai Trace Ledger.

Principles:
- Never store raw secrets, tokens, or credentials.
- Never store raw PII where avoidable.
- Hash large prompts/results to keep a fingerprint without the payload.
- Preserve summaries and opaque references for replay/debug.
- Corrections append new events; never silently mutate history.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

# ── Secret / credential patterns ──────────────────────────────────────────────

_SECRET_PATTERNS: list[re.Pattern] = [
    # Generic API key-like strings
    re.compile(r'(?i)(api[_\-]?key|apikey|secret|token|password|passwd|pwd|bearer|auth)["\s:=]+[A-Za-z0-9+/=_\-]{8,}'),
    # AWS credentials
    re.compile(r'(?i)(AKIA[0-9A-Z]{16})'),                          # AWS access key
    re.compile(r'(?i)(aws[_\-]?secret[_\-]?access[_\-]?key)["\s:=]+[A-Za-z0-9+/]{40}'),
    # Snowflake / DB connection strings
    re.compile(r'(?i)(snowflake://[^\s"\']+)'),
    re.compile(r'(?i)(postgresql://[^\s"\']+)'),
    re.compile(r'(?i)(mysql://[^\s"\']+)'),
    # JWT tokens
    re.compile(r'eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+'),
    # Bearer tokens in headers
    re.compile(r'(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*'),
    # SSH / PEM blocks
    re.compile(r'-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----'),
]

# ── PII heuristics (column names / field names in structured data) ─────────────

_PII_FIELD_NAMES: frozenset[str] = frozenset({
    "ssn", "social_security", "tax_id", "passport", "driver_license",
    "credit_card", "card_number", "cvv", "pin",
    "date_of_birth", "dob", "birth_date",
    "phone", "mobile", "fax",
    "email", "email_address",
    "address", "street", "postcode", "zip_code",
    "name", "first_name", "last_name", "full_name",
    "ip_address", "mac_address",
    "bank_account", "account_number", "routing_number", "iban",
    "medical_record", "diagnosis", "prescription",
})

_REDACTED = "[REDACTED]"
_MAX_SUMMARY_LEN = 500


# ── Public API ────────────────────────────────────────────────────────────────

def mask_secrets(text: str) -> str:
    """
    Replace secret/credential patterns in *text* with [REDACTED].

    Does NOT mutate the original; returns a new string.
    """
    result = text
    for pat in _SECRET_PATTERNS:
        result = pat.sub(_REDACTED, result)
    return result


def summarise(text: Optional[str], max_len: int = _MAX_SUMMARY_LEN) -> Optional[str]:
    """
    Return a safe, truncated summary of *text* suitable for trace storage.

    1. Masks secrets.
    2. Collapses whitespace.
    3. Truncates to max_len and appends "…" if needed.
    """
    if text is None:
        return None
    cleaned = mask_secrets(text)
    cleaned = " ".join(cleaned.split())   # collapse whitespace
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "…"
    return cleaned


def hash_payload(text: Optional[str]) -> Optional[str]:
    """Return SHA-256 hex digest of *text*, or None if text is None/empty."""
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_sql_summary(sql: Optional[str]) -> Optional[str]:
    """
    Return a short structural summary of a SQL string without raw data values.

    Example: "SELECT … FROM orders WHERE … LIMIT 100"
    """
    if not sql:
        return None
    # Strip string literals and numeric literals
    stripped = re.sub(r"'[^']*'", "'?'", sql)          # string literals
    stripped = re.sub(r"\b\d+(\.\d+)?\b", "?", stripped)  # numbers
    stripped = mask_secrets(stripped)
    return summarise(stripped, max_len=300)


def redact_dict(d: dict, sensitive_keys: Optional[frozenset] = None) -> dict:
    """
    Return a copy of *d* with values for sensitive keys replaced by [REDACTED].

    Uses _PII_FIELD_NAMES plus any caller-supplied *sensitive_keys*.
    """
    keys = _PII_FIELD_NAMES | (sensitive_keys or frozenset())
    out = {}
    for k, v in d.items():
        if k.lower() in keys:
            out[k] = _REDACTED
        elif isinstance(v, str):
            out[k] = mask_secrets(v)
        elif isinstance(v, dict):
            out[k] = redact_dict(v, sensitive_keys)
        else:
            out[k] = v
    return out


def safe_repr(obj: Any, max_len: int = 200) -> str:
    """
    Safe, truncated string representation of any object.
    Masks secrets and limits length.
    """
    raw = repr(obj) if not isinstance(obj, str) else obj
    return summarise(raw, max_len=max_len) or ""

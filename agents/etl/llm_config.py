"""
LLM provider configuration and guardrail layers for the AG2 ETL pipeline.

Provider auto-detection priority (override with DATAPAI_LLM_PROVIDER):
  1. bedrock    AWS Bedrock     private / enterprise, supports native Guardrails
  2. anthropic  Anthropic API   paid, best reasoning quality
  3. openai     OpenAI API      paid, broad ecosystem
  4. ollama     Ollama local    fully private, no data leaves the machine

Guardrail layers (applied in this order for every LLM interaction):
  [L1] Prompt sanitization   — strip/redact PII patterns from text sent to any LLM
  [L2] Bedrock Guardrails    — AWS-managed content policy enforced at the API gateway
  [L3] Response validation   — detect PII leakage in LLM outputs and warn/block

Environment variables
─────────────────────────────────────────────────────────────────────────────
General
  DATAPAI_LLM_PROVIDER       Force provider: bedrock | anthropic | openai | ollama
  DATAPAI_SANITIZE_PROMPTS   true* | false   — enable prompt PII sanitization
  DATAPAI_VALIDATE_RESPONSES true* | false   — enable response PII validation

AWS Bedrock
  AWS_REGION                 us-east-1 (default)
  BEDROCK_MODEL_ID           anthropic.claude-3-5-sonnet-20241022-v2:0 (default)
  AWS_ACCESS_KEY_ID          (or IAM role / ~/.aws/credentials)
  AWS_SECRET_ACCESS_KEY
  BEDROCK_GUARDRAIL_ID       Your Guardrail identifier  ← leave blank to skip
  BEDROCK_GUARDRAIL_VERSION  DRAFT | 1 | 2 …  (default: DRAFT)

Anthropic
  ANTHROPIC_API_KEY
  ANTHROPIC_MODEL            claude-3-5-sonnet-20241022 (default)

OpenAI
  OPENAI_API_KEY
  OPENAI_MODEL               gpt-4o (default)

Ollama
  OLLAMA_BASE_URL            http://localhost:11434 (default)
  OLLAMA_MODEL               llama3.1 (default)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex patterns used for prompt sanitization + response validation ─────────
# Same family as compliance_tools.py but applied to raw *text* (not tabular data)
_TEXT_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"), "[CARD_NUMBER]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"), "[IBAN]"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Provider detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_provider() -> str:
    """
    Auto-detect which LLM provider to use.

    Returns one of: 'bedrock' | 'anthropic' | 'openai' | 'ollama'
    Can be overridden by setting DATAPAI_LLM_PROVIDER.
    """
    forced = os.getenv("DATAPAI_LLM_PROVIDER", "").lower().strip()
    if forced in ("bedrock", "anthropic", "openai", "ollama"):
        logger.info("LLM provider forced via DATAPAI_LLM_PROVIDER=%s", forced)
        return forced

    # Auto-detect in priority order
    if os.getenv("BEDROCK_MODEL_ID") or (
        os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
    ):
        return "bedrock"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL"):
        return "ollama"

    raise ValueError(
        "No LLM provider configured. Set one of:\n"
        "  BEDROCK_MODEL_ID / AWS credentials    → AWS Bedrock (private)\n"
        "  ANTHROPIC_API_KEY                     → Anthropic Claude (paid)\n"
        "  OPENAI_API_KEY                        → OpenAI (paid)\n"
        "  OLLAMA_BASE_URL or OLLAMA_MODEL       → Ollama local (private)\n"
        "Or set DATAPAI_LLM_PROVIDER=<provider> to force a choice."
    )


def provider_label() -> dict:
    """Return a human-readable summary of the active LLM configuration."""
    provider = detect_provider()
    labels = {
        "bedrock": {
            "provider": "AWS Bedrock",
            "model": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "data_residency": "private (AWS VPC)",
            "guardrails": "AWS Bedrock Guardrails" if os.getenv("BEDROCK_GUARDRAIL_ID") else "none configured",
            "cost": "pay-per-token (AWS pricing)",
        },
        "anthropic": {
            "provider": "Anthropic",
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "data_residency": "Anthropic servers (paid API)",
            "guardrails": "prompt sanitization + response validation",
            "cost": "pay-per-token (Anthropic pricing)",
        },
        "openai": {
            "provider": "OpenAI",
            "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "data_residency": "OpenAI servers (paid API)",
            "guardrails": "prompt sanitization + response validation",
            "cost": "pay-per-token (OpenAI pricing)",
        },
        "ollama": {
            "provider": "Ollama (local)",
            "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
            "data_residency": "local machine — no data leaves",
            "guardrails": "prompt sanitization + response validation",
            "cost": "free (local compute)",
        },
    }
    return labels[provider]


# ═══════════════════════════════════════════════════════════════════════════════
# LLM config builders
# ═══════════════════════════════════════════════════════════════════════════════

def _bedrock_config() -> dict:
    model_id = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    region = os.getenv("AWS_REGION", "us-east-1")

    entry: dict = {
        "model": model_id,
        "api_type": "bedrock",
        "aws_region": region,
    }

    # Optional: AWS credentials (falls back to IAM role / ~/.aws/credentials)
    if os.getenv("AWS_ACCESS_KEY_ID"):
        entry["aws_access_key"] = os.getenv("AWS_ACCESS_KEY_ID")
        entry["aws_secret_key"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    # Embed Bedrock Guardrails at the model invocation level
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID")
    if guardrail_id:
        entry["additional_model_fields"] = {
            "guardrailConfig": {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
                "trace": "enabled",   # surface guardrail decisions in logs
            }
        }
        logger.info(
            "Bedrock Guardrail enabled: id=%s version=%s",
            guardrail_id,
            os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
        )
    else:
        logger.warning(
            "No BEDROCK_GUARDRAIL_ID set. "
            "Consider configuring an AWS Bedrock Guardrail for regulated workloads. "
            "Falling back to software-layer prompt sanitization only."
        )

    return {
        "config_list": [entry],
        "temperature": 0,
    }


def _anthropic_config() -> dict:
    return {
        "config_list": [
            {
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "api_type": "anthropic",
            }
        ],
        "temperature": 0,
    }


def _openai_config() -> dict:
    return {
        "config_list": [
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ],
        "temperature": 0,
    }


def _ollama_config() -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return {
        "config_list": [
            {
                "model": model,
                "api_type": "ollama",
                "client_host": base_url,
            }
        ],
        "temperature": 0,
    }


_BUILDERS = {
    "bedrock": _bedrock_config,
    "anthropic": _anthropic_config,
    "openai": _openai_config,
    "ollama": _ollama_config,
}


def build_llm_config(provider: Optional[str] = None) -> dict:
    """
    Build an AG2-compatible llm_config dict for the detected (or forced) provider.

    Args:
        provider: Override provider. One of: bedrock | anthropic | openai | ollama
                  If None, auto-detected from environment.

    Returns:
        AG2 llm_config dict with config_list and temperature.
    """
    p = (provider or detect_provider()).lower()
    if p not in _BUILDERS:
        raise ValueError(f"Unknown LLM provider '{p}'. Choose from: {list(_BUILDERS)}")
    cfg = _BUILDERS[p]()
    info = provider_label()
    logger.info(
        "LLM config built: provider=%s model=%s residency=%s",
        info["provider"], info["model"], info["data_residency"],
    )
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# [L1] Prompt sanitization — pre-call
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_text(text: str) -> tuple[str, list[str]]:
    """
    Replace PII patterns in text with placeholders before sending to any LLM.

    Returns:
        (sanitized_text, list_of_redaction_descriptions)

    Example:
        "customer john@example.com called 555-123-4567"
        → "customer [EMAIL] called [PHONE]"
    """
    redactions: list[str] = []
    for pattern, placeholder in _TEXT_PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            redactions.append(f"{placeholder}: {len(matches)} instance(s) redacted")
            text = pattern.sub(placeholder, text)
    return text, redactions


def sanitize_context_for_llm(context_variables: dict) -> dict:
    """
    Return a copy of context_variables safe to embed in LLM prompts.

    Removes or redacts fields that may contain raw PII data:
      - Strips actual column *values* from quality reports
      - Keeps schema/metadata (column names, types, counts) — needed for reasoning
      - Removes df_preview, sample data, raw file content

    This is applied automatically by the pipeline before initiate_swarm_chat.
    """
    _STRIP_KEYS = {
        "_df",                 # raw DataFrame object
        "df_preview",          # raw sample rows
        "raw_content",         # raw file content
    }
    safe: dict = {}
    for k, v in context_variables.items():
        if k in _STRIP_KEYS:
            safe[k] = "[stripped for LLM safety]"
        elif isinstance(v, str) and len(v) > 500:
            # For long strings, sanitize PII patterns
            sanitized, redactions = sanitize_text(v)
            safe[k] = sanitized
            if redactions:
                logger.debug("sanitize_context_for_llm: %s → %s", k, redactions)
        else:
            safe[k] = v
    return safe


def _sanitize_enabled() -> bool:
    return os.getenv("DATAPAI_SANITIZE_PROMPTS", "true").lower() != "false"


def _validate_enabled() -> bool:
    return os.getenv("DATAPAI_VALIDATE_RESPONSES", "true").lower() != "false"


# ═══════════════════════════════════════════════════════════════════════════════
# [L2] Bedrock Guardrails — at-API level
# ═══════════════════════════════════════════════════════════════════════════════

def apply_bedrock_guardrail(
    text: str,
    direction: str = "INPUT",
) -> tuple[str, bool, str]:
    """
    Call the AWS Bedrock ApplyGuardrail API to evaluate text.

    Only active when BEDROCK_GUARDRAIL_ID is set.
    Falls back gracefully (returns original text) if Bedrock is unavailable.

    Args:
        text:      The text to evaluate (prompt or response).
        direction: 'INPUT' (prompt) or 'OUTPUT' (response).

    Returns:
        (filtered_text, was_blocked, action_taken)
        - filtered_text: original or guardrail-modified text
        - was_blocked:   True if the guardrail blocked the content
        - action_taken:  'NONE' | 'GUARDRAIL_INTERVENED'
    """
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID")
    if not guardrail_id:
        return text, False, "NONE"

    try:
        import boto3  # noqa: PLC0415

        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        response = client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
            source=direction,
            content=[{"text": {"text": text}}],
        )

        action = response.get("action", "NONE")
        was_blocked = action == "GUARDRAIL_INTERVENED"

        if was_blocked:
            outputs = response.get("outputs", [])
            filtered = outputs[0]["text"]["text"] if outputs else "[BLOCKED BY GUARDRAIL]"
            logger.warning(
                "Bedrock Guardrail intervened (%s). "
                "Guardrail assessments: %s",
                direction,
                response.get("assessments", []),
            )
            return filtered, True, action

        return text, False, action

    except Exception as exc:
        logger.warning(
            "Bedrock Guardrail call failed (non-fatal, continuing): %s", exc
        )
        return text, False, "ERROR"


# ═══════════════════════════════════════════════════════════════════════════════
# [L3] Response validation — post-call
# ═══════════════════════════════════════════════════════════════════════════════

def validate_llm_response(response_text: str) -> tuple[bool, list[str]]:
    """
    Scan an LLM response for PII leakage.

    Returns:
        (is_clean, list_of_findings)
        - is_clean: True if no PII patterns found in response
        - findings: description of any detected patterns
    """
    if not _validate_enabled():
        return True, []

    findings: list[str] = []
    for pattern, label in _TEXT_PII_PATTERNS:
        matches = pattern.findall(response_text)
        if matches:
            findings.append(f"{label}: {len(matches)} potential match(es) in response")

    if findings:
        logger.warning("LLM response PII validation: %s", findings)

    return len(findings) == 0, findings


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: full guardrail pipeline (L1 + L2 on prompt, L2 + L3 on response)
# ═══════════════════════════════════════════════════════════════════════════════

def guard_prompt(prompt: str) -> tuple[str, dict]:
    """
    Apply all pre-call guardrail layers to a prompt string.

    Returns:
        (guarded_prompt, guardrail_metadata)
    """
    meta: dict = {}

    # L1: Prompt sanitization
    if _sanitize_enabled():
        prompt, redactions = sanitize_text(prompt)
        meta["l1_sanitization"] = redactions

    # L2: Bedrock Guardrail (INPUT direction)
    prompt, blocked, action = apply_bedrock_guardrail(prompt, "INPUT")
    meta["l2_bedrock_guardrail"] = {"action": action, "blocked": blocked}

    return prompt, meta


def guard_response(response: str) -> tuple[str, dict]:
    """
    Apply all post-call guardrail layers to an LLM response string.

    Returns:
        (guarded_response, guardrail_metadata)
    """
    meta: dict = {}

    # L2: Bedrock Guardrail (OUTPUT direction)
    response, blocked, action = apply_bedrock_guardrail(response, "OUTPUT")
    meta["l2_bedrock_guardrail"] = {"action": action, "blocked": blocked}

    # L3: Response validation
    is_clean, findings = validate_llm_response(response)
    meta["l3_response_validation"] = {"is_clean": is_clean, "findings": findings}

    return response, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Cost-aware LLM config builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_cost_aware_llm_config(
    provider: Optional[str] = None,
    budget_pct_used: float = 0.0,
) -> dict:
    """
    Build an AG2 llm_config, automatically downgrading the model when
    the pipeline is approaching its cost budget.

    Args:
        provider:        Force provider. Auto-detected from env if None.
        budget_pct_used: Current spend as a % of the per-run budget (0–100).
                         Sourced from CostController.check_mid_run().

    Downgrade thresholds (from DATAPAI_BUDGET_WARN_PCT, default 80 %):
      ≥ warn_pct  → switch to the cheaper fallback model for the same provider
      ≥ 95 %      → switch to Ollama local if available (free)

    Returns AG2-compatible llm_config dict.
    """
    from .cost_control import BudgetConfig, _MODEL_FALLBACKS  # noqa: PLC0415

    budget = BudgetConfig()
    p = (provider or detect_provider()).lower()
    cfg = build_llm_config(p)

    if budget_pct_used < budget.warn_pct:
        return cfg   # budget comfortable — use default model

    current_model: str = cfg["config_list"][0].get("model", "")

    # ── Tier 1: same-provider cheaper model (≥ warn_pct) ──────────────────
    fallback_model = _MODEL_FALLBACKS.get(current_model)
    if fallback_model and budget_pct_used < 95:
        logger.warning(
            "[COST] Budget at %.1f%% — downgrading model: %s → %s",
            budget_pct_used, current_model, fallback_model,
        )
        cfg["config_list"][0]["model"] = fallback_model
        return cfg

    # ── Tier 2: Ollama local at ≥ 95 % (free) ─────────────────────────────
    if budget_pct_used >= 95 and (
        os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL")
    ):
        logger.warning(
            "[COST] Budget at %.1f%% — switching to Ollama local (free) "
            "to avoid exceeding limit.",
            budget_pct_used,
        )
        return _ollama_config()

    # Fallback: same tier-1 cheaper model (even if > 95%)
    if fallback_model:
        cfg["config_list"][0]["model"] = fallback_model
    return cfg

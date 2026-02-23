from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
import requests

# BUGFIX: Strip Markdown code fences for reviewer output
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    # remove first fence line
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    # remove last fence line if it's also ```
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# =============================================================================
# Base interface
# =============================================================================

class BaseChatClient(ABC):
    """Abstract base class for chat LLMs."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Must return at least: {"role": "assistant", "content": "<text>"}
        """
        raise NotImplementedError


# =============================================================================
# Ollama client (local LLM, LAST resort)
# =============================================================================

class OllamaChatClient(BaseChatClient):
    """
    Minimal chat client for Ollama.

    Model from env:
      OLLAMA_MODEL

    Example:
      export OLLAMA_MODEL=llama3.1
      export OLLAMA_MODEL=deepseek-coder:33b
    """

    def __init__(self, model: Optional[str] = None, base_url: str = "http://localhost:11434"):
        self.model = model or os.environ.get("OLLAMA_MODEL")
        if not self.model:
            raise ValueError(
                "OLLAMA_MODEL env var is not set and no model was passed to OllamaChatClient.\n"
                "Example: export OLLAMA_MODEL=llama3.1"
            )
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "message" in data:
            msg = data["message"]
            return {
                "role": msg.get("role", "assistant"),
                "content": msg.get("content", ""),
            }
        return data


# =============================================================================
# OpenAI client (GPT-5.1 etc)
# =============================================================================

class OpenAIChatClient(BaseChatClient):
    """
    Wrapper for OpenAI chat models.

    Env:
      OPENAI_API_KEY   – required
      OPENAI_MODEL     – required (no hardcoded default)

    Examples:
      export OPENAI_MODEL=gpt-5.1
      export OPENAI_MODEL=gpt-4.1
      export OPENAI_MODEL=o3-mini
    """

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI  # lazy import
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY env var not set.")

        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise ValueError(
                "OPENAI_MODEL env var is not set and no model was passed to OpenAIChatClient.\n"
                "Example: export OPENAI_MODEL=gpt-5.1"
            )

        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )

        if stream:
            full_content = ""
            for chunk in resp:
                delta = chunk.choices[0].delta
                full_content += (delta.content or "")
            return {"role": "assistant", "content": full_content}
        else:
            msg = resp.choices[0].message
            return {"role": msg.role, "content": msg.content}

# -------------------------
# Google Gemini Chat Client (HTTP-based, works on Python 3.8)
# -------------------------

class GoogleChatClient(BaseChatClient):
    """
    Simple HTTP client for Google Gemini (Developer API).

    Uses:
      - GOOGLE_API_KEY  (required)
      - GOOGLE_MODEL    (optional, default 'gemini-2.5-flash-lite')

    We call the HTTP endpoint:
      POST https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key=API_KEY

    Messages are OpenAI-style:
      [{"role": "user"|"assistant"|"system", "content": "text"}]
    """

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY env var is not set.")

        self.model_name = model or os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash-lite")
        self.base_url = "https://generativelanguage.googleapis.com/v1"

    def _build_contents(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages into Gemini 'contents' array.
        """
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            # Gemini expects roles like "user" / "model" typically;
            # it also tolerates "system" in practice, we'll just pass through.
                        # Map OpenAI roles -> Gemini roles
            if role == "assistant":
                gem_role = "model"
            else:
                # system + user -> 'user'
                gem_role = "user"
            contents.append(
                {
                    "role": gem_role,
                    "parts": [{"text": text}],
                }
            )
        return contents

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        if stream:
            # For now, we don't implement streaming via HTTP in this client.
            # Your agents don't rely on streaming, so we keep it simple.
            raise NotImplementedError("Streaming not implemented for GoogleChatClient.")

        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"

        payload = {
            "contents": self._build_contents(messages),
            "generationConfig": {
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
        except Exception as e:
            raise RuntimeError(f"HTTP request to Gemini failed: {e}")

        if resp.status_code != 200:
            raise RuntimeError(
                f"Gemini API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        # Extract the first candidate's text
        text_parts: List[str] = []
        for cand in data.get("candidates", []):
            content = cand.get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])

        final_text = "\n".join(text_parts) if text_parts else ""

        return {"role": "assistant", "content": final_text}


# =============================================================================
# Bedrock client (Claude Sonnet 4.5 etc)
# =============================================================================

class BedrockChatClient(BaseChatClient):
    """
    Wrapper for AWS Bedrock text/chat models (e.g. Claude Sonnet 4.5).

    Env:
      BEDROCK_MODEL_ID   – required
      BEDROCK_REGION     – optional, default 'ap-southeast-2' (Sydney)

    Example:
      export BEDROCK_MODEL_ID=anthropic.claude-4.5-sonnet
      export BEDROCK_REGION=ap-southeast-2
    """

    def __init__(self, model_id: Optional[str] = None, region_name: Optional[str] = None):
        import boto3

        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
        if not self.model_id:
            raise ValueError(
                "BEDROCK_MODEL_ID env var is not set and no model_id was passed to BedrockChatClient.\n"
                "Example: export BEDROCK_MODEL_ID=anthropic.claude-4.5-sonnet"
            )

        # Default to Sydney
        self.region_name = region_name or os.environ.get("BEDROCK_REGION", "ap-southeast-2")
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Assumes Anthropic-style "messages" model on Bedrock.
        If you switch to Nova or another style, adjust the body/response parsing.
        """
        # Simple conversion: flatten messages into one prompt.
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            prompt += f"{role.upper()}: {content}\n"

        body = {
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "inferenceConfig": {
                "temperature": temperature,
            },
        }

        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
        )

        resp_body = json.loads(resp["body"].read().decode("utf-8"))
        # NOTE: this structure may vary between models; adjust if needed.
        model_message = resp_body["output"]["message"]
        text = "".join(part.get("text", "") for part in model_message.get("content", []))

        return {"role": "assistant", "content": text}


# =============================================================================
# Router client (OpenAI primary + Bedrock reviewer, Ollama last-resort)
# =============================================================================

class RouterChatClient(BaseChatClient):
    """
    Multi-provider router with optional dual-review.

    Design:
      - Primary model: OpenAI or Bedrock (Claude)
      - Secondary model: optional reviewer (e.g. Claude reviewing GPT-5.1)
      - Ollama is only used if you explicitly choose local/hybrid mode.

    Env:

      LLM_MODE:
        - "paid"    -> use cloud models (OpenAI/Bedrock) ONLY   [recommended]
        - "local"   -> use Ollama ONLY
        - "hybrid"  -> try Ollama, fallback to primary provider

      LLM_PRIMARY_PROVIDER:
        - "openai"  (GPT-5.1, etc)
        - "bedrock" (Claude Sonnet 4.5, etc)
        default: "openai"

      LLM_SECONDARY_PROVIDER:
        - "openai"
        - "bedrock"
        - "google"
        default: "google"

      LLM_DUAL_REVIEW:
        - "1" -> enable secondary review of primary output
        - else -> disabled

      OLLAMA_MODEL     – e.g. llama3.1
      OPENAI_MODEL     – e.g. gpt-5.1
      BEDROCK_MODEL_ID – e.g. anthropic.claude-4.5-sonnet
      BEDROCK_REGION   – default 'ap-southeast-2'
    """

    def __init__(self):
        self.llm_enabled = os.environ.get("DATAPAI_LLM_ENABLED", "true").lower() == "true"

        mode = os.environ.get("LLM_MODE", "paid").lower()
        if mode not in ("paid", "local", "hybrid"):
            mode = "paid"
        self.mode = mode

        self.primary_provider = os.environ.get("LLM_PRIMARY_PROVIDER", "openai").lower()
        self.secondary_provider = os.environ.get("LLM_SECONDARY_PROVIDER", "google").lower()
        self.dual_review_enabled = os.environ.get("LLM_DUAL_REVIEW", "0") == "1"

        self._ollama: Optional[OllamaChatClient] = None
        self._openai: Optional[OpenAIChatClient] = None
        self._bedrock: Optional[BedrockChatClient] = None
        self._google: Optional[GoogleChatClient] = None


    # ----- backend getters -----

    def _get_ollama(self) -> OllamaChatClient:
        if self._ollama is None:
            self._ollama = OllamaChatClient()
        return self._ollama

    def _get_openai(self) -> OpenAIChatClient:
        if self._openai is None:
            self._openai = OpenAIChatClient()
        return self._openai

    def _get_bedrock(self) -> BedrockChatClient:
        if self._bedrock is None:
            self._bedrock = BedrockChatClient()
        return self._bedrock

    def _get_google(self) -> GoogleChatClient:
        if self._google is None:
            self._google = GoogleChatClient()
        return self._google


    def _call_provider(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        temperature: float,
        stream: bool,
    ) -> Dict[str, Any]:
        provider = provider.lower()
        if provider == "openai":
            return self._get_openai().chat(messages, temperature=temperature, stream=stream)
        elif provider == "bedrock":
            return self._get_bedrock().chat(messages, temperature=temperature, stream=stream)
        elif provider == "ollama":
            return self._get_ollama().chat(messages, temperature=temperature, stream=stream)
        elif provider == "google":
            return self._get_google().chat(messages, temperature=temperature, stream=stream)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ----- dual-review logic -----

    def _dual_review(
        self,
        draft: Dict[str, Any],
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Dict[str, Any]:
        """
        Call secondary provider (reviewer) to check/possibly rewrite draft.
        Reviewer must respond with JSON, e.g.:

        {"decision": "approve"}

        or

        {"decision": "rewrite", "answer": "<fixed answer>"}
        """

        print("[RouterChatClient] Dual review active…")
        print("[RouterChatClient] Primary provider =", self.primary_provider)
        print("[RouterChatClient] Secondary provider =", self.secondary_provider)

        review_messages = messages + [
            {
                "role": "user",
                "content": (
                    "You are reviewing another model's answer to the previous prompt. "
                    "You must respond with STRICT JSON only, no extra text.\n\n"
                    "If the answer is correct, safe, and high quality, reply:\n"
                    '{"decision": "approve"}\n\n'
                    "If the answer is wrong, unsafe, hallucinated, or low quality, reply:\n"
                    '{"decision": "rewrite", "answer": "<your improved answer>"}'
                ),
            }
        ]

        try:
            reviewer_reply = self._call_provider(
                self.secondary_provider,
                review_messages,
                temperature,
                stream=False,
            )

            raw = reviewer_reply.get("content", "").strip()
            print("[RouterChatClient] Reviewer RAW output:", raw)

            import json
            decision = _strip_code_fences(raw)

            if decision.get("decision") == "rewrite":
                fixed = (decision.get("answer") or "").strip()
                if fixed:
                    print("[RouterChatClient] Reviewer REWROTE the answer.")
                    return {"role": "assistant", "content": fixed}

            print("[RouterChatClient] Reviewer APPROVED primary answer.")
            return draft

        except Exception as e:
            print("[RouterChatClient] Reviewer FAILED:", e)
            return draft


    # ----- main chat -----

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Main router:

        - If LLM_MODE="paid":
            -> use cloud primary provider (OpenAI or Bedrock)
        - If LLM_MODE="local":
            -> use Ollama only (for paranoid, no-cloud customers)
        - If LLM_MODE="hybrid":
            -> try Ollama first, then fall back to cloud primary
        """
        if not self.llm_enabled:
            raise RuntimeError(
                "LLM usage is disabled (DATAPAI_LLM_ENABLED=false). "
                "Deterministic executor mode enforced."
            )

        
        if self.mode == "paid":
            draft = self._call_provider(self.primary_provider, messages, temperature, stream)

        elif self.mode == "local":
            draft = self._call_provider("ollama", messages, temperature, stream)

        elif self.mode == "hybrid":
            try:
                draft = self._call_provider("ollama", messages, temperature, stream)
            except Exception as e:
                print(f"[RouterChatClient] Local Ollama failed, using paid backend: {e}")
                draft = self._call_provider(self.primary_provider, messages, temperature, stream)
        else:
            draft = self._call_provider(self.primary_provider, messages, temperature, stream)

        # Optional second-pass review (e.g. Claude reviewing GPT-5.1)
        draft = self._dual_review(draft, messages, temperature)

        return draft


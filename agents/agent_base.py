# agent_base.py

import json
from typing import Any, Dict, List, Optional

from llm_client import BaseChatClient  # NOTE: we no longer import RouterChatClient here
#from tools import list_tools, call_tool_from_json_call
import tools


# BUGFIX: Robust JSON parsing for LLM outputs (handles code fences + NDJSON)
def _parse_single_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()

    # Strip Markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Drop first and last ``` lines
        if len(lines) >= 2:
            # remove leading ``` / ```json line
            first = lines[0].strip().lower()
            if first.startswith("```"):
                lines = lines[1:]
            # remove trailing ``` if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

    # First try: whole string is one JSON object
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Second try: NDJSON – multiple JSON objects separated by newlines
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Ignore obvious non-JSON lines
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                # BUGFIX: Return first valid JSON object (typically the first tool_call)
                return parsed
        except Exception:
            continue

    # If we reach here, we couldn't parse anything usable
    raise ValueError("LLM output is not a valid JSON object")

DEFAULT_SYSTEM_PROMPT = """
You are an AI agent that can call Python tools to achieve a modern cloud ETL/ELT lakehouse goal.

You MUST respond in strict JSON only, with no additional text.

Two response types:

1) To call a tool:
{
  "type": "tool_call",
  "tool_name": "<tool_name>",
  "args": { ... }
}

2) To return a final result and stop:
{
  "type": "final_answer",
  "result": "Human-readable summary of what you did and any results."
}

You have access to tools defined in 'available_tools'.
Use multiple tool calls in sequence if needed.
Use 'history' to see what happened in earlier steps.
Never invent tool names; only use those in 'available_tools'.
""".strip()


class BaseAgent:
    """
    Base class for AI-driven agents.

    Implements a generic loop:
      (goal + context) -> LLM -> (tool_call | final_answer) -> repeat

    IMPORTANT:
      - This class does NOT create its own LLM.
      - You MUST pass an llm instance (e.g. RouterChatClient) from outside.

      Example wiring (in app_ai_agent.py):

          from llm_client import RouterChatClient
          from dbt_agent import DbtAgent

          shared_llm = RouterChatClient()
          dbt_agent = DbtAgent(llm=shared_llm)
    """

    def __init__(
        self,
        name: str,
        llm: Optional[BaseChatClient],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        temperature: float = 0.1,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature

    def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the generic tool-using loop until:
          - the LLM returns type='final_answer', or
          - max_steps is reached, or
          - an error occurs.

        Returns a dict with:
          - status: "ok" | "error" | "timeout"
          - result / reason
          - history
          - steps
        """
        if self.llm is None:
            return {
                "status": "error",
                "reason": "BaseAgent.llm is None. You must pass an LLM (e.g. RouterChatClient) when creating the agent.",
                "history": [],
            }

        context = context or {}
        history: List[Dict[str, Any]] = []
        step = 0

        while step < self.max_steps:
            step += 1

            payload = {
                "agent_name": self.name,
                "goal": goal,
                "context": context,
                "history": history,
                "available_tools": tools.list_tools(),
            }

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ]

            # Call the shared LLM router (OpenAI primary + Bedrock reviewer, etc.)
            response = self.llm.chat(messages, temperature=self.temperature)
            raw_content = (response.get("content") or "").strip()

            # We expect STRICT JSON from the LLM
            try:
                decision = _parse_single_json_object(raw_content)
            except:
                return {
                    "status": "error",
                    "reason": "LLM did not return valid JSON",
                    "raw_content": raw_content,
                    "history": history,
                }

            rtype = decision.get("type")

            # -------------------------
            # Case 1: Final answer
            # -------------------------
            if rtype == "final_answer":
                return {
                    "status": "ok",
                    "result": decision.get("result"),
                    "history": history,
                    "steps": step,
                }

            # -------------------------
            # Case 2: Tool call
            # -------------------------
            elif rtype == "tool_call":
                tool_name = decision.get("tool_name")
                args = decision.get("args", {})

                tool_call_json_str = json.dumps(
                    {
                        "tool_name": tool_name,
                        "args": args,
                    }
                )

                try:
                    tool_result = tools.call_tool_from_json_call(tool_call_json_str)
                except Exception as e:
                    tool_result = {"error": str(e)}

                history.append(
                    {
                        "step": step,
                        "action": "tool_call",
                        "tool_name": tool_name,
                        "args": args,
                        "result": tool_result,
                    }
                )

            # -------------------------
            # Case 3: Unknown type
            # -------------------------
            else:
                return {
                    "status": "error",
                    "reason": f"Unknown decision type: {rtype}",
                    "decision": decision,
                    "history": history,
                }

        # If we exit the loop without final_answer
        return {
            "status": "timeout",
            "reason": f"Max steps ({self.max_steps}) exceeded",
            "history": history,
        }


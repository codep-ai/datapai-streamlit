# supervisor_agent.py
"""
SupervisorAgent:
----------------
A high-level AI “manager agent” that decides:
  - which sub-agent to call (dbt_agent, file_ingest_agent, etc.)
  - in what order
  - and orchestrates their results

Supervisor receives a GOAL and uses LLM reasoning + tool calls
to coordinate your ETL/ELT multi-agent pipeline.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Optional, List

from agent_base import BaseAgent
from llm_client import BaseChatClient, RouterChatClient


# ------------------------------
# SYSTEM PROMPT
# ------------------------------
SUPERVISOR_SYSTEM_PROMPT = """
You are the SUPERVISOR AGENT for a multi-agent AI ETL/ELT framework.

Your job:
  - Understand the user's GOAL
  - Select the correct agent or tool(s)
  - Execute them in proper order
  - Produce a final human-readable summary

You MUST respond in STRICT JSON:
  - For tool call:
    {
      "type": "tool_call",
      "tool_name": "<tool_name>",
      "args": { ... }
    }

  - For final answer:
    {
      "type": "final_answer",
      "result": "A concise summary"
    }

Rules:
  - NEVER invent tool names.
  - Only use tools listed in "available_tools".
  - You can call multiple tools sequentially.
  - Use history to see previous results.
  - ALWAYS produce valid JSON. No markdown, no comments.
  - If you need more info, ask via final_answer (the UI will request clarification).

You are NOT performing SQL/ETL logic yourself — you delegate to:
  - dbt_agent
  - file_ingest_agent
  - knowledge_ingest_agent
  - any other registered tools
""".strip()


# ------------------------------
# SUPERVISOR AGENT CLASS
# ------------------------------

class SupervisorAgent(BaseAgent):
    """
    High-level orchestrator agent.

    Chooses which agent to run, based on the goal.
    """

    def __init__(
        self,
        llm: Optional[BaseChatClient] = None,
        max_steps: int = 12,
        temperature: float = 0.10,
    ):
        # Default to using your unified GPT-5.1 + Gemini reviewer router
        if llm is None:
            llm = RouterChatClient()

        super().__init__(
            name="supervisor_agent",
            llm=llm,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            max_steps=max_steps,
            temperature=temperature,
        )

    # (Most of the logic lives in BaseAgent.run(), so SupervisorAgent is simple.)


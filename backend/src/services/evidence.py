from __future__ import annotations

import json

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState
from config import Configuration
from utils import strip_thinking_tokens
from services.text_processing import strip_tool_calls

class EvidenceVerificationService:
    """Verifies whether task conclusions are sufficiently supported by search evidence."""

    def __init__(self, agent: ToolAwareSimpleAgent) -> None:
        self._agent = agent

    def verify(
        self,
        task,
        context: str,
    ) -> str:
        prompt = f"""
{evidence_verifier_instructions}

<RESEARCH_TASK>
任务标题：{task.title}
任务目标：{task.intent}
检索查询：{task.query}
</RESEARCH_TASK>

<SEARCH_RESULTS>
{context}
</SEARCH_RESULTS>
"""

        response = self._agent.run(prompt)
        self._agent.clear_history()

        return response.strip()
"""Tool registry and safe result formatting for academic researchers."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from tools import research_tools


@tool
def think_tool(reflection: str) -> str:
    """Record a short reflection about evidence gathered and the next research step."""
    return f"Research reflection recorded: {reflection}"


RESEARCH_TOOLS: list[BaseTool] = [*research_tools, think_tool]
RESEARCH_TOOLS_BY_NAME = {item.name: item for item in RESEARCH_TOOLS}


def format_tool_result(value: Any, max_chars: int) -> str:
    """Serialize provider output without hiding failures or silently losing URLs."""
    if isinstance(value, str):
        result = value
    else:
        result = json.dumps(value, ensure_ascii=False, default=str)
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + "\n[工具结果过长，后续内容已截断；需要时缩小查询范围。]"

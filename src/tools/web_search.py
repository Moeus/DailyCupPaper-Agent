"""Public web search tool backed by Tavily."""

from __future__ import annotations

import os
from typing import Literal

import httpx
from langchain_core.tools import tool


@tool(parse_docstring=True)
async def web_search(
    query: str,
    limit: int = 5,
    topic: Literal["general", "news"] = "general",
) -> str:
    """搜索公开网页以核查最新信息，返回标题、原始 URL 和内容摘要。

    Args:
        query: 要搜索的问题、主题或关键词。
        limit: 希望返回的网页数量，默认 5。
        topic: 搜索类别；general 查询普通网页，news 查询新闻，默认 general。

    Returns:
        网页标题、原始 URL 和内容摘要的文本列表；有发布日期时一并给出。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY environment variable is required")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "max_results": limit,
                "topic": topic,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        response.raise_for_status()

    results = response.json().get("results", [])
    lines = ["Web search results:"]
    for index, item in enumerate(results, start=1):
        lines.extend(
            [
                f"{index}. {item.get('title') or 'Untitled'}",
                f"   URL: {item.get('url') or ''}",
                f"   Content: {item.get('content') or ''}",
            ]
        )
        if item.get("published_date"):
            lines.append(f"   Published: {item['published_date']}")
    if not results:
        lines.append("No results found.")
    return "\n".join(lines)

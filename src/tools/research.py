"""Research actions exposed to the agent, independent of search providers."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from tools.semantic_scholar import (
    DEFAULT_PAPER_FIELDS,
    DEFAULT_SNIPPET_FIELDS,
    GRAPH_BASE_URL,
    RECOMMENDATIONS_BASE_URL,
    _paper_path_id,
    _semantic_scholar_request,
)


@tool(parse_docstring=True)
async def get_paper_by_key_word(
    query: str,
    limit: int = 10,
    year: str | None = None,
    venue: str | None = None,
    min_citation_count: int | None = None,
) -> list[dict[str, Any]]:
    """按研究主题或关键词查找相关论文，适合尚未确定具体论文标题时使用。返回论文详情列表。

    Args:
        query: 研究主题、关键词或检索短语。
        limit: 希望返回的论文篇数，默认 10。
        year: 可选的发表年份或年份范围，例如 "2023-2025"。
        venue: 可选的会议或期刊名称。
        min_citation_count: 可选的最低被引用次数。

    Returns:
        含论文标题、ID、摘要、作者、年份、引用数和链接的 JSON 对象列表。
    """
    return await _semantic_scholar_request(
        "GET",
        GRAPH_BASE_URL,
        "/paper/search",
        params={
            "query": query,
            "limit": limit,
            "year": year,
            "venue": venue,
            "minCitationCount": min_citation_count,
            "fields": DEFAULT_PAPER_FIELDS,
        },
    )


@tool(parse_docstring=True)
async def get_paper_by_title(title: str) -> list[dict[str, Any]]:
    """根据完整或近似标题定位最匹配的一篇论文，返回其详情。

    Args:
        title: 要查找的论文标题或近似标题。

    Returns:
        含匹配论文详情的单元素 JSON 对象列表。
    """
    return await _semantic_scholar_request(
        "GET",
        GRAPH_BASE_URL,
        "/paper/search/match",
        params={"query": title, "fields": DEFAULT_PAPER_FIELDS},
    )


@tool(parse_docstring=True)
async def get_paper_by_id(paper_id: str) -> list[dict[str, Any]]:
    """根据已知标识符获取一篇论文的详情，包括摘要、作者和链接。

    Args:
        paper_id: 论文 ID、DOI: 前缀的 DOI、ARXIV: 前缀的 arXiv ID、PMID 或论文 URL。

    Returns:
        含该论文详情的单元素 JSON 对象列表。
    """
    return await _semantic_scholar_request(
        "GET",
        GRAPH_BASE_URL,
        f"/paper/{_paper_path_id(paper_id)}",
        params={"fields": DEFAULT_PAPER_FIELDS},
    )


@tool(parse_docstring=True)
async def get_paper_citations(paper_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """查找引用指定论文的后续论文，返回后续论文列表。

    Args:
        paper_id: 被引用论文的标识符。
        limit: 希望返回的后续论文篇数，默认 100。

    Returns:
        JSON 对象列表；每项包含 citingPaper 及其标题、年份等信息。
    """
    return await _semantic_scholar_request(
        "GET",
        GRAPH_BASE_URL,
        f"/paper/{_paper_path_id(paper_id)}/citations",
        params={"fields": DEFAULT_PAPER_FIELDS, "limit": limit},
    )


@tool(parse_docstring=True)
async def get_paper_references(paper_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """查找指定论文引用的参考文献，返回基础论文列表。

    Args:
        paper_id: 要查看参考文献的论文标识符。
        limit: 希望返回的参考文献篇数，默认 100。

    Returns:
        JSON 对象列表；每项包含 citedPaper 及其标题、年份等信息。
    """
    return await _semantic_scholar_request(
        "GET",
        GRAPH_BASE_URL,
        f"/paper/{_paper_path_id(paper_id)}/references",
        params={"fields": DEFAULT_PAPER_FIELDS, "limit": limit},
    )


@tool(parse_docstring=True)
async def find_paper_evidence(
    query: str,
    paper_ids: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """查找支持某个问题或论点的论文文本片段，返回片段及所属论文。

    Args:
        query: 要寻找证据的问题、术语或关键陈述。
        paper_ids: 可选的论文 ID 列表；提供时只在这些论文中检索。
        limit: 希望返回的片段数，默认 10。

    Returns:
        含相关片段、章节及所属论文 ID 和标题的 JSON 对象列表。
    """
    return await _semantic_scholar_request(
        "GET",
        GRAPH_BASE_URL,
        "/snippet/search",
        params={
            "query": query,
            "paperIds": paper_ids,
            "fields": DEFAULT_SNIPPET_FIELDS,
            "limit": limit,
        },
    )


@tool(parse_docstring=True)
async def get_related_papers(paper_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """从一篇核心论文扩展主题相近的论文，返回推荐论文列表。

    Args:
        paper_id: 作为推荐种子的论文标识符。
        limit: 希望返回的相关论文篇数，默认 100。

    Returns:
        含推荐论文标题、摘要、年份、作者、引用数和链接的 JSON 对象列表。
    """
    return await _semantic_scholar_request(
        "GET",
        RECOMMENDATIONS_BASE_URL,
        f"/papers/forpaper/{_paper_path_id(paper_id)}",
        params={"fields": DEFAULT_PAPER_FIELDS, "limit": limit},
    )


@tool(parse_docstring=True)
async def recommend_papers_by_examples(
    positive_paper_ids: list[str],
    negative_paper_ids: list[str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """根据相关与不相关的示例论文寻找候选论文，返回推荐论文列表。

    Args:
        positive_paper_ids: 相关或偏好的种子论文 ID 列表，至少提供一篇。
        negative_paper_ids: 可选的不相关种子论文 ID 列表，用于排除方向。
        limit: 希望返回的推荐论文篇数，默认 100。

    Returns:
        含推荐论文标题、摘要、年份、作者、引用数和链接的 JSON 对象列表。
    """
    return await _semantic_scholar_request(
        "POST",
        RECOMMENDATIONS_BASE_URL,
        "/papers",
        params={"fields": DEFAULT_PAPER_FIELDS, "limit": limit},
        json={
            "positivePaperIds": positive_paper_ids,
            "negativePaperIds": negative_paper_ids or [],
        },
    )


paper_tools = [
    get_paper_by_key_word,
    get_paper_by_title,
    get_paper_by_id,
    get_paper_citations,
    get_paper_references,
    find_paper_evidence,
    get_related_papers,
    recommend_papers_by_examples,
]

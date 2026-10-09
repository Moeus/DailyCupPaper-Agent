"""LangGraph/LangChain tools for the Semantic Scholar API."""

from __future__ import annotations

import os
from asyncio import Lock, sleep
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import Any
from urllib.parse import quote

import httpx

GRAPH_BASE_URL = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE_URL = "https://api.semanticscholar.org/recommendations/v1"
MIN_REQUEST_INTERVAL_SECONDS = 1.0
REQUEST_RETRY_DELAY_SECONDS = 30.0
MAX_REQUEST_ATTEMPTS = 3
# 同一进程的所有论文工具共用此锁：最大并发为 1，退避期间也不放行其他请求。
_request_lock = Lock()
_last_request_started_at = 0.0

DEFAULT_PAPER_FIELDS = (
    "paperId,corpusId,externalIds,url,title,abstract,venue,year,"
    "publicationDate,authors,referenceCount,citationCount,"
    "influentialCitationCount,openAccessPdf,fieldsOfStudy,"
    "publicationTypes"
)
DEFAULT_SNIPPET_FIELDS = (
    "snippet.text,snippet.snippetKind,snippet.section,paper.title,paper.paperId"
)


def _api_key() -> str:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        msg = "SEMANTIC_SCHOLAR_API_KEY environment variable is required"
        raise RuntimeError(msg)
    return api_key


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or value == "":
            continue
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            if not value:
                continue
            clean[key] = ",".join(str(item) for item in value)
            continue
        clean[key] = value
    return clean


def _paper_path_id(paper_id: str) -> str:
    if paper_id.lower().startswith("s2:"):
        paper_id = paper_id[3:]
    return quote(paper_id, safe=":")


def _is_rate_limit_error(error: Exception) -> bool:
    return (
        isinstance(error, httpx.HTTPStatusError)
        and error.response.status_code == httpx.codes.TOO_MANY_REQUESTS
    )


def _retry_delay(error: Exception, attempt: int) -> float:
    delay = REQUEST_RETRY_DELAY_SECONDS * (2**attempt)
    if _is_rate_limit_error(error):
        retry_after = error.response.headers.get("retry-after")
        if retry_after and retry_after.isdigit():
            return max(delay, float(retry_after))
        if retry_after:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                return max(delay, (retry_at - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return delay


async def _wait_for_rate_limit_slot() -> None:
    global _last_request_started_at

    elapsed = monotonic() - _last_request_started_at
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        await sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_started_at = monotonic()


async def _semantic_scholar_request(
    method: str,
    base_url: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    json: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    headers = {"x-api-key": _api_key()}
    async with (
        _request_lock,
        httpx.AsyncClient(base_url=base_url, timeout=None) as client,
    ):
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            try:
                await _wait_for_rate_limit_slot()
                response = await client.request(
                    method,
                    path,
                    params=_clean_params(params or {}),
                    json=json,
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                if not isinstance(payload, dict):
                    raise TypeError(
                        "Semantic Scholar returned an invalid JSON structure"
                    )
                for key in ("data", "recommendedPapers"):
                    if isinstance(payload.get(key), list):
                        return payload[key]
                return [payload]
            except (httpx.HTTPError, ValueError, TypeError) as error:
                if attempt + 1 >= MAX_REQUEST_ATTEMPTS:
                    return [
                        {
                            "error": "查询错误",
                            "source": "semantic_scholar",
                            "attempts": attempt + 1,
                            "status_code": error.response.status_code
                            if isinstance(error, httpx.HTTPStatusError)
                            else None,
                            "error_type": type(error).__name__,
                            "message": "该查询连续三次失败；请根据已有资料和错误情况决定更换查询或结束检索，不要将此结果视为没有相关论文。",
                        }
                    ]
                await sleep(_retry_delay(error, attempt))

    msg = "Semantic Scholar request failed without an HTTP response"
    raise RuntimeError(msg)

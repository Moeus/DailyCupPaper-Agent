"""Small local JSON cache for extracted abstracts and snippets."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path


def cache_root() -> Path:
    return Path(
        os.getenv("DCP_RESEARCH_SOURCE_CACHE_DIR", ".data/research/sources")
    ).resolve()


def save_passages(paper_key: str, passages: list[dict[str, str]]) -> tuple[str, str]:
    payload = json.dumps(passages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = sha256(payload).hexdigest()
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}.json"
    path.write_bytes(payload)
    return digest, digest


def load_passages(content_ref: str) -> list[dict[str, str]]:
    if len(content_ref) != 64 or any(
        ch not in "0123456789abcdef" for ch in content_ref
    ):
        raise ValueError("invalid source cache reference")
    return json.loads(
        (cache_root() / f"{content_ref}.json").read_text(encoding="utf-8")
    )

"""Independent Alibaba Cloud Qwen configuration for the two supported graphs."""

import os
import re
from typing import Literal

from langchain_openai import ChatOpenAI

AgentKind = Literal["research", "translation"]


def _max_tokens_from_env(name: str) -> int:
    raw = os.getenv(name, "4096").strip()
    if not re.fullmatch(r"[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+", raw):
        raise ValueError(f"{name} must be a positive integer (commas may separate thousands)")
    return int(raw.replace(",", ""))


def model_settings(kind: AgentKind) -> dict:
    if kind not in ("research", "translation"):
        raise ValueError("Unknown agent kind")
    prefix = f"DCP_{kind.upper()}_AGENT_"
    return {
        "model": os.getenv(prefix + "MODEL", "qwen-plus"),
        "api_key": os.getenv(prefix + "API_KEY", "").strip(),
        "base_url": os.getenv(
            prefix + "API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "temperature": float(os.getenv(prefix + "TEMPERATURE", "0")),
        "max_tokens": _max_tokens_from_env(prefix + "MAX_TOKENS"),
        "timeout": float(os.getenv(prefix + "TIMEOUT_SECONDS", "90")),
        "max_retries": 1,
        "extra_body": {"enable_thinking": False},
    }


def build_model(kind: AgentKind) -> ChatOpenAI:
    settings = model_settings(kind)
    # Reject missing credentials before the SDK can fall back to shared provider variables.
    if not settings["api_key"]:
        raise ValueError(f"{kind} model provider is not configured")
    return ChatOpenAI(**settings)

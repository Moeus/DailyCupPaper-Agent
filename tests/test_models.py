from unittest.mock import Mock

import pytest

from agent_runtime import models


@pytest.mark.parametrize("kind", ["research", "translation"])
def test_qwen_defaults_and_independent_credentials(monkeypatch, kind):
    for agent in ("RESEARCH", "TRANSLATION"):
        for suffix in ("MODEL", "API_KEY", "API_BASE", "MAX_TOKENS"):
            monkeypatch.delenv(f"DCP_{agent}_AGENT_{suffix}", raising=False)

    monkeypatch.setenv(f"DCP_{kind.upper()}_AGENT_API_KEY", "workspace-key")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(models, "ChatOpenAI", constructor)

    result = models.build_model(kind)

    assert result is constructor.return_value
    settings = constructor.call_args.kwargs
    assert settings["model"] == "qwen-plus"
    assert settings["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings["api_key"] == "workspace-key"
    assert settings["extra_body"] == {"enable_thinking": False}

    other = "translation" if kind == "research" else "research"
    with pytest.raises(ValueError, match="model provider is not configured"):
        models.build_model(other)


def test_workspace_endpoint_is_passed_to_openai_client(monkeypatch):
    endpoint = "https://llm-25y6gpisv0ejnxo8.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    monkeypatch.setenv("DCP_TRANSLATION_AGENT_API_KEY", "workspace-key")
    monkeypatch.setenv("DCP_TRANSLATION_AGENT_API_BASE", endpoint)
    monkeypatch.setenv("DCP_TRANSLATION_AGENT_MODEL", "qwen-flash")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(models, "ChatOpenAI", constructor)

    models.build_model("translation")

    assert constructor.call_args.kwargs["base_url"] == endpoint
    assert constructor.call_args.kwargs["model"] == "qwen-flash"


@pytest.mark.parametrize("kind", ["research", "translation"])
@pytest.mark.parametrize("raw", ["131,072", "131072"])
def test_max_tokens_accepts_plain_and_grouped_integers(monkeypatch, kind, raw):
    monkeypatch.setenv(f"DCP_{kind.upper()}_AGENT_MAX_TOKENS", raw)

    assert models.model_settings(kind)["max_tokens"] == 131072


@pytest.mark.parametrize("raw", ["13,10", "0", "1.5"])
def test_max_tokens_rejects_invalid_values_with_variable_name(monkeypatch, raw):
    monkeypatch.setenv("DCP_RESEARCH_AGENT_MAX_TOKENS", raw)

    with pytest.raises(ValueError, match="DCP_RESEARCH_AGENT_MAX_TOKENS"):
        models.model_settings("research")

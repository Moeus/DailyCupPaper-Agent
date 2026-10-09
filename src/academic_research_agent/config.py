"""Runtime limits for the academic research graph."""

from __future__ import annotations

import os

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class ResearchConfig(BaseModel):
    # 主管最多做多少轮判断；每次模型响应算一轮，达到上限后进入报告生成。
    max_supervisor_iterations: int = Field(
        default=4, ge=1, le=20, description="主管决策轮数上限"
    )
    # 同一轮最多启动多少个独立研究员；Semantic Scholar 请求层仍会串行限速。
    max_concurrent_researchers: int = Field(
        default=3, ge=1, le=10, description="单轮并行研究员上限"
    )
    # 每位研究员最多做多少轮模型判断；一轮可以包含多个工具调用。
    max_researcher_iterations: int = Field(
        default=6, ge=1, le=30, description="单个研究员决策轮数上限"
    )
    # 一位研究员在一次模型响应中最多执行多少个工具调用，超出的调用会收到限额提示。
    max_tool_calls_per_turn: int = Field(
        default=4, ge=1, le=20, description="研究员单轮工具调用上限"
    )
    # 对支持 limit 的文献工具，每次请求最多返回多少条结果。
    max_papers_per_call: int = Field(
        default=10, ge=1, le=100, description="单次文献工具结果条数上限"
    )
    # 单条工具结果进入模型上下文前允许的最大字符数，超出时标明截断。
    max_tool_result_chars: int = Field(
        default=24000, ge=1000, le=200000, description="单条工具结果字符上限"
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None) -> ResearchConfig:
        # 单次运行的 configurable 优先，其次读取同名环境变量，最后使用字段默认值。
        configurable = (config or {}).get("configurable", {})
        values = {
            name: configurable.get(name, os.getenv(f"DCP_RESEARCH_{name.upper()}"))
            for name in cls.model_fields
        }
        return cls(
            **{name: value for name, value in values.items() if value is not None}
        )

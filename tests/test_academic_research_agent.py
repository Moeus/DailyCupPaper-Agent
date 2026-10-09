"""Exercise the nested research graph without external model or search calls."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from academic_research_agent import nodes, researcher_agent
from academic_research_agent.graph import graph
from academic_research_agent.state import ClarificationDecision, ResearchBrief


@tool
async def get_paper_by_key_word(query: str, limit: int = 10) -> list[dict]:
    """Return one known paper for the graph smoke test."""
    assert query == "retrieval augmented generation"
    assert limit <= 10
    return [
        {
            "paperId": "test-paper-1",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "year": 2020,
            "url": "https://example.org/paper-1",
            "abstract": "We combine parametric and non-parametric memory.",
        }
    ]


class FakeModel:
    def __init__(
        self,
        counters: dict[str, int],
        role: str = "plain",
        schema=None,
        parallel: bool = False,
    ):
        self.counters = counters
        self.role = role
        self.schema = schema
        self.parallel = parallel

    def with_structured_output(self, schema, **_kwargs):
        return FakeModel(self.counters, schema=schema, parallel=self.parallel)

    def bind_tools(self, tools):
        role = (
            "supervisor"
            if any(getattr(item, "__name__", "") == "ConductResearch" for item in tools)
            else "researcher"
        )
        return FakeModel(self.counters, role=role, parallel=self.parallel)

    async def ainvoke(self, _messages, config=None):
        if self.schema is ClarificationDecision:
            return ClarificationDecision(need_clarification=False)
        if self.schema is ResearchBrief:
            return ResearchBrief(research_brief="调研检索增强生成的基础论文")
        if self.role == "supervisor":
            turn = self.counters.get("supervisor", 0)
            self.counters["supervisor"] = turn + 1
            if turn == 0:
                calls = [
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": "检索增强生成的基础论文"},
                        "id": "supervisor-call-1",
                    }
                ]
                if self.parallel:
                    calls.append(
                        {
                            "name": "ConductResearch",
                            "args": {"research_topic": "检索增强生成的后续论文"},
                            "id": "supervisor-call-2",
                        }
                    )
                return AIMessage(
                    content="",
                    tool_calls=calls,
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "ResearchComplete", "args": {}, "id": "supervisor-done"}
                ],
            )
        if "研究主题：" in _messages[-1].content:
            assert "https://example.org/paper-1" in _messages[-1].content
            return AIMessage(content="RAG 基础论文：https://example.org/paper-1")
        assert "https://example.org/paper-1" in _messages[-1].content
        return AIMessage(content="文献调研报告：https://example.org/paper-1")


class FakeResearchChatModel(BaseChatModel):
    counters: ClassVar[dict[str, int]] = {}
    calls_per_turn: ClassVar[int] = 1
    repeat_search: ClassVar[bool] = False

    @property
    def _llm_type(self) -> str:
        return "fake_research"

    def bind_tools(self, _tools, **_kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **_kwargs):
        self.counters["researcher"] = self.counters.get("researcher", 0) + 1
        if self.repeat_search or not any(
            isinstance(item, ToolMessage) for item in messages
        ):
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_paper_by_key_word",
                        "args": {
                            "query": "retrieval augmented generation",
                            "limit": 50,
                        },
                        "id": f"researcher-search-{self.counters['researcher']}-{index}",
                    }
                    for index in range(self.calls_per_turn)
                ],
            )
        else:
            response = AIMessage(content="已获得相关论文，完成研究。")
        return ChatResult(generations=[ChatGeneration(message=response)])


def _patch_researcher(monkeypatch, counters):
    monkeypatch.setattr(FakeResearchChatModel, "counters", counters)
    monkeypatch.setattr(FakeResearchChatModel, "calls_per_turn", 1)
    monkeypatch.setattr(FakeResearchChatModel, "repeat_search", False)
    monkeypatch.setattr(
        researcher_agent,
        "build_model",
        lambda _kind: FakeResearchChatModel(),
    )
    monkeypatch.setattr(researcher_agent, "RESEARCH_TOOLS", [get_paper_by_key_word])


def test_academic_graph_runs_nested_research_and_writes_report(monkeypatch):
    counters: dict[str, int] = {}
    monkeypatch.setattr(nodes, "build_model", lambda _kind: FakeModel(counters))
    _patch_researcher(monkeypatch, counters)

    result = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="请调研检索增强生成的基础论文")]},
            config={"configurable": {"max_papers_per_call": 10}},
        )
    )

    assert counters == {"supervisor": 2, "researcher": 2}
    assert result["final_report"] == "文献调研报告：https://example.org/paper-1"
    assert result["messages"][-1].content == result["final_report"]
    assert "https://example.org/paper-1" in result["notes"][0]


def test_parallel_researchers_are_visible_and_joined(monkeypatch):
    counters: dict[str, int] = {}
    monkeypatch.setattr(
        nodes, "build_model", lambda _kind: FakeModel(counters, parallel=True)
    )
    _patch_researcher(monkeypatch, counters)

    result = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="请比较检索增强生成的基础与后续论文")]},
            config={"configurable": {"max_concurrent_researchers": 2}},
        )
    )

    assert counters == {"supervisor": 2, "researcher": 4}
    assert len(result["notes"]) == 2
    expanded = graph.get_graph(xray=True).draw_mermaid()
    assert "researcher_subgraph" in expanded
    assert "researcher_react" in expanded
    assert "compress_research" in expanded


def test_react_researcher_enforces_tool_limit(monkeypatch):
    counters: dict[str, int] = {}
    actual_calls: list[int] = []

    @tool
    async def get_paper_by_key_word(query: str, limit: int = 10) -> str:
        """Return one test paper."""
        assert query == "retrieval augmented generation"
        actual_calls.append(limit)
        return "https://example.org/paper-1"

    _patch_researcher(monkeypatch, counters)
    monkeypatch.setattr(FakeResearchChatModel, "calls_per_turn", 2)
    monkeypatch.setattr(researcher_agent, "RESEARCH_TOOLS", [get_paper_by_key_word])
    result = asyncio.run(
        researcher_agent.researcher_react(
            {
                "research_topic": "检索增强生成",
                "tool_call_id": "supervisor-call-1",
                "researcher_messages": [HumanMessage(content="检索增强生成")],
                "researcher_iterations": 0,
                "research_results": [],
            },
            {"configurable": {"max_tool_calls_per_turn": 1, "max_papers_per_call": 3}},
        )
    )
    observations = [
        message
        for message in result["researcher_messages"]
        if isinstance(message, ToolMessage)
    ]
    assert counters == {"researcher": 2}
    assert actual_calls == [3]
    assert len(observations) == 2
    assert "超过本轮工具调用上限" in observations[1].content


def test_react_researcher_reaches_model_limit_without_recursion_error(monkeypatch):
    counters: dict[str, int] = {}
    _patch_researcher(monkeypatch, counters)
    monkeypatch.setattr(FakeResearchChatModel, "repeat_search", True)
    result = asyncio.run(
        researcher_agent.researcher_react(
            {
                "research_topic": "检索增强生成",
                "tool_call_id": "supervisor-call-1",
                "researcher_messages": [HumanMessage(content="检索增强生成")],
                "researcher_iterations": 0,
                "research_results": [],
            },
            {"configurable": {"max_researcher_iterations": 6}},
        )
    )
    observations = [
        message
        for message in result["researcher_messages"]
        if isinstance(message, ToolMessage)
    ]
    assert counters == {"researcher": 6}
    assert len(observations) == 6
    assert all("https://example.org/paper-1" in item.content for item in observations)
    assert not any(
        "GraphRecursionError" in str(message.content)
        for message in result["researcher_messages"]
    )


def test_react_researcher_keeps_tool_results_on_failure(monkeypatch):
    counters: dict[str, int] = {}
    _patch_researcher(monkeypatch, counters)

    class FailingAgent:
        async def astream(self, input, config, stream_mode):
            assert stream_mode == "values"
            yield {
                "messages": [
                    *input["messages"],
                    ToolMessage(
                        content="https://example.org/paper-1",
                        name="get_paper_by_key_word",
                        tool_call_id="search-1",
                    ),
                ]
            }
            raise RuntimeError("research interrupted")

    monkeypatch.setattr(
        researcher_agent, "create_agent", lambda **_kwargs: FailingAgent()
    )
    result = asyncio.run(
        researcher_agent.researcher_react(
            {
                "research_topic": "检索增强生成",
                "tool_call_id": "supervisor-call-1",
                "researcher_messages": [HumanMessage(content="检索增强生成")],
                "researcher_iterations": 0,
                "research_results": [],
            },
            {},
        )
    )
    assert isinstance(result["researcher_messages"][0], ToolMessage)
    assert "https://example.org/paper-1" in result["researcher_messages"][0].content
    assert "research interrupted" in result["researcher_messages"][-1].content

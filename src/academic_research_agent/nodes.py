"""Graph nodes for clarifying, delegating, researching and reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, Send

from academic_research_agent.config import ResearchConfig
from academic_research_agent.prompts import (
    BRIEF_PROMPT,
    CLARIFY_PROMPT,
    COMPRESS_PROMPT,
    REPORT_PROMPT,
    SUPERVISOR_PROMPT,
)
from academic_research_agent.state import (
    AcademicResearchState,
    ClarificationDecision,
    ConductResearch,
    ResearchBrief,
    ResearchComplete,
    ResearcherState,
    ResearchResult,
    SupervisorState,
)
from academic_research_agent.tools import (
    think_tool,
)
from agent_runtime.models import build_model


def _today() -> str:
    return datetime.now().astimezone().date().isoformat()


async def clarify_with_user(
    state: AcademicResearchState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Only pause for a question when the research subject cannot be inferred."""
    messages = state.get("messages", [])
    if not messages:
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content="请提供希望调研的学术问题或主题。")]
            },
        )
    model = build_model("research").with_structured_output(
        ClarificationDecision, method="function_calling"
    )
    decision = await model.ainvoke(
        [
            HumanMessage(
                content=CLARIFY_PROMPT.format(messages=get_buffer_string(messages))
            )
        ],
        config=config,
    )
    if decision.need_clarification:
        question = decision.question.strip() or "请说明希望调研的学术问题或主题。"
        return Command(goto=END, update={"messages": [AIMessage(content=question)]})
    return Command(goto="write_research_brief")


async def write_research_brief(
    state: AcademicResearchState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """Turn the conversation into one scoped research assignment."""
    messages = state.get("messages", [])
    model = build_model("research").with_structured_output(
        ResearchBrief, method="function_calling"
    )
    brief = await model.ainvoke(
        [
            HumanMessage(
                content=BRIEF_PROMPT.format(
                    date=_today(), messages=get_buffer_string(messages)
                )
            )
        ],
        config=config,
    )
    limits = ResearchConfig.from_runnable_config(config)
    supervisor_prompt = SUPERVISOR_PROMPT.format(
        date=_today(),
        max_concurrent_researchers=limits.max_concurrent_researchers,
        max_supervisor_iterations=limits.max_supervisor_iterations,
    )
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": brief.research_brief,
            "supervisor_messages": [
                SystemMessage(content=supervisor_prompt),
                HumanMessage(content=brief.research_brief),
            ],
        },
    )


async def supervisor(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor_tools", "__end__"]]:
    """Choose the next independent topics or finish the research phase."""
    limits = ResearchConfig.from_runnable_config(config)
    turns = state.get("supervisor_iterations", 0)
    if turns >= limits.max_supervisor_iterations:
        return Command(goto=END)
    model = build_model("research").bind_tools(
        [ConductResearch, ResearchComplete, think_tool]
    )
    response = await model.ainvoke(state["supervisor_messages"], config=config)
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "supervisor_iterations": turns + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["researcher_subgraph", "collect_research", "__end__"]]:
    """Route a bounded batch of research calls to explicit subgraph nodes."""
    calls = state["supervisor_messages"][-1].tool_calls
    if not calls or any(call["name"] == "ResearchComplete" for call in calls):
        return Command(goto=END)

    limits = ResearchConfig.from_runnable_config(config)
    research_calls = [call for call in calls if call["name"] == "ConductResearch"]
    allowed = research_calls[: limits.max_concurrent_researchers]
    assignments: list[Send] = []
    for call in allowed:
        topic = str(call.get("args", {}).get("research_topic", "")).strip()
        if topic:
            assignments.append(
                Send(
                    "researcher_subgraph",
                    {
                        "research_topic": topic,
                        "tool_call_id": call["id"],
                        "researcher_messages": [HumanMessage(content=topic)],
                        "researcher_iterations": 0,
                    },
                )
            )
    return Command(
        goto=assignments or "collect_research",
        update={"pending_research_calls": calls},
    )


async def collect_research(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor"]]:
    """Join parallel researcher results and return observations to the supervisor."""
    calls = state.get("pending_research_calls", [])
    limits = ResearchConfig.from_runnable_config(config)
    allowed_ids = {
        call["id"]
        for call in [item for item in calls if item["name"] == "ConductResearch"][
            : limits.max_concurrent_researchers
        ]
    }
    by_id: dict[str, ResearchResult] = {
        result["tool_call_id"]: result
        for result in state.get("research_results", [])
        if result["tool_call_id"] in allowed_ids
    }
    tool_messages: list[ToolMessage] = []
    notes: list[str] = []
    raw_notes: list[str] = []
    for call in calls:
        name = call["name"]
        if name == "think_tool":
            content = f"Research reflection recorded: {call.get('args', {}).get('reflection', '')}"
        elif name == "ConductResearch" and call["id"] in by_id:
            result = by_id[call["id"]]
            content = result["compressed_research"]
            if result["raw_notes"]:
                notes.append(content)
            raw_notes.extend(result["raw_notes"])
        elif name == "ConductResearch" and call["id"] in allowed_ids:
            content = "研究任务缺少主题或执行失败；本次未获得文献材料。"
        elif name == "ConductResearch":
            content = (
                f"超过本轮 {limits.max_concurrent_researchers} 个研究员的并发上限，"
                "此任务未执行；请减少并行任务数。"
            )
        else:
            content = f"未知主管工具：{name}"
        tool_messages.append(
            ToolMessage(content=content, name=name, tool_call_id=call["id"])
        )
    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": tool_messages,
            "notes": notes,
            "raw_notes": raw_notes,
        },
    )


async def compress_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Convert tool observations into a sourced, portable research note."""
    observations = [
        str(message.content)
        for message in state.get("researcher_messages", [])
        if isinstance(message, ToolMessage)
        and message.name not in {"think_tool", "ResearchComplete"}
    ]
    raw = "\n\n".join(observations)
    if not raw.strip():
        return {
            "research_results": [
                {
                    "tool_call_id": state["tool_call_id"],
                    "compressed_research": "没有获得可核查的文献材料。",
                    "raw_notes": [],
                }
            ]
        }
    # The model needs a bounded context even when many tool calls returned abstracts.
    bounded = raw[:150000]
    if len(raw) > len(bounded):
        bounded += "\n[材料超过上下文预算，后续内容省略。]"
    try:
        response = await build_model("research").ainvoke(
            [
                SystemMessage(content=COMPRESS_PROMPT),
                HumanMessage(
                    content=f"研究主题：{state['research_topic']}\n\n材料：\n{bounded}"
                ),
            ],
            config=config,
        )
        compressed = str(response.content)
    except Exception as exc:  # noqa: BLE001 - preserve raw notes if compression fails
        compressed = (
            f"文献整理失败（{type(exc).__name__}）。以下是原始检索材料：\n{bounded}"
        )
    return {
        "research_results": [
            {
                "tool_call_id": state["tool_call_id"],
                "compressed_research": compressed,
                "raw_notes": [raw],
            }
        ]
    }


async def final_report_generation(
    state: AcademicResearchState, config: RunnableConfig
) -> dict:
    """Write the user-facing report from the supervisor's sourced notes."""
    notes = state.get("notes", [])
    if not notes:
        report = "本次未能获得可核查的文献材料，因此无法生成可靠的文献调研报告。请检查检索工具配置或缩小研究范围后重试。"
    else:
        prompt = REPORT_PROMPT.format(
            research_brief=state.get("research_brief", ""),
            messages=get_buffer_string(state.get("messages", [])),
            findings="\n\n".join(notes)[:220000],
        )
        response = await build_model("research").ainvoke(
            [HumanMessage(content=prompt)], config=config
        )
        report = str(response.content)
    return {"final_report": report, "messages": [AIMessage(content=report)]}

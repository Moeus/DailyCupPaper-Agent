"""The literature researcher's ReAct agent and its tool boundaries."""

from __future__ import annotations

from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallRequest,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from academic_research_agent.config import ResearchConfig
from academic_research_agent.prompts import RESEARCHER_PROMPT
from academic_research_agent.state import ResearcherState
from academic_research_agent.tools import RESEARCH_TOOLS, format_tool_result
from agent_runtime.models import build_model


def _tool_limits(limits: ResearchConfig):
    @wrap_tool_call
    async def enforce_limits(request: ToolCallRequest, handler):
        call = request.tool_call
        name = call["name"]
        messages = request.state.get("messages", [])
        latest = messages[-1] if messages else None
        calls = latest.tool_calls if isinstance(latest, AIMessage) else []
        index = next(
            (i for i, item in enumerate(calls) if item["id"] == call["id"]),
            0,
        )
        if index >= limits.max_tool_calls_per_turn:
            return ToolMessage(
                content="超过本轮工具调用上限，此调用未执行。",
                name=name,
                tool_call_id=call["id"],
                status="error",
            )

        args = dict(call.get("args") or {})
        if "limit" in request.tool.args:
            try:
                args["limit"] = min(
                    max(1, int(args.get("limit", limits.max_papers_per_call))),
                    limits.max_papers_per_call,
                )
            except (TypeError, ValueError):
                args["limit"] = limits.max_papers_per_call
        try:
            result = await handler(request.override(tool_call={**call, "args": args}))
        except Exception as exc:  # noqa: BLE001 - expose a failed search to the agent
            return ToolMessage(
                content=(
                    f"工具 {name} 执行失败（{type(exc).__name__}）：{exc}。"
                    "这不代表没有相关文献。"
                ),
                name=name,
                tool_call_id=call["id"],
                status="error",
            )
        if isinstance(result, ToolMessage):
            result.content = format_tool_result(
                result.content, limits.max_tool_result_chars
            )
        return result

    return enforce_limits


async def researcher_react(state: ResearcherState, config: RunnableConfig) -> dict:
    """Run one self-contained ReAct agent for the assigned research topic."""
    limits = ResearchConfig.from_runnable_config(config)
    previous = state["researcher_messages"]
    latest_messages = previous
    child_config = {**config}
    # 研究员内部图的步骤上限；模型调用次数仍由中间件单独限制。
    child_config["recursion_limit"] = 10000
    try:
        agent = create_agent(
            model=build_model("research"),
            tools=RESEARCH_TOOLS,
            system_prompt=RESEARCHER_PROMPT.format(
                date=datetime.now().astimezone().date().isoformat(),
                max_tool_calls_per_turn=limits.max_tool_calls_per_turn,
            ),
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=limits.max_researcher_iterations, exit_behavior="end"
                ),
                _tool_limits(limits),
            ],
            name="literature_react_agent",
        )
        async for snapshot in agent.astream(
            {"messages": previous}, config=child_config, stream_mode="values"
        ):
            latest_messages = snapshot["messages"]
        return {"researcher_messages": latest_messages[len(previous) :]}
    except Exception as exc:  # noqa: BLE001 - isolate one failed researcher
        return {
            "researcher_messages": [
                *latest_messages[len(previous) :],
                AIMessage(content=f"研究模型调用失败（{type(exc).__name__}）：{exc}"),
            ]
        }

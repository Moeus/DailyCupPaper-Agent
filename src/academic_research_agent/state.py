"""State contracts and structured model outputs for the research graph."""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ClarificationDecision(BaseModel):
    need_clarification: bool
    question: str = ""


class ResearchBrief(BaseModel):
    research_brief: str = Field(min_length=1)


class ConductResearch(BaseModel):
    """Delegate one independent academic literature research topic."""

    research_topic: str = Field(
        description="A standalone, specific literature research task with scope, key terms and expected evidence."
    )


class ResearchComplete(BaseModel):
    """Signal that enough literature has been gathered to write the report."""


class AcademicResearchState(MessagesState):
    """Outer graph state. MessagesState also supplies the public input contract."""

    research_brief: str
    supervisor_messages: Annotated[list[AnyMessage], operator.add]
    notes: Annotated[list[str], operator.add]
    raw_notes: Annotated[list[str], operator.add]
    final_report: str


class ResearchResult(TypedDict):
    tool_call_id: str
    compressed_research: str
    raw_notes: list[str]


class SupervisorState(TypedDict):
    research_brief: str
    supervisor_messages: Annotated[list[AnyMessage], operator.add]
    notes: Annotated[list[str], operator.add]
    raw_notes: Annotated[list[str], operator.add]
    supervisor_iterations: int
    pending_research_calls: list[dict]
    research_results: Annotated[list[ResearchResult], operator.add]


class ResearcherState(TypedDict):
    research_topic: str
    tool_call_id: str
    researcher_messages: Annotated[list[AnyMessage], operator.add]
    researcher_iterations: int
    research_results: Annotated[list[ResearchResult], operator.add]


class ResearcherOutput(TypedDict):
    research_results: Annotated[list[ResearchResult], operator.add]

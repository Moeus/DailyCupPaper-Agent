"""Nested LangGraph workflow for academic literature research."""

from langgraph.graph import END, START, MessagesState, StateGraph

from academic_research_agent.nodes import (
    clarify_with_user,
    collect_research,
    compress_research,
    final_report_generation,
    supervisor,
    supervisor_tools,
    write_research_brief,
)
from academic_research_agent.researcher_agent import researcher_react
from academic_research_agent.state import (
    AcademicResearchState,
    ResearcherOutput,
    ResearcherState,
    SupervisorState,
)

researcher_builder = StateGraph(ResearcherState, output_schema=ResearcherOutput)
researcher_builder.add_node("researcher_react", researcher_react)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher_react")
researcher_builder.add_edge("researcher_react", "compress_research")
researcher_builder.add_edge("compress_research", END)
researcher_graph = researcher_builder.compile()


supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("researcher_subgraph", researcher_graph)
supervisor_builder.add_node("collect_research", collect_research)
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("researcher_subgraph", "collect_research")
supervisor_graph = supervisor_builder.compile()


builder = StateGraph(AcademicResearchState, input_schema=MessagesState)
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("write_research_brief", write_research_brief)
builder.add_node("research_supervisor", supervisor_graph)
builder.add_node("final_report_generation", final_report_generation)
builder.add_edge(START, "clarify_with_user")
builder.add_edge("research_supervisor", "final_report_generation")
builder.add_edge("final_report_generation", END)
graph = builder.compile()

"""Task-oriented tools exposed to the research agent."""

from tools.research import paper_tools
from tools.web_search import web_search

research_tools = [*paper_tools, web_search]

__all__ = ["research_tools"]

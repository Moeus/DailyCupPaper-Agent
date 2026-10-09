# 学术文献调研 Agent

`langgraph.json` 和 `langgraph.local.json` 注册了 `academic_research` 与 `translation` 两张图。学术调研图的入口为 `academic_research_agent.graph:graph`，输入为 LangGraph `MessagesState`，即 `{"messages": [{"role": "user", "content": "..."}]}`。

## 工作流程

1. `clarify_with_user`：仅在研究主题无法判断时提问并结束当前运行。
2. `write_research_brief`：将对话整理为独立的研究任务，保留用户指定的范围与语言。
3. `research_supervisor`：主管选择子题，`supervisor_tools` 使用 LangGraph `Send` 派发到显式的 `researcher_subgraph` 节点；`collect_research` 汇总并行结果，再交主管决定继续或结束。
4. 每个研究员由 `researcher_react` 节点中的 ReAct agent 自行循环调用文献工具，完成或达到模型轮数上限后由 `compress_research` 整理材料。
5. `final_report_generation`：根据各研究员的资料撰写报告，并将文本写入 `final_report` 和最后一条 `messages`。

在 Studio 中展开 `research_supervisor`，可继续展开 `researcher_subgraph` 查看 `researcher_react → compress_research`。在 Python 中使用 `graph.get_graph(xray=True)` 可取得包含两层子图的完整静态图。各节点职责、状态合并与 `Send` 调度详见 [工作流说明](academic-research-workflow.md)。

主管只使用 `think_tool`、`ConductResearch` 和 `ResearchComplete`。研究员可使用 `src/tools/__init__.py` 中的全部 `research_tools`：8 个 Semantic Scholar 文献工具和 Tavily `web_search`，另有 `think_tool`。论文工具的参数及返回格式以 `src/tools/research.py` 的 docstring 为准。报告主要依据论文元数据、摘要和片段，不声称已阅读全文。

## 配置

在 `.env` 中填写 `DCP_RESEARCH_AGENT_API_KEY`、`SEMANTIC_SCHOLAR_API_KEY`；需要网页搜索时再填写 `TAVILY_API_KEY`。模型配置沿用 `src/agent_runtime/models.py` 的 `DCP_RESEARCH_AGENT_*` 变量。研究图预算见 `.example.env` 的中文注释和 `src/academic_research_agent/config.py`。

预算配置的优先级：单次运行的 `config.configurable` > 同名 `DCP_RESEARCH_*` 环境变量 > 默认值。例如 `configurable.max_concurrent_researchers=1` 可限制一次运行只派一个研究员。`max_supervisor_iterations` 和 `max_researcher_iterations` 统计的是模型决策轮数，不是单个查询数量。

Semantic Scholar 的现有请求层将论文 API 请求串行执行，遇到限流会退避重试。并行研究员可以并发思考与整理，但不会提高该 API 的请求吞吐量。

## 本地启动

```powershell
.venv\Scripts\langgraph.exe dev --config langgraph.local.json --no-browser --port 2024
```

启动后访问 `http://127.0.0.1:2024/docs`，或在 `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024` 选择 `academic_research`。本地 `dev` 服务是内存模式；生产部署配置使用 `langgraph.json`。

项目测试：

```powershell
.venv\Scripts\python.exe -m pytest -q tests
```

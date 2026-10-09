# 学术文献调研 Agent：节点、状态与调度

入口是 `academic_research_agent.graph:graph`，由 `langgraph.json` 和 `langgraph.local.json` 中的 `academic_research` 注册。外层图的输入 schema 是 `MessagesState`：调用方只需传入 `messages`。运行中的状态是继承 `MessagesState` 的 `AcademicResearchState`。

## 图的结构

```text
START
  → clarify_with_user
  → write_research_brief
  → research_supervisor（主管子图）
       supervisor ↔ supervisor_tools
                        │ Send(每个独立子题)
                        ▼
                 researcher_subgraph（研究员子图，可并行运行多份）
                   researcher_react → compress_research
                        │ 研究结果按 reducer 合并
                        ▼
                   collect_research → supervisor
       主管决定结束 → END
  → final_report_generation → END
```

`clarify_with_user` 也可能直接走 `END`，此时只返回澄清问题。图中的主管子图和研究员子图是显式注册的；研究员内部的 `create_agent` 自己执行“模型决定 → 工具执行 → 模型再决定”的 ReAct 循环。静态图展开到 `researcher_react` 节点，内部模型和工具调用可以在运行轨迹中查看。

## 每个节点的工作

| 节点 | 输入重点 | 工作与输出 |
| --- | --- | --- |
| `clarify_with_user` | `messages` | 判断用户是否给出了可识别的研究主题。若完全无法判断，追加澄清问题并结束；否则转到 brief。 |
| `write_research_brief` | `messages` | 把对话整理成 `research_brief`，保留范围、时间、指定论文与语言；初始化 `supervisor_messages`，转入主管子图。 |
| `supervisor` | `supervisor_messages`、`supervisor_iterations` | 一次模型调用，选择 `ConductResearch`、`think_tool` 或 `ResearchComplete`；将模型响应追加到主管消息，并增加轮数。达到轮数上限时结束主管子图。 |
| `supervisor_tools` | 主管最后一条消息的 `tool_calls` | 对每个允许的 `ConductResearch` 构造一个 `Send("researcher_subgraph", ...)`；保存 `pending_research_calls`。若没有可执行研究任务，则直接转到汇总；调用 `ResearchComplete` 或没有工具调用时结束。 |
| `researcher_subgraph` | 单个 `research_topic`、`tool_call_id`、`researcher_messages` | 每份 `Send` 启动一份独立研究员子图；仅向主管返回 `research_results`。 |
| `researcher_react` | 该子题的 `researcher_messages` | 用 `create_agent` 运行 ReAct 循环，按需要调用文献搜索、证据、网页和思考工具；把新增的模型消息和工具结果追加到研究员消息。 |
| `compress_research` | 研究员的工具结果 | 排除 `think_tool` 记录，整理可核查论文、证据与局限；产出带原主管工具调用 ID 的 `ResearchResult`。没有可用工具材料时明确返回无材料。 |
| `collect_research` | `pending_research_calls`、合并后的 `research_results` | 按 `tool_call_id` 将每份研究结果对应回主管的 `ConductResearch` 调用，生成 `ToolMessage`，把整理内容加入 `notes`、原始材料加入 `raw_notes`，再返回 `supervisor`。 |
| `final_report_generation` | `research_brief`、`messages`、`notes` | 按已检索证据写报告；把报告写到 `final_report` 并追加为最后一条对话消息。没有研究材料时返回明确的材料不足说明。 |

主管仍是手写的 `supervisor` 与 `supervisor_tools` 两个节点；本次只有研究员改为 ReAct agent。

## 状态如何流转

1. **外层状态**：`AcademicResearchState(MessagesState)` 保存用户对话 `messages`、`research_brief`、主管消息、`notes`、`raw_notes` 和 `final_report`。外层公开输入仍是 `MessagesState`。
2. **主管状态**：`SupervisorState` 读取外层传入的 brief 和主管消息，维护 `supervisor_iterations`、`pending_research_calls` 与 `research_results`。`supervisor_messages`、`notes`、`raw_notes`、`research_results` 都有列表追加 reducer。
3. **研究员状态**：每个 `Send` 传入自己的主题、主管工具调用 ID、初始 `HumanMessage`。研究员在自己的 `researcher_messages` 中保存 ReAct 对话；`compress_research` 只向父图输出 `research_results`，因此不同研究员的内部消息不会混进主管对话。原有的 `researcher_iterations=0` 仍随分派状态传入，但 ReAct 模型轮数实际由 `ModelCallLimitMiddleware` 控制。
4. **并行合并**：同一批 `Send` 的研究员各自产生一条 `ResearchResult`，父图通过 `research_results` 的列表追加 reducer 合并它们。所有已派发的研究员子图在该批完成后，`collect_research` 读取合并后的列表；它按 `tool_call_id` 配对，不能依赖结果返回顺序。
5. **反馈闭环**：汇总节点把研究结果封装成与主管工具调用对应的 `ToolMessage`。主管下一轮看到这些观察结果，决定补充子题或调用 `ResearchComplete`。主管子图结束后，外层报告节点读取积累的 `notes`。

`Send` 是 LangGraph 的动态分派指令：每个 `Send` 指定目标节点和该次调用的输入状态。它不要求预先在静态图中画出固定数量的研究员，也不只是向节点发送一条字符串消息。当前每轮最多派发 `max_concurrent_researchers` 个有效主题；超额调用由汇总节点回复上限提示。

## 研究员的 ReAct 工具

研究员可用的工具由 `src/academic_research_agent/tools.py` 组装：

| 工具 | 用途 |
| --- | --- |
| `get_paper_by_key_word` | 关键词、同义词、年份或来源检索论文。 |
| `get_paper_by_title` | 按完整或近似标题核实论文。 |
| `get_paper_by_id` | 按 paper ID、DOI、arXiv ID 等取论文详情。 |
| `get_paper_citations` | 找引用某篇论文的后续工作。 |
| `get_paper_references` | 找某篇论文引用的早期工作。 |
| `find_paper_evidence` | 按问题或论点查论文文本片段及来源。 |
| `get_related_papers` | 根据一篇核心论文扩展相近工作。 |
| `recommend_papers_by_examples` | 用相关与不相关的种子论文推荐候选工作。 |
| `web_search` | 补充论文入口或核查公开网页，使用 Tavily。 |
| `think_tool` | 记录检索缺口和下一步思考；其内容不作为论文证据。 |

前八个论文工具使用 Semantic Scholar；具体参数和返回字段见 `src/tools/research.py`。研究员提示词要求优先核查论文元数据、摘要和片段，不把片段说成全文，也不凭标题编造实验结论。`ResearchComplete` 是主管结束研究阶段的工具，研究员不使用它；研究员认为证据足够时以普通最终回复结束 ReAct 循环。

研究员限额在 `src/academic_research_agent/config.py`：`max_researcher_iterations` 限制单个 ReAct 运行中的模型决策次数；`max_tool_calls_per_turn` 限制一次模型响应实际执行的工具数；带 `limit` 参数的工具会按 `max_papers_per_call` 截顶；单条工具结果按 `max_tool_result_chars` 截断。工具报错会作为错误观察返回给研究员，不能据此断言没有相关文献。

## 代码位置

- `src/academic_research_agent/graph.py`：三层图的注册与边。
- `src/academic_research_agent/state.py`：输入、子图状态和结构化输出。
- `src/academic_research_agent/nodes.py`：澄清、brief、主管、汇总、压缩和报告节点。
- `src/academic_research_agent/researcher_agent.py`：研究员 ReAct agent 与工具调用中间件。
- `src/academic_research_agent/prompts.py`：各角色的领域提示词。
- `src/academic_research_agent/config.py`：运行限额与配置来源。
- `docs/langgraph-server-tools.md`：本地启动与服务配置。

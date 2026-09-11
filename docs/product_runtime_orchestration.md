# 产品运行时编排：P2.8.4 收口记录

## 目标

P2.8.1-C 的目标不是重跑 E5/P2 实验，也不是一次删除历史页面。它解决的是产品入口的一个一致性问题：同一个用户问题在 Streamlit 和 API 中应先经过同一套、可解释且不泄漏评测 Gold 的任务判断，再决定是否检索、检索哪个论文范围，或明确拒绝。

本轮新增：

```text
src/researchpilot/routing.py
src/researchpilot/orchestration.py
tests/test_routing.py
tests/test_orchestration.py
```

## 运行时输入边界

`RuntimeRequest` 只包含：

```text
question
user_id / session_id
available_papers
selected_paper_ids
recent_history
product_mode
last_evidence_chunk_ids
```

它不接受 `gold_evidence`、`expected_answer`、`required_paper_keys` 或评测 `query_type`。因此产品路由不会通过评测答案得知“应检索哪篇论文”。P2 的 `RuntimeRequest` / `QueryPlan` 继续服务于实验；产品的 `RuntimeRequest` 位于 `routing.py`，两者刻意分开命名空间，避免把 Oracle 标注混入产品调用。

## product_rules_v2

路由器不调用 LLM，按以下顺序做保守判断：

| 用户问题特征 | Route | 产品行为 |
| --- | --- | --- |
| 简单问候、能力介绍 | `general_chat` | 不进行论文检索，不伪造 Citation |
| 总结我们刚才的讨论 | `chat_summary` | 不检索论文；由拥有会话历史的入口处理 |
| 未上传论文却提论文问题 | `scope_insufficient` | 结构化拒答，提示选择或上传论文 |
| 明确说“两篇/多篇/跨论文比较”，但 Scope 少于两篇 | `scope_insufficient` | 拒绝强行比较 |
| 比较/对比且 Scope 至少两篇 | `cross_document` | 当前 Feature Flag 下明确拒绝，不退化为全局 Top-K |
| 单篇论文中作者自己的 Baseline / 方法比较 | `experimental_results` 或 `document_detail` | 继续在该论文内查找结果，不误判成跨论文 |
| 作者、年份、会议、DOI 等 | `metadata` | 检索第 1--2 页 Front Matter，返回页码 Citation |
| 全文总结、整体介绍 | `document_summary` | 按摘要/方法/实验/结论章节覆盖后生成 |
| 实验设置、数据集、基线、指标、训练配置 | `experimental_setup` | 强制覆盖实验章节并用双语 Hybrid 补充 |
| 方法与实验、原因与影响等跨节问题 | `multi_section` | 进入可追溯 RAG |
| 其他论文问题 | `document_detail` | 进入当前 Profile 的 RAG |

`selected_paper_ids` 优先于“当前会话全部论文”。单论文 Scope 会被一路传至 Dense、BM25、RRF、Reranker 和 API；多论文比较不会因当前产品尚未接好 P2 Gate 而偷偷放宽范围。

`last_evidence_chunk_ids` 只用于用户明确说“我的批注 / 这里有问题”等情况下，把笔记绑定到上一轮可见证据；普通问题不会触发批注写入，也不增加模型分类调用。

## 两个入口如何使用

### FastAPI

`POST /v1/chat` 先从同一 `user_id + session_id` 的 `DocumentRegistry` 读取论文目录，构造 `RuntimeRequest`，然后调用：

```python
RuntimeOrchestrator(...).handle(runtime_request)
```

响应包含 `route` 字段。`general_chat`、`scope_insufficient` 和未开放的 `cross_document` 返回不带论文 Citation 的本地安全结果；`metadata`、`document_summary`、`experimental_setup`、`document_detail` 与 `multi_section` 均调用 `ResearchPilotService.answer_with_retrieval(...)`，并携带对应 `QueryType`。

真实 `RuntimeServiceProvider` 会把同一运行时中的 `ConversationService`、`ScopedMemoryService` 与 `AnnotationService` 一并交给编排器；测试替身仍可只提供最小 `service.get()` 契约。这样 API 与 Streamlit 共用 Router、Memory Policy 和 Answer Contract，同时保持 `/health` 惰性加载模型。

### Streamlit

对于新上传、带 `paper_id` 的结构化索引，页面把用户问题、当前论文目录、显式选择和上一轮证据构造成 `RuntimeRequest`。所有论文任务随后进入同一 `ResearchPilotService`；页面不再读取无引用的 `doc_meta/doc_summary` 作为新版答案，也不再直接拼论文 Prompt、调用 Chroma/BM25/Reranker 或执行 `llm.invoke`。

历史 H0 会话缺少 `paper_id` / 结构化文档元数据时，`LegacySessionAdapter` 将其置为只读：保留历史消息，但新的论文问题只提示重新上传，不再运行旧 LLM A/B/C/D/E 分类器或旧 Hybrid。新版 Service 自身失败时同样不会回退，而是返回可恢复提示。legacy 是历史可读性边界，不是默认产品链路。

## 当前明确不宣称的能力

1. API 的独立文档目录持久化文件名、页数、Parser 和 Chunker；作者、会议等由 Front Matter 证据按需生成并进入版本化 `PaperMetadata` 缓存，但尚未建设 BibTeX 导入/导出和人工字段校正工作流。
2. `strict_crossdoc` 虽保留 P2 的实验结果和 API Feature Flag，但尚未将 P2 Coverage Gate 接到通用上传会话，所以仍返回明确拒绝。
3. Claim-Citation v1 是确定性范围和数字出现性校验，不是完整自然语言蕴含证明；人工语义复核仍是 `product-rc2` 的发布门禁。
4. Streamlit 仍管理页面状态、按钮和保存触发；这属于 UI 适配职责，不等于把业务 Prompt 放回页面。
5. 本轮没有修改任何冻结的 E5/P2 Dev/Test 配置、数据或实验报告。

## 回归验收

自动测试覆盖：

- general chat 不触发论文检索；
- metadata 使用 Front Matter 检索并返回 Citation；
- 全文总结固定覆盖关键章节，实验设置优先覆盖数据集、Baseline、指标与实现配置；
- 单论文 `paper_id` 会传入 Core；
- Scope 缺失、单论文比较、多论文比较分别得到显式安全结果；
- API 响应和本地 Trace 都记录 Route；
- Streamlit 新索引使用共享规则与 Service，旧索引保留受限兼容分支；
- Runtime 输入对象无法接收评测 Gold 字段。

人工冒烟将在本轮全量自动测试通过后执行：上传一篇新文本型 PDF，依次问“作者是谁”“核心方法是什么”“总结这篇论文”，再上传第二篇论文并询问比较问题，确认最后一项显示产品级拒绝而非混合引用。

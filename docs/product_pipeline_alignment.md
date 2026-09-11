# 产品链路对齐记录

## 目的

本记录用于证明用户实际运行的 Streamlit/API 与离线 E5 结构化检索实验之间的配置关系。它不是新的评测报告，不修改已冻结的 E5、P1-E 或 P2 Test 结果。

## 对齐前的真实状态

截至 `product-rc2`，产品上传与检索链路为：

```text
pdfplumber + page_parent_child_v1
Query Expansion + Dense + Persistent BM25 + Set Union + BGE Reranker
```

因此，不能把 E5 的 `Docling + structure_aware + RRF + Reranker` 结果直接标为当时 Streamlit/API 的默认能力。

## 本轮已完成的对齐

产品运行时改为读取版本化 YAML Profile：

```text
configs/product/
├─ standard.yaml
├─ high_precision.yaml
├─ full_paper.yaml
└─ strict_crossdoc.yaml
```

默认产品 Profile 是 `high_precision`，其当前真实链路为：

```text
Docling 2.120.2
  ↓ 失败时记录原因并回退
pdfplumber 0.11.9
  ↓
structure_aware_v1 Parent–Child Chunking
  ↓
Dense + Persistent BM25 + RRF(k=60)
  ↓
BGE Reranker
  ↓
带页码 Citation / Retrieval Trace 的 Answer
```

Streamlit 通过 `build_service_from_components(...)`，FastAPI 通过 `build_product_runtime(...)` 装配 `ResearchPilotService`。二者读取相同的 `high_precision.yaml`，不再各自硬编码 `fusion="set_union"`。

产品体验 MVP 已在此基础上提供“精确问答 / 快速问答”与单论文范围选择；两种模式都读取相同的结构化 Parser/Chunker 与 RRF 配置，区别仅在是否运行 Reranker。

P2.8.3 进一步完成论文阅读任务对齐：新版论文的作者/元信息、全文总结、实验设置和正文细节统一进入 `ResearchPilotService`。产品运行时在冻结 Hybrid Retriever 上增加 `ProductTaskRetrievalService`，分别执行 Front Matter、章节均衡和实验章节强制覆盖策略。上传阶段不再调用 DeepSeek；元信息和摘要改为带页码证据的按需生成。完整记录见 `docs/p2_8_3_product_quality.md`。

P2.8.4 将产品入口进一步收口到 `product_rules_v2 + RuntimeOrchestrator`：Query Rewrite、通用聊天、论文级工作记忆、长期结论整理、用户批注和 Answer Contract 均位于后端服务；Streamlit 不再直接调用 Chroma、BM25、Reranker 或 `llm.invoke`。`product_rules_v2` 还修复了“单篇论文内部与 Baseline 比较”被误判为跨论文比较的问题。

P2.8.5 已接通最短研究闭环：Citation 可打开本地 PDF 对应页，回答可保存为独立研究笔记并导出带页码 / Chunk ID 的 Markdown，会话标题由“论文名 + 首个任务类型”确定性生成，删除论文前展示关联资产并二次确认。

## 回退与索引边界

- 仅当主 Parser 抛出解析错误时才尝试 `pdfplumber`；低文本/扫描 PDF 若仍无可用文本，继续拒绝建索引；
- `IngestionResult` 和 API 文档元数据记录 requested/actual Parser、fallback 状态、解析质量、Chunker、Profile 与配置哈希；
- 同一 `paper_id` 在当前 user/session 重新成功解析后，旧 Parent/Child 会先被删除，再写入新版本，避免旧 H0 Chunk 与新 E5 Chunk 在一次检索中混合；
- 失败重建不会删除旧索引，因为删除发生在新解析成功之后；
- 仍未建立“跨会话论文库”或完整的产品索引 Manifest；这属于后续产品体验阶段，不应把当前会话级元数据宣传为论文库。

## 当前边界

链路、可信回答、研究工作流和 `product-rc2` 真实验收已经完成，但以下事项仍不能提前宣称：

1. `LegacySessionAdapter` 仍保留旧会话只读兼容，但旧页面 Hybrid 主循环已经移除；新版失败不会静默回退旧检索；
2. P2 严格跨论文模式继续默认关闭，不能降级为普通全局 Top-K；
3. Claim-Citation v1 能验证引用编号范围、实际使用证据、结构化表格数值与比较方向，但尚未验证任意自然语言 Claim 的完整语义蕴含或因果关系；
4. Streamlit 尚未具备跨会话论文库、任意多论文研究空间、PDF 逐句高亮或结构化比较工作台；
5. 18 条可信回答回归已真实运行并通过；该结果只覆盖固定两篇论文和 18 个产品用例，不应外推为任意论文回答正确率 100%；
6. 没有重新运行任何冻结 E5/P2 Test，也没有把产品回归当作新的算法指标。

## 本轮自动验证

`product-rc2` 的确定性自动检查结果为：

```text
274 passed
```

覆盖 Profile 加载/校验、E5/RRF 配置、Reranker 模式边界、Parser 回退记录、API 文档契约、论文范围传递、任务感知检索、模型服务 503 降级、会话文档元数据、Citation 归一化与缓存幂等、结构化表格、摘要结构、全文字段扫描、论文级记忆与批注隔离、PDF 页渲染、研究笔记、费用 Trace 与 Streamlit 薄前端边界。共享路由与统一编排见 `docs/product_runtime_orchestration.md`；最终真实复核见 `docs/p2_8_4_d_product_rc2_review.md`。

全仓已经通过 `ruff format --check .` 与 `ruff check .`；编译、依赖和密钥扫描也已通过。真实产品回归预期页覆盖为 36/42（85.7%），该数值继续作为检索泛化边界保留。

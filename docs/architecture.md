# ResearchPilot 架构说明

## 运行时分层

```text
Streamlit（研究工作台）            FastAPI（P2.8 服务入口）
             │                           │
             └──── RuntimeOrchestrator ──┘
                    product_rules_v2
                      │      │
      Conversation / Scoped Memory / Annotation
                      │
              ResearchPilotService ────────────────┐
                                                   │
 Ingestion → Parser / Chunker → VectorIndex + BM25 │
                                                   │
 Retrieval → Task Evidence Selection               │
             + Dense + BM25 + RRF + Rerank         │
                                                   │
 Generation → Evidence Prompt → Answer              │
              → Claim/Citation + Numeric Contract ──┘

Storage: Chroma / BM25 / SQLite scoped memory / annotations / notes / session JSON / PDF copy
Observability: OpenTelemetry Span + redacted JSONL trace
Evaluation: frozen corpus / JSONL / deterministic metrics / answer protocol
```

## 关键边界

FastAPI 的文档导入、检索和问答全部通过 `RuntimeOrchestrator` / `ResearchPilotService`。Streamlit 对新版结构化论文的上传、元信息、总结、实验设置和正文问答也使用同一 Service；页面不再直接解析 PDF、拼论文 Prompt、执行 BM25/Reranker 或调用 Chroma，也不会在新版 Service 失败时静默切换算法。

通用聊天、Query Rewrite、论文级工作记忆、长期结论整理与用户批注捕捉已迁入 Conversation / Memory / Annotation Service。Streamlit 仍负责交互状态、触发保存和展示结果，但不拥有上述 Prompt 与存储规则。旧 Demo 会话由 `LegacySessionAdapter` 判定为只读：可以查看历史消息，但不能执行旧 Hybrid 检索，必须重新上传论文才进入新版页码引用链路。

所有产品数据带 `user_id + session_id`；Chroma 查询、会话 JSON、SQLite 记忆、BM25 Scope 以及 API Upload Registry 都必须使用同一 Scope。评测语料使用独立评测索引，不能与浏览器/API 会话混用。

## 典型 API 流程

```text
POST /v1/documents
  → PDF bytes
  → ResearchPilotService.ingest_pdf()
  → parent/child chunks 写入 scoped Chroma
  → BM25 scope invalidated
  → data/runtime/uploads/<user>/<session>/<paper>.{pdf,json}

POST /v1/chat
  → Scope validation
  → product_rules_v2 task routing
  → same-scope Query Rewrite / Memory Policy
  → ProductTaskRetrievalService / frozen Hybrid Retriever
  → ResearchPilotService.answer_with_retrieval()
  → Answer / Claim / Citation / numeric-fidelity status / RetrievalTrace
  → redacted API trace
```

显式删除论文的顺序为向量索引（父/子块）→ BM25 缓存失效 → 论文级 Metadata / Summary 缓存 → 同论文 Scope 的工作记忆 / 批注 / 研究笔记 / 长期结论 → 上传文件与元数据；删除接口不会触碰其他用户、会话或论文。重新上传同一论文只失效可再生成的索引与缓存，不会静默清空用户研究笔记。

## P2 的运行时隔离

P2 的 Gold Annotation（`required_paper_keys`、答案可答性、Gold Evidence）只被评测器读取，不能传入端到端 Planner。Oracle、Scoped、End-to-End 的差别必须由 Trace 中的来源字段标记。P2.8 默认 API 暂不把 strict cross-document 模式伪装成普通聊天：没有显式论文目录和 Coverage Gate 时会拒绝启用。

# ResearchPilot 核心代码包

公开版本采用分层核心，Streamlit 与 FastAPI 共用同一运行时编排和业务服务。

## 模块分组

- `parsing.py`、`chunking.py`、`ingestion.py`：PDF 结构解析、质量检测与 Parent–Child 分块；
- `index.py`、`sparse.py`、`retrieval.py`、`product_retrieval.py`：Chroma、持久化 BM25、Dense/RRF/Rerank 和任务感知证据选择；
- `routing.py`、`orchestration.py`、`services.py`：产品路由、Scope 边界和统一业务入口；
- `generation.py`、`answer_contract.py`、`claim_verification.py`、`table_evidence.py`、`summary_validation.py`：证据约束生成、引用/数字/表格/摘要校验；
- `memory_service.py`、`annotation_service.py`、`research_workflow.py`、`storage.py`：论文级记忆、批注、研究笔记和会话存储；
- `evaluation.py`、`answer_evaluation.py`、`p2_*`：检索、回答与跨论文受控评测；
- `api/`：FastAPI Schema、错误契约和运行时装配；
- `observability.py`：不记录论文正文的本地 Trace。

项目中的 `user_id` 是本地作用域标识，不是生产鉴权。严格跨论文、OCR 和视觉图表能力仍受 README 中的公开边界约束。

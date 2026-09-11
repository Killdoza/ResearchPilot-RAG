# ResearchPilot HTTP API 契约（v1）

启动：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn researchpilot.api.main:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs` 查看由 FastAPI 生成的交互式 OpenAPI 文档。

## 系统接口

| 方法 | 路径 | 不加载模型 | 用途 |
|---|---|---:|---|
| GET | `/health` | 是 | 进程存活检查 |
| GET | `/ready` | 是 | 运行目录、Key 是否存在、Feature Flag 状态 |
| GET | `/v1/config` | 是 | 非敏感能力声明 |

## 文档接口

| 方法 | 路径 | 必填 Scope | 行为 |
|---|---|---|---|
| POST | `/v1/documents` | multipart `user_id`, `session_id` | 导入一个 PDF |
| GET | `/v1/documents` | query `user_id`, `session_id` | 列出本 Scope 的导入记录 |
| DELETE | `/v1/documents/{paper_id}` | query `user_id`, `session_id` | 删除该论文向量与本地上传物 |

上传示例：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/documents `
  -F "file=@E:\papers\example.pdf;type=application/pdf" `
  -F "user_id=demo_user" -F "session_id=demo_session"
```

仅支持文本型 PDF。扫描件或文字不足的 PDF 返回 `422 pdf_ingestion_failed`，错误消息会说明当前应转 OCR 路线；不会生成空索引。

## RAG 接口

`POST /v1/retrieve` 请求：

```json
{"question":"论文的核心方法是什么？","user_id":"demo_user","session_id":"demo_session"}
```

它返回页码、Chunk ID、来源、Dense/BM25/RRF/Rerank 分数和检索耗时，但不返回论文正文。

`POST /v1/chat` 在上面字段基础上可添加 `recent_history` 与 `mode`。默认：

```json
{"question":"论文的核心方法是什么？","user_id":"demo_user","session_id":"demo_session","mode":"default"}
```

返回 `answer`、`citations`、`refused`、`refusal_reason`、检索/生成耗时、产品 Route 和内部检索 `trace_id`。作者/元信息、全文总结、实验设置和正文细节会使用不同的证据选择策略，但保持同一响应结构。`strict_crossdoc` 目前默认关闭，并返回 `409 strict_crossdoc_not_available`；它不会悄悄降级为不安全的全局检索。

## 错误契约

每个响应含 `X-Trace-Id`；错误一律为：

```json
{"error":{"code":"invalid_scope","message":"...","trace_id":"api_..."}}
```

常见错误码：`invalid_scope`、`request_validation_failed`、`unsupported_media_type`、`empty_file`、`pdf_ingestion_failed`、`document_not_found`、`service_not_configured`、`model_service_unavailable`、`strict_crossdoc_not_available`。其中模型连接失败返回可重试的 `503 model_service_unavailable`；上传索引不会因此失效。严禁把 API Key、内部堆栈或论文全文放进错误体。

# ResearchPilot-RAG

[![CI](https://github.com/Killdoza/ResearchPilot-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Killdoza/ResearchPilot-RAG/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Killdoza/ResearchPilot-RAG)](https://github.com/Killdoza/ResearchPilot-RAG/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README_EN.md) | 简体中文

<p align="center">
  <img src="resources/full_logo.png" alt="ResearchPilot-RAG" width="520">
</p>

面向 AI 与计算机科学论文的可评测、可追溯 RAG 研究系统。它不仅提供论文问答页面，还包含结构感知解析、混合检索、页级 Citation、可信回答约束、版本化评测、消融实验、FastAPI、Docker、CI 与研究笔记工作流。

> 当前公开版本为 `v1.1.0`，对应内部验证里程碑 `product-rc2`。它适合作为本地论文研究工具、研究型开源项目和可复现的简历项目；它不是具备公网认证、多租户授权和生产 SLA 的 SaaS。

## 为什么不是普通 ChatPDF Demo

- 论文解析：Docling 结构化解析，失败时可追踪回退 pdfplumber；保留页码、章节路径、Block 类型和解析质量。
- 分块与检索：Structure-aware Parent–Child、Dense、持久化 BM25、RRF 与 BGE CrossEncoder Reranker。
- 任务路由：区分元信息、全文总结、实验设置、实验结果、多章节问题、正文细节、普通聊天和跨论文请求。
- 可信回答：统一 Citation、Claim-Citation 对齐、结构化表格回放、数字忠实度检查、摘要结构校验和全文缺失字段扫描。
- 记忆边界：工作记忆、长期结论和用户批注按用户、会话、论文 Scope 隔离；失败轮次不进入记忆。
- 研究工作流：查看 Citation 对应 PDF 页、保存研究笔记、导出带页码与 Chunk ID 的 Markdown。
- 可复现实验：E0--E5 检索消融、Long-context 对照和无 Oracle 泄漏的 P2 Query-aware 实验。
- 工程闭环：共享 Core、Streamlit、FastAPI、CPU Docker Compose、脱敏 Trace、GitHub Actions 和自动测试。

## 当前真实结果

| 实验或门禁 | 结果 | 解释边界 |
| --- | ---: | --- |
| E5 Test MRR | 0.6767 | 论文级留出 Test，不等于回答正确率 |
| E5 Test Recall@5 | 0.8000 | 固定分块 E4 为 0.5500 |
| E5 Test nDCG@5 | 0.6744 | 衡量证据页排序质量 |
| A0 单论文 Citation F1 | 0.6067 | Advanced RAG 回答对照 |
| `product-rc2` 回归 | 18/18 通过 | 固定两篇论文的产品可信度回归，不代表任意论文 100% 正确 |
| `product-rc2` 预期页覆盖 | 36/42（85.7%） | 保留未命中页这一真实边界 |
| 自动测试 | 274 passed | 证明代码契约，不等于语义回答正确率 |

P2 的 Query-aware / Evidence-Constrained 策略改善了来源边界和无依据比较控制，但没有证明总体 Citation F1 自动提升。因此严格跨论文能力保持 Feature Flag 关闭，不包装为默认高性能算法。完整报告见：

- [E0--E4 检索消融](eval/reports/p1c_retrieval_ablation.md)
- [E4/E5 结构感知解析对照](eval/reports/p1d_parser_chunking_report.md)
- [Long-context 与 RAG 对照](eval/reports/p1e_long_context_report.md)
- [P2 Query-aware 实验](eval/reports/p2_query_aware_report.md)
- [`product-rc2` 最终产品验收](docs/p2_8_4_d_product_rc2_review.md)

## 架构

```text
Streamlit / FastAPI
        │
RuntimeOrchestrator
  ├── product_rules_v2
  ├── paper-scoped memory
  ├── annotation policy
  └── retry / refusal boundary
        │
ResearchPilotService
  ├── Ingestion: Docling/pdfplumber → Blocks → Parent/Child Chunks
  ├── Index: scoped Chroma + persistent BM25
  ├── Retrieval: task-aware Dense + BM25 + RRF + optional Rerank
  └── Generation: evidence prompt → Answer/Claims/Citations
                         → citation/numeric/summary/evidence gates

Runtime: Chroma / BM25 / SQLite / session JSON / PDF copies / redacted traces
Evaluation: corpus manifest / JSONL / metrics / ablation / failure audit
```

更完整的数据边界与模块职责见 [架构说明](docs/architecture.md) 和 [产品运行时编排](docs/product_runtime_orchestration.md)。

## 快速开始

要求：Python 3.13、Git。首次运行会下载本地 Embedding、Reranker 和 Docling 相关模型。

### Windows PowerShell

```powershell
git clone https://github.com/Killdoza/ResearchPilot-RAG.git
cd ResearchPilot_RAG

py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

Copy-Item .env.example .env
# 只在本机 .env 中填写 DEEPSEEK_API_KEY；不要提交 .env
```

启动 Streamlit：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

启动 FastAPI：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn researchpilot.api.main:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/docs`。`/health` 不加载模型；首次上传、检索或问答才会加载本地模型，Query Rewrite 和正文生成可能调用 DeepSeek。

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

然后访问：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/docs
```

CPU Compose 已在 Windows + WSL2 下完成真实构建和健康检查。镜像与运行数据边界见 [部署说明](docs/deployment_linux.md)。

## 测试与质量门禁

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m compileall -q src scripts streamlit_app.py
.venv\Scripts\python.exe scripts\check_no_secrets.py
.venv\Scripts\python.exe -m pip check
```

`product-rc2` 冻结时：274 项测试、全仓 Ruff、编译、依赖和密钥扫描全部通过。

## 数据与复现边界

- 原始 PDF、上传文件、Chroma、BM25、SQLite、会话、模型文件和本地 Trace 均被 Git 忽略。
- 仓库提交 Corpus Manifest、评测 JSONL、配置、聚合结果和失败审计，不提交受版权限制的论文全文。
- `.env.example` 只包含变量模板；真实 Key 只能放在本地 `.env` 或部署环境变量中。
- Test 按论文划分并冻结；已完成 Test 不用于继续调参。
- DeepSeek 单价未固定时只记录调用数和 Token，不虚构费用。

## 当前限制

- 主要支持文本型 PDF；扫描件 OCR、图片表格、曲线图和流程图理解尚未产品化。
- 严格跨论文比较仍是实验能力，默认关闭。
- Claim-Citation 契约能够检查引用范围、数字和结构，但不是完整自然语言蕴含证明。
- 没有公网鉴权、限流、异步任务队列、多租户授权或生产 SLA。
- 尚未建立跨会话论文库、任意多论文研究空间和 PDF 逐句高亮。

## 文档入口

- [API 契约](docs/api_contract.md)
- [产品链路对齐](docs/product_pipeline_alignment.md)
- [研究工作流 MVP](docs/p2_8_5_research_workflow_mvp.md)
- [可信回答协议](docs/p2_8_4_trustworthy_answer_protocol.md)
- [`product-rc2` 最终验收](docs/p2_8_4_d_product_rc2_review.md)
- [文档导航](docs/README.md)
- [v1.1.0 发布说明](docs/release_v1.1.0.md)
- [2--3 分钟演示脚本](docs/demo_script_v1.1.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [环境自检工具](docs/environment_preflight.md)
## 开源协议

本项目使用 [MIT License](LICENSE)。第三方模型、数据集和论文仍分别遵循其原始许可证或使用条款。

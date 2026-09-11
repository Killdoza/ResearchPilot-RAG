# P2.8 基础性能记录

## 目标

本文件不把离线 E0--E6 质量实验和 HTTP 压测混为一个指标。P2.8 只记录服务边界在固定环境下的健康检查/检索延迟、吞吐和错误率；模型冷启动、下载和 DeepSeek 网络时延应单独注明。

## 复现命令

先启动 API，再运行：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts\load_test_api.py --endpoint health --requests 30 --concurrency 5
```

`health` 不会花 API 费用，是服务并发边界的最小检查。只有在上传 PDF、Embedding/Reranker 已预热、且愿意承担模型调用成本时，才运行：

```powershell
.venv\Scripts\python.exe scripts\load_test_api.py --endpoint retrieve --requests 10 --concurrency 1 `
  --user-id demo_user --session-id demo_session
```

脚本将 JSON 报告写入被 Git 忽略的 `data/runtime/benchmarks/`，包括成功数、错误数、P50、P95、平均时延与吞吐。不得使用 `/chat` 做并发压测：它会同时触发 DeepSeek 调用、计费和供应商限流，不能反映本地 API 的单一性能。

## 已记录的真实数值

| 日期 | 环境 | 接口 | 请求 / 并发 | 成功 | P50 | P95 | 平均 | 吞吐 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2026-08-26 | Windows 本地 `.venv`、Uvicorn 单进程、未加载模型 | `/health` | 30 / 5 | 30 / 30 | 7.64 ms | 166.66 ms | 33.90 ms | 144.091 RPS |

逐次报告保存于忽略目录 `data/runtime/benchmarks/p2_8_health_20260826.json`。这是 HTTP 存活检查，不包含 PDF 解析、Embedding、Chroma、BM25、Reranker 或 DeepSeek，因此不能被表述为“RAG 吞吐量”。

本机 Docker Desktop、WSL2 与 Docker Engine 已于 2026-09-02 验证通过，运行数据根目录设为 `E:\DockerData`。同日已成功构建 CPU 限制镜像并运行 Compose：容器 `healthy`，`/health` 和 `/ready` 均返回 `200`，容器内为 `torch=2.13.0+cpu`、`cuda_available=False`。`docker image ls` 显示逻辑镜像大小约 `3.87GB`；它是 Docling/OCR、Embedding、Reranker 等研究型文档处理依赖的体积边界，不是性能指标。

该容器验收只证明可构建、可启动、可检查就绪状态，**不产生也不声称**容器化 RAG 的 P50/P95。真实 `/retrieve` 与 `/chat` 压测必须在 PDF 已导入、模型预热、确认 DeepSeek 成本、固定机器配置后另行记录；其中 `/chat` 不应进行并发压测。

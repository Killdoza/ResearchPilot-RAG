# Linux / Docker 本地部署说明

## 前提

- Docker Engine 与 Docker Compose v2；
- 约 8GB 以上内存更稳妥（Docling、Embedding、CrossEncoder 首次加载较重）；
- DeepSeek Key 只保存在宿主机 `.env` 或部署平台 Secret 中，不写进镜像、Compose 文件或 Git。

## Docker 启动

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
docker compose up --build -d
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

也可以将 `DEEPSEEK_API_KEY` 仅保存在宿主机的环境变量中；Compose 会在启动时转发它，不会把它写进镜像或提交到仓库。不要在终端输出、截图或提交该变量值。`/ready` 只返回 Key 是否配置的布尔值。

当前镜像固定使用 CPU PyTorch，适合本地演示与 CPU 推理验证；不需要 NVIDIA Docker Runtime。Docling/OCR、Embedding 与 Reranker 使镜像较大，首次构建也会较慢，这属于当前研究原型的可见成本。

Docker volume `researchpilot-runtime` 保存 Chroma、BM25、上传 PDF 和 Trace。若要完全清空本地演示数据，先停止服务，先通过 `docker volume ls` **确认实际卷名**，再将下方占位符替换为该精确名称：

默认 Compose 仅绑定 `127.0.0.1:8000`，避免将未认证的 Demo API 暴露到局域网。若确实需要远程访问，应在反向代理、TLS 和认证方案明确后单独配置，不要直接把默认端口改为公网监听。

```bash
docker compose down
docker volume rm <confirmed-volume-name>
```

这会删除本地上传文件、向量索引、会话/Trace，无法从项目 Git 恢复；不要在包含真实用户数据的环境执行。

## 直接以 Uvicorn 运行

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=src
uvicorn researchpilot.api.main:app --host 0.0.0.0 --port 8000
```

Windows 路径含非 ASCII 字符时，Chroma 大型 HNSW 索引应通过 `.env` 改到 ASCII 绝对路径；Docker 容器内的 `/app/data/runtime` 是 ASCII 路径。

## 不把这个 Demo 当生产系统

当前 API 没有认证、授权、限流、TLS 终止、审计留存策略或真正的多租户控制。因此仅绑定到本机、可信内网或在反向代理认证之后使用。公网部署、真实用户数据和长期存储属于 P4。

# ResearchPilot v1.1.0 发布说明

状态：公开源码版本

验证来源：内部里程碑 `product-rc2`

公开版本：`v1.1.0`

## 版本定位

`v1.1.0` 在 `v1.0.0` 的 FastAPI、Docker、CI 和可复现实验基础上，补齐面向真实论文阅读的可信回答与研究工作流。它适合作为本地 AI/CS 论文助手、RAG 研究原型和可复现简历项目；不应被描述为生产级多租户 SaaS。

## 相比 v1.0.0 的主要变化

- 统一 `product_rules_v2` 与 `RuntimeOrchestrator`，Streamlit 不再自行拼接检索和模型逻辑；
- 元信息、全文总结、实验设置、实验结果、多章节、正文细节、普通聊天与跨论文请求采用显式任务路由；
- Citation 归一化、Claim-Citation 范围检查、数字忠实度检查和结构化表格证据回放；
- 摘要结构校验、全文 URL/DOI/代码地址扫描与证据不足拒答；
- 工作记忆、长期结论与用户批注按论文隔离，原文和用户观点分别展示；
- Citation 可打开本地 PDF 对应页，回答可保存为研究笔记并导出 Markdown；
- 模型调用、Token、缓存节省和估算成本进入 Trace；失败轮次不写入记忆；
- 修复 390px 移动端布局、Logo、流式显示和重复上传状态框等产品问题；
- 全仓质量门禁扩展为 Ruff format、Ruff lint、274 项 Pytest、编译、密钥和依赖检查。

## 已验证事实

| 项目 | 结果 | 不代表什么 |
| --- | ---: | --- |
| E5 单论文 Test MRR | 0.6767 | 不是回答正确率 |
| E5 单论文 Test Recall@5 | 0.8000 | 不是任意论文都能命中 |
| E5 单论文 Test nDCG@5 | 0.6744 | 不是生成质量评分 |
| A0 单论文 Citation F1 | 0.6067 | 不是 Claim 语义完全正确率 |
| `product-rc2` 产品回归 | 18/18 | 只覆盖冻结的真实回归用例 |
| 预期页覆盖 | 36/42（85.7%） | 仍有 6 个预期页未进入最终证据 |
| 自动测试 | 274 passed | 证明代码契约，而非所有模型回答正确 |

结构感知 E5 相比固定分块 E4 将留出 Test 的 Recall@5 从 0.55 提升到 0.80。P2 的 Query-aware/Evidence-Constrained 策略改善来源边界和无依据比较控制，但没有证明总体 Citation F1 自动提升，因此严格跨论文能力仍默认关闭。

## 快速验收

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe scripts\check_no_secrets.py
.venv\Scripts\python.exe -m pip check
```

Streamlit 与 FastAPI 启动命令见根目录 `README.md`。Docker 验收：

```powershell
docker compose up --build
Invoke-WebRequest http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/ready
```

## 已知限制

- 主要支持文本型 PDF；扫描件 OCR、图片表格、架构图、曲线图和流程图理解尚未产品化；
- 严格跨论文比较仍为 Feature Flag，默认不会静默降级到全局 Top-K；
- Claim-Citation v1 检查引用范围、结构和数字，不是完整自然语言蕴含证明；
- 未实现公网鉴权、授权、限流、异步任务队列、多租户数据治理和生产 SLA；
- 当前研究型 CPU 镜像包含 Docling、Embedding 和 Reranker 等依赖，体积较大；
- 原始 PDF、模型缓存和运行时数据库不随仓库发布，首次运行需要下载模型并准备自己的论文。

## 升级与兼容

- 旧会话仍可读取，但缺少新版论文资产的会话可能提示重新上传，不会回退到无边界旧检索；
- `.env`、运行时数据库和上传 PDF 不应提交；
- Windows 请始终使用 `.venv\Scripts\python.exe -m <module>`，避免虚拟环境入口记录旧路径造成 launcher 错误；
- Windows Chroma 持久化目录应使用 ASCII 路径，项目运行时已有显式防护。

## 发布后的展示收尾

1. 观察首次公开 GitHub Actions；
2. 创建 `v1.1.0` GitHub Release，并使用本说明作为版本边界；
3. 按 `demo_script_v1.1.md` 录制演示，不展示 Key、私人路径或未授权论文；
4. 将仓库和演示链接加入保研简历。

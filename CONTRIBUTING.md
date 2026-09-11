# 参与 ResearchPilot-RAG 开发

感谢你愿意改进 ResearchPilot-RAG。本项目优先接受能够提升论文证据可追溯性、可复现性和真实研究工作流的修改。

## 开始之前

1. 不要提交 API Key、`.env`、原始论文 PDF、会话、数据库、模型文件或运行 Trace。
2. Bug 请提供最小复现、期望行为、实际行为、环境和脱敏后的首段错误。
3. 新算法必须说明对照配置、数据划分、指标和失败案例，不能只提交一个“看起来更先进”的组件。
4. 不得使用已冻结 Test 反复调参；新研究问题应建立新的数据版本或 Dev 集。

## 本地开发

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

提交前运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe scripts\check_no_secrets.py
.venv\Scripts\python.exe -m pip check
```

## Pull Request 要求

- 一个 PR 聚焦一个明确问题；
- 说明为什么需要修改，以及用户可见行为是否变化；
- 为新行为增加测试；
- 涉及检索或生成质量时，提交配置、聚合结果和错误分析，不提交原始受限 PDF；
- 涉及 UI 时，提供桌面端和 390px 小屏截图；
- 不覆盖已有实验结果，应创建新版本；
- 更新相关 README、API 契约或阶段报告。

## 项目边界

当前默认链路面向文本型 AI/CS 论文和单论文可信问答。OCR、多模态、严格跨论文和生产级多租户能力仍是独立路线。贡献可以探索这些方向，但必须有独立评测和安全回退，不能默认打开未经验证的能力。

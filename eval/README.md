# ResearchPilot 评测目录

本目录保存可公开复核的评测协议、语料清单、人工标注问题和聚合结果；不保存论文 PDF、Chroma 索引、浏览器会话或用户对话。

## 目录约定

- `manifests/`：论文元数据、来源地址、文件名和 SHA-256；原始 PDF 仅放在被 Git 忽略的 `data/raw/ai_cs_eval_v0.1/`。
- `datasets/`：人工核验的问题、标准答案和页级证据定位。当前 `ai_cs_eval_v0.1.jsonl` 已含 53 条 Dev 题（5 条种子题 + 48 条人工准备题）与 24 条 Test 题；Test 覆盖 4 篇不与 Dev 重叠的论文。Dev 用于开发、调试和选参；Test 只在配置冻结后运行，不能反复据此调参。
- `reports/`：经过冻结与人工复核的最终实验报告。逐题运行记录和重复生成的 CSV/Markdown 输出不随公开仓库分发，可通过运行脚本在本地 `eval/results/` 与 `data/runtime/` 重新生成。

## 评测隔离

评测语料必须导入专用的 `user_id + session_id` Scope，例如 `eval_ai_cs_v01 / corpus_v01`，并使用独立的 Chroma 目录和集合；不能复用浏览器地址中的 `rp_user`，也不能混入日常聊天上传的论文。

在 Windows 上，如果项目目录本身包含中文等非 ASCII 字符，评测 Chroma 目录还必须通过 `.env` 中的 `RESEARCHPILOT_EVALUATION_CHROMA_DIR` 指向一个 ASCII 路径；否则大型 HNSW 索引可能在跨进程读取时损坏。公开仓库名称和 README 示例均使用 ASCII 路径。

## 固定工作流

1. 在 `manifests/` 填写论文元数据和预期 SHA-256。
2. 将对应开放获取 PDF 放入忽略目录 `data/raw/ai_cs_eval_v0.1/`。
3. 执行数据校验和语料导入；导入前默认 dry-run。
4. 完成人工页级证据标注，并再次校验 JSONL。
5. 先只在 Dev 集运行、调试和选参；冻结配置后再运行一次 Test 集。

完整的字段规则见 [标注指南](../docs/annotation_guide.md)。

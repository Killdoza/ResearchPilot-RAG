# AI/CS 论文评测集标注指南 v0.1

## 1. 目的与边界

本指南用于构建 ResearchPilot 的人工核验评测集。每一条可回答问题都必须能定位到指定论文的页级原文证据；评测集衡量的是“系统是否找对论文证据”，而不是模型是否能凭常识写出看似合理的答案。

只选择开放获取的 AI/CS 论文。仓库不提交 PDF 全文、完整抽取文本、私人对话或 API Key；只提交论文元数据、问题、简短标准答案和证据定位。

## 2. 语料选择

v0.1 的目标是 10 篇论文：4 篇 LLM/NLP/RAG、2 篇信息检索或推荐、2 篇视觉或多模态、2 篇系统/软件工程/人机交互。每篇论文应有稳定的公开来源 URL，优先选择可复制文本的 PDF。

每篇论文在 Manifest 中记录 `paper_key`、标题、领域、年份、来源 URL、本地文件名和 PDF 的 SHA-256。`paper_key` 一经使用不得重命名。

## 3. 问题类型

允许的 `task_type`：

- `metadata`：标题、作者、会议、年份等元信息；
- `keyword_exact`：模型名、数据集名、指标名、公式编号等精确术语；
- `local_fact`：单段可验证事实，如超参数、样本量、模块设置；
- `semantic_explanation`：方法动机、设计机制或语义解释；
- `experimental_setup`：数据集、Baseline、训练/实验设置；
- `multi_section`：需要结合多个章节的结论；
- `cross_lingual`：中文问题定位英文论文证据；
- `unanswerable`：论文中不存在足以支持答案的证据。

## 4. 可回答问题的证据规则

1. `answerable: true` 时，`gold_evidence` 至少包含一项；每项必须填写 `paper_key`、`page_start`、`page_end`，且页码从 1 开始。
2. 证据必须直接支持 `expected_answer` 的关键事实。不能只因为主题相近、术语相同或模型常识合理就标为证据。
3. 涉及多个独立结论的问题必须标注全部必要证据，而非只标其中一页。
4. `section_hint` 填论文中可见的章节标题；当前解析器尚不能可靠抽取标题时，可填空字符串，但需在标注备注中说明。
5. `expected_answer` 应简短、可核验，不应加入论文没有明确说明的推断。

## 5. 不可回答问题规则

`task_type: unanswerable` 必须同时满足：

- `answerable: false`；
- `gold_evidence: []`；
- `expected_answer` 明确说明“论文未提供该信息，不能可靠作答”；
- 在全文中人工检查过相关章节、附录和实验部分；
- 问题不是措辞含糊，而是论文确实没有给出所需事实。

当前版本只记录不可回答问题的误召回和 Top-1 结果；在拒答阈值校准前，不报告“拒答准确率”。

## 6. 中英文与跨论文规则

中文问题可以标注英文论文，但标准答案应使用中文概述，并保持模型名、数据集名、指标名等原术语。跨论文问题使用多个 `paper_keys`；每篇论文对应的关键结论都要有独立证据定位。

## 7. Dev/Test 切分与质量控制

v0.1 当前有 77 条：Dev 53 条、Test 24 条。Test 论文不能与 Dev 论文重叠，以避免系统通过记忆同一篇论文的固定模式获得虚高结果。

开发、调参和错误修复只查看 Dev。冻结实验配置后再运行 Test；若需要继续调参，应创建新版本数据集或保留新的 Test 集。

每完成 10 条问题，至少抽查 3 条：重新打开 PDF，验证页码、章节、标准答案和问题类型是否一致。

## 8. JSONL 样例

```json
{
  "id": "ai_cs_v01_q001",
  "split": "dev",
  "paper_keys": ["paper_key_example"],
  "question": "论文的在线检索阶段如何工作？",
  "question_language": "zh",
  "task_type": "semantic_explanation",
  "answerable": true,
  "gold_evidence": [
    {
      "paper_key": "paper_key_example",
      "page_start": 5,
      "page_end": 5,
      "section_hint": "Online Retrieval"
    }
  ],
  "expected_answer": "系统先解析用户意图，再在候选控件中检索并重排。"
}
```

真实记录必须通过 `scripts/validate_eval_dataset.py`；不要将本样例直接放入数据集。

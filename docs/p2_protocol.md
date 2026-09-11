# P2：Query-aware Evidence-Constrained Retrieval 实验协议

## 1. 研究问题与边界

P2 不验证“模型能否把问题分到若干类别”这一孤立能力，而验证以下端到端研究问题：

> 在冻结 E5 解析、分块、表示模型和混合检索组件后，系统能否依据问题类型、目标论文范围和当前证据覆盖状态，动态选择检索预算、候选来源、重排、扩检与拒答动作，从而提升跨论文证据覆盖、引用可靠性和拒答可靠性，并把延迟和成本控制在可解释范围内？

P2 只改变**检索控制策略**。下列项目必须固定，不能在 E6 中同时替换：

| 固定变量 | 冻结值 |
|---|---|
| 解析与分块 | E5 的 Docling `2.120.2`、`structure_aware_v1` Parent–Child Block |
| OCR | 关闭；扫描件 / 图像理解属于后续 P3.1 |
| 向量模型 | `BAAI/bge-small-zh-v1.5` |
| 稀疏检索 | 持久化 BM25 |
| 融合 | RRF，`k=60` |
| 重排器 | `BAAI/bge-reranker-base` |
| 页面产品链路 | 不作为实验对照；H0 Streamlit 链路不能冒充 E5、A0 或 E6 |

## 2. 对照、实验对象与公平性

P2 必须同时回答“如果计划完全正确，策略本身值不值得做”和“真实用户输入下系统实际能做到什么”。因此不再把所有自适应实验统称为 E6：

| 模式 | Planner 可见信息 | 跨论文约束 | 作用 |
|---|---|---|---|
| Static E5 / A0 | 用户问题 + 当前可检索 Scope；固定 Dense=10、BM25=10、RRF、Rerank、全局 Top-K | 无论文配额、无 Coverage Gate | 唯一静态对照 |
| E6-Oracle | 额外提供 Gold Query Type 与 Gold 目标论文 | 每篇 Gold 论文配额、Gate、一次扩检、结构化拒答 | 策略理论上界；不能称为真实产品效果 |
| E6-Scoped | 用户上传、勾选或当前会话明确指定的论文范围 | 每篇用户指定论文配额、Gate、一次扩检、结构化拒答 | 用户已给定研究范围时的真实效果 |
| E6-End-to-End | 用户问题、会话目录、标题 / 文件名 / 别名解析 | 预测论文范围的配额、Gate、一次扩检、结构化拒答 | 最严格的端到端效果 |

`static_e5_a0.yaml`、`e6_oracle.yaml`、`e6_scoped.yaml` 与 `e6_end_to_end.yaml` 是可审计的协议文件；P2-B0 已实现分模式计划与运行时 / Gold 隔离，P2-B1 已实现不调用 LLM 的规则 Query Analyzer，P2-B2/B3 已实现论文级 Persistent Hybrid 适配、均衡候选、G0/G1 Gate、一次扩检后结构化拒答，以及 G2 Claim-Citation-Context 协议校验。P2 专用 Answer/Trace 运行接口与正式 E6 运行器仍属于后续工作。真实实验开始后，运行记录必须保存配置 SHA-256、数据集 SHA-256、路由版本、`query_type_source`、`scope_source`、预测 Scope、每篇候选数、各阶段耗时和最终拒答原因。

### 2.1 Gold 与运行时输入隔离

`GoldAnnotation` 只能由 Evaluator 使用，包含 `query_type`、`required_paper_keys`、`answerable`、Gold Evidence 和页码。`RuntimeRequest` 只能包含用户问题、用户选定 Scope、会话可访问论文目录与配置。端到端 Planner 不能读取前者；若使用 Oracle，则必须在 Trace 与报告中写明 `query_type_source=oracle_eval`、`scope_source=oracle_eval`。

### 2.2 P2-B1：规则 Query Analyzer 与保守范围回退

P2-B1 的 `RuleQueryAnalyzer` 只读取问题文本、会话可见论文目录、文件名/标题/别名和用户显式选择。它输出可审计的 `QueryAnalysis`：预测 `query_type`、Scope 来源、表面规则信号、`unanswerable_risk` 与回退原因。它不会读取 Gold Query Type、`required_paper_keys`、标准答案或页码证据。

当前规则优先级为：显式用户 Scope > 标题/文件名/别名解析；两个论文范围或明确比较信号触发跨论文候选；随后依次识别元数据、表/图、精确术语、局部实验事实、多章节问题和语义解释。问候类 `general_chat` 跳过论文检索。“是否提到 / 是否报告”等措辞只记为 `unanswerable_risk`，不能把运行时问题伪装成已知不可回答的 Gold 标签。

以下情况必须停止而不是猜测：没有解析到论文范围、同一标题/别名命中多篇论文、比较问题只识别到一篇论文、或 Scoped 模式没有用户显式选择。Trace 必须保留 `scope_insufficient` 的细分原因。Router 评测仅对数据集中实际覆盖的类型报告 Accuracy / Macro-F1，同时报告 Target-document Precision / Recall 和 Scope Fallback Rate；不得以 v0.2 当前未覆盖的类型宣称完整九分类能力。

## 3. P2 v0.2 语料与 Split 纪律

- Corpus：10 篇开放获取 AI/CS 论文，Dev 6 篇、Test 4 篇；文件名、SHA-256 和来源记录在 `eval/manifests/ai_cs_p2_v0.2.corpus.json`。
- 题单：`eval/datasets/ai_cs_p2_v0.2.jsonl`，由人工标注并经过冻结校验；原始 Markdown 工作稿不随公开仓库分发。
- 当前共 36 题：Dev 24，Test 12；所有题均为论文级 Split 隔离。
- 可回答的跨论文题必须给出 C1（论文 A）、C2（论文 B）、C3（比较边界或综合结论）及两篇论文的页级 Evidence；不可回答题不得伪造证据。
- 当前题单的严格性能可比题为 **0**（`comparison_allowed=true` 为 0）。这是语料真实结论，不是缺失数据。E6 的“允许比较”分支先由单元测试覆盖；若未来加入共享任务、数据集、指标和设置的论文对，再以新数据版本加入该分支，不能回写本次 Test。
- Dev 可用于改变 P2 规则；Test 在 P2 Dev 冻结后只运行一次。最终报告必须注明 Test 是论文级留出集，不是标注者盲测。

## 4. P2-A 数据冻结与验证命令

独立校验已提交的 JSONL、Corpus、论文 Split、PDF 页码和冻结哈希：

```powershell
.venv\Scripts\python.exe scripts\validate_p2_dataset.py
```

校验器必须拒绝：非法页码、未知 `paper_key`、一题混用 Dev/Test、可回答题缺少 C1/C2/C3 或任一目标论文证据、不可回答题附带伪造 Evidence、冻结后被悄悄改写的数据集。

PDF 位于 Git 忽略的 `data/raw/ai_cs_eval_v0.2/{dev,test}/`。若本地 PDF 缺失或哈希与 Manifest 不一致，不能继续运行评测，应先恢复对应的冻结文件并重新核验页码。

## 5. E6 的可解释控制契约（P2-B 实现目标）

每个检索请求必须产生一个可序列化的 `QueryPlan`，至少记录：

```json
{
  "query_type": "cross_document",
  "query_type_source": "rule_predicted",
  "target_paper_keys": ["paper_a", "paper_b"],
  "scope_source": "user_selected",
  "retrieval_policy": "balanced_cross_document",
  "dense_limit": 12,
  "sparse_limit": 12,
  "rerank_enabled": true,
  "minimum_evidence_per_paper": 1,
  "allow_one_retry": true,
  "coverage_gate_required": true,
  "router_version": "p2_rules_v1"
}
```

第一版路由器是规则系统，而非让 LLM 黑盒决定预算：

| Query Type | 首选动作 |
|---|---|
| `metadata` | 优先首页元数据 / 标题，低预算，不必全文重排 |
| `keyword_exact` | 提高 BM25 与标题、术语、表格命中权重 |
| `local_fact` | 优先实验段、表格、图注邻近正文 |
| `semantic_explanation` | Child 召回后回填 Parent / Section 上下文 |
| `table_figure` | 优先表格、图注和邻近段；若必须读像素，显式报告当前能力边界 |
| `multi_section` | 子问题分解、分节召回再合并证据 |
| `cross_document` | 对每篇目标论文分别召回，禁止由全局 Top-K 垄断 |
| `unanswerable` | 执行证据、分数、解析质量检查，不把“不知道”交给 Prompt 一句话 |
| `general_chat` | 跳过论文检索，并标明回答不来自论文证据 |

## 6. Coverage Gate 与 retry-then-refuse

对于要求两篇论文共同支撑的跨论文问题，Gate 分为三层：

1. **G0 Document Coverage**：运行时目标 Scope 中每篇论文至少有候选；
2. **G1 Relevant Evidence Coverage**：每篇论文至少有一条分数、Block 类型、非空文本和 Parser 质量均达标的最终证据；不能借助 Gold 页码判断；
3. **G2 Claim-level Coverage**：每个最终比较性 Claim 均映射至对应论文、页码和 Context ID；`comparison_allowed=false` 时不得输出整体优劣 / 总排名等越界结论。

Gold Evidence 仅在离线评分时判定 G0/G1 是否找到了正确证据。解析质量低、分数不达阈值或 Scope / 证据覆盖不足时，Gate 不通过。

当前代码实现中，`PersistentHybridPerPaperProvider` 将每个运行时 `paper_key` 映射到独立的 `paper_id` 过滤条件，并把 `QueryPlan` 的 Dense、BM25、Rerank 预算下传到底层 E5 Hybrid Retriever；`QueryAwareCandidateController` 先运行每篇论文的召回，再执行 `per_paper_quota` 选择和 G0/G1。它不接收 Gold 页码。若初检失败，`retrieve_with_retry` 至多以固定倍数扩检一次；仍失败时返回 `refused=true` 与 `insufficient_evidence:<gate_reason>`，生成层不得将残留候选伪装成答案引用。

失败路径只能执行一次扩检：提高相应论文的候选预算并保留检索 Trace。扩检后仍不通过时，返回：

```json
{"refused": true, "refusal_reason": "insufficient_evidence"}
```

拒答不附加无关引用；它必须指出缺少哪篇论文或哪类证据。P2-B 会把 Gate、阈值与候选配额实现为可测试代码，P2-C 才能在 Dev 上调整并冻结具体数值。

## 7. 成本与运行纪律

P2-A 和 P2-B 的单元测试、导入、Dry-run 都不得调用 DeepSeek。真正生成答案之前，运行器必须先提供 `--dry-run`，输出题数、调用次数、每种模式的上下文长度、估计 Token、费用上限和最大重试次数。

P2-C 已在 Dev 24 题上完成四种模式的 Dry-run：最大 94 次生成调用、356,593 输入 Token 和 112,800 输出 Token，且无容量失败；End-to-End 中有 2 题因 `scope_insufficient` 回退，不触发模型调用。详细数字见 `docs/p2_dev_dry_run.md`。当前价格字段为 `null`，因此不得从 Token 上界臆造人民币费用；项目作者确认预算后才允许启动真实 Dev。

P2 原始论文按 `data/raw/ai_cs_eval_v0.2/{dev,test}/` 存放。评测导入脚本已同时支持该论文级 Split 布局和旧版扁平目录；Manifest 只保存文件名。若递归查找出现多个同名 PDF，脚本会停止并报错，不会猜测导入 Dev 或 Test 的哪一份。

只在人工确认预算后才运行 Dev。任何一次真实运行要保存：输入问题、QueryPlan、每篇论文候选数、Gate 决策、扩检是否触发、引用、Token、费用、检索 / 生成 / 端到端延迟。Test 不能作为规则调优循环的一部分；单题超过上下文或预算上限时记为 `capacity_failure`，保持配置不变并继续运行剩余 Test，最终单独报告 Capacity Failure Rate。

## 8. P2 成功判据与下一步

P2-C 至少报告 Per-paper Recall、G0/G1/G2 Evidence Coverage、Citation Precision/Recall/F1、Wrong Comparison Rate、Correct Refusal Rate、Target Document Scope Precision/Recall、可覆盖 Query Type 的 Router 指标、P50/P95 延迟和 Token/费用，并展示 Oracle—Scoped—End-to-End 差距。E6 不要求所有指标同时上升；若证据覆盖提高但成本显著恶化，也必须如实呈现。

只有 E6 在 Dev 的失败案例经人工复核、配置和数据哈希被冻结后，才运行一次新的 P2 Test。P2 后优先进入 P2.8 的 FastAPI、Docker、CI、Trace、Linux 部署说明与基础压测，形成可投递版本；随后才依据失败审计选择表示模型单变量实验（P2.5）、OCR/图表能力（P3/P3.1）或有界 Agent，而不是凭感觉并行堆叠功能。

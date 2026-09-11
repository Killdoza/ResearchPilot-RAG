# P2 v0.2 跨论文题标注规范

本规范约束本地 P2 Markdown 标注稿到 `ai_cs_p2_v0.2.jsonl` 的转换。公开仓库保留冻结 JSONL 和 Manifest，不分发个人工作稿。目标是评测“证据是否同时覆盖每篇目标论文，以及系统能否在不可比时停止越界推断”，不是让模型写一段看起来流畅的论文综述。

## 1. 每题必须具备的字段

| 字段 | 规则 |
|---|---|
| `split` | `dev` 或 `test`；题内所有 `paper_keys` 必须在同一 Split |
| `task_type` | 可回答题为 `cross_document`；应拒答题为 `unanswerable` |
| `answerable` | 只能是 `true` / `false` |
| `paper_keys` | 题目涉及的全部论文，至少两篇 |
| `required_paper_keys` | 生成前必须覆盖的论文，至少两篇，且是 `paper_keys` 子集 |
| `comparison_dimension` | 可审计英文短标识，如 `retrieval_trigger_information_use` |
| `comparison_allowed` | 只有“任务、数据集、指标、实验条件”均有页级证据对齐时才能为 `true` |
| `refusal_reason_target` | 仅不可回答题使用：`comparison_not_allowed`、`incompatible_setup`、`not_reported` 或 `insufficient_evidence` |
| `question` | 明确问什么；不要把多个没有共同条件的性能结论揉在一句里 |

## 2. 三类题及正确系统行为

| 类型 | `answerable` | `comparison_allowed` | Gold 要求 | 预期系统行为 |
|---|---:|---:|---|---|
| A：严格可比 | true | true | C1 / C2 都明确记录同条件数值，C3 限制结论边界 | 分别引用两篇论文，仅在可比范围内比较 |
| B：机制或职责差异 | true | false | C1 / C2 是各自事实，C3 解释差异和不可统一排序的原因 | 允许解释，但不能输出整体优劣排名 |
| C：不可回答或不可比 | false | false | 不写虚构 Claim / Evidence，写清拒答原因 | `refused=true`，指出证据缺口或不可比条件 |

当前 v0.2 的 36 题均是 B 或 C 类，A 类为 0。这是已审核的研究限制：不应为了指标覆盖率虚构同条件性能比较。实现中仍会为 A 分支编写单元测试；若以后新增 A 类题，必须升版本、更新 Manifest 并在 Dev 冻结前完成审核。

## 3. 可回答题的 C1 / C2 / C3 与证据要求

可回答题必须恰好有三项预期 Claim：

- **C1**：论文 A 的可证实事实；
- **C2**：论文 B 的可证实事实；
- **C3**：两者的受约束差异、比较边界，或仅在严格同条件下可成立的结果比较。

每条 Evidence 必须有 `evidence_id`、绑定的 `claim_ids`、`paper_key`、物理 PDF 页码范围、章节提示和中文证据概括。每篇 `required_paper_key` 至少有一条 Evidence；**C3 也必须由所有 required 论文的 Evidence 共同支撑**。页码以 PDF 阅读器从 1 开始的物理页码为准，不是文档印刷页码。

## 4. 不可回答题如何写

不可回答不表示“题目不重要”，它用于验证系统不会在证据不足时编造比较。不可回答题要：

1. 写清到底缺少什么：同任务 / 同数据集 / 同指标 / 同设置，或其中一篇根本未报告信息；
2. 用 `refusal_reason_target` 表示标准化原因；
3. 在“预期答案”写出应当拒答时的说明边界；
4. “页级证据”填写模板中的“无需伪造不存在的证据”，导入 JSONL 后它必须为空。

不要给不可回答题塞进“看起来相关”的引用，也不要把论文 A 的数值当作论文 B 的结论。

## 5. 导入前人工检查清单

- [ ] `paper_key` 出现在 P2 Corpus Manifest 中，且不混用 Dev 与 Test；
- [ ] 每个可回答题都有 C1、C2、C3；
- [ ] 每个 Gold 页码已对照本地原 PDF；
- [ ] C3 没有超出两篇论文证据允许的比较边界；
- [ ] `comparison_allowed=true` 的题已逐页确认四个可比条件；
- [ ] 不可回答题没有 Claim 和 Gold Evidence；
- [ ] 修改题单后运行导入器与校验器，确认 Manifest 哈希同步更新。

## 6. 版本纪律

JSONL、Corpus Manifest、P2 数据 Manifest 和本规范都应提交 Git；PDF、向量索引、模型缓存和运行输出不提交。Test JSONL 在 P2 Dev 冻结后不可改写；发现 Gold 错误时先记录问题，建立新版本而不是静默覆盖旧 Test。

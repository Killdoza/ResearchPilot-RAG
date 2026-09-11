# P1-D：结构感知解析与 E5 对照报告

E4 与 E5 使用同一份 AI/CS 论文、同一页级评测集、Embedding、Dense/BM25 候选数、RRF、BGE Reranker 与 Top-K；唯一实验变量是 Parser 与 Chunker。所有配置选择只基于 Dev。

## 解析与索引质量

- Parser：`docling 2.120.2`；OCR：`False`。
- 10 篇论文共生成 2551 个 Block、1273 个 Parent Chunk、4468 个 Child Chunk；空页数为 0。
- Block JSON 与导入 Manifest 均在 Git 忽略的 `data/runtime/` 中，原始 PDF 未被复制或提交。

## Dev 总体结果

| 指标 | E4 固定分块 | E5 结构感知 | 变化 |
|---|---:|---:|---:|
| MRR | 0.7395 | 0.7899 | +0.0504 |
| Recall@5 | 0.8113 | 0.8868 | +0.0755 |
| nDCG@5 | 0.7557 | 0.8146 | +0.0588 |
| P95 检索耗时（ms） | 3309.3053 | 2689.2255 | -620.0798 |

## 按问题类型的 Dev 结果

| 类型 | 题数 | E4 Recall@5 | E5 Recall@5 | E4 MRR | E5 MRR |
|---|---:|---:|---:|---:|---:|
| cross_lingual | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| experimental_setup | 17 | 0.7647 | 0.8824 | 0.7255 | 0.8039 |
| keyword_exact | 3 | 1.0000 | 1.0000 | 0.7500 | 0.8333 |
| local_fact | 5 | 0.4000 | 0.6000 | 0.4222 | 0.6000 |
| multi_section | 4 | 1.0000 | 1.0000 | 1.0000 | 0.7500 |
| semantic_explanation | 22 | 0.8636 | 0.9091 | 0.7500 | 0.8045 |

## E5 新增找回的 E4 完全漏召回案例

| 题目 ID | 类型 | 问题 |
|---|---|---|
| ai_cs_v01_q005 | experimental_setup | 这篇 RAG 论文训练时使用了什么 GPU 配置？ |
| ai_cs_v01_q015 | semantic_explanation | Transformer 的 encoder 和 decoder 分别由什么结构组成？两者最主要的结构差异是什么？ |
| ai_cs_v01_q018 | semantic_explanation | Transformer 为什么需要 positional encoding？论文采用的 positional encoding 是如何构造的，为什么最终选择这种方法而不是 learned positional embedding？ |
| ai_cs_v01_q027 | experimental_setup | 论文在 MuJoCo 连续控制实验中如何比较不同 surrogate objectives？哪种设置取得了最高的平均归一化得分？ |
| ai_cs_v01_q040 | local_fact | LoRA 中矩阵 (A) 和 (B) 如何初始化？为什么训练开始时 LoRA 不会改变预训练模型原本的输出？ |
| ai_cs_v01_q042 | semantic_explanation | 为什么 LoRA 相比 adapter 不会增加额外的推理延迟？在部署时 LoRA 权重如何与原模型权重合并？ |

## E5 新增完全漏召回案例

| 题目 ID | 类型 | 问题 |
|---|---|---|
| ai_cs_v01_q001 | local_fact | ResNet 在 ImageNet 上评估的最大网络深度是多少？ |
| ai_cs_v01_q039 | semantic_explanation | LoRA 如何用低秩矩阵表示预训练权重 (W_0) 的更新？训练时哪些矩阵被冻结，哪些矩阵被更新？ |
| ai_cs_v01_q050 | semantic_explanation | MapReduce 的 locality optimization 是如何减少网络通信的？调度 Map task 时 Master 如何利用 GFS 中的数据副本？ |

## 一次性留出 Test 结果（配置已冻结）

E5 配置已在页面级人工复核和 Dev 结果后，以提交 `099b406` 冻结；以下 24 条 Test 只运行一次，结果不再用于调节 Parser、Chunker 或检索参数。E4 Test 为此前已冻结的固定分块对照，两个 Test 均含 20 条可回答题和 4 条不可回答题。

| 指标 | E4 固定分块 | E5 结构感知 | 变化 |
|---|---:|---:|---:|
| MRR | 0.4056 | 0.6767 | +0.2711 |
| Recall@5 | 0.5500 | 0.8000 | +0.2500 |
| nDCG@5 | 0.4372 | 0.6744 | +0.2372 |
| P95 检索耗时（ms） | 3356.9483 | 2799.6071 | -557.3412 |

- E5 Test 的不可回答题 Top-1 仍有召回比例为 `1.0000`。这不是拒答准确率：当前评测尚未校准拒答阈值，后续需单独开展拒答/置信度实验。
- Test 论文按论文级留出，但问题由项目作者预先编写，因此它不是完全盲测；从本次运行起不再依据 Test 指标改动 E5。
- 页面级视觉复核见 `docs/p1d_page_audit.md`：本次修复了公式 Block 漏失；图像内容、复杂双栏顺序和公式无损转写仍是已记录的边界。

## 当前决策

- E5 已完成页面级人工复核并冻结，作为当前高精度结构感知检索基线；P1-D 不再继续使用 Test 调参。
- 交互产品可继续以 E3 为低延迟检索策略，但其底层 Parser/Chunker 可在 E5 验证后复用结构感知 Block；E5 的 Reranker 链路作为高精度对照。

Dev 逐题记录：E4 `data/runtime/evaluations/e4_dense_bm25_rrf_rerank_dev_20260817T112707Z.jsonl`；E5 `data/runtime/evaluations/e5_structure_aware_dev_20260817T161108Z.jsonl`。
Test 逐题记录：E4 `data/runtime/evaluations/e4_dense_bm25_rrf_rerank_test_20260817T112946Z.jsonl`；E5 `data/runtime/evaluations/e5_structure_aware_test_20260817T161452Z.jsonl`。

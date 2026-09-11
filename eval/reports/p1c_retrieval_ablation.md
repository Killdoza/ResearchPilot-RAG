# P1-C：E0--E4 检索消融报告

本报告严格使用 Dev 结果选择后续默认检索链路；Test 不参与调参。

## 总体结果

| 实验 | MRR | Recall@5 | nDCG@5 | 平均 ms | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| e0_dense | 0.381853 | 0.596154 | 0.421296 | 63.126502 | 59.477300 | 64.051360 |
| e1_bm25 | 0.549596 | 0.692308 | 0.567141 | 31.344309 | 2.394600 | 3.097300 |
| e2_dense_bm25_union | 0.388851 | 0.557692 | 0.405575 | 87.200913 | 58.420000 | 63.180160 |
| e3_dense_bm25_rrf | 0.529447 | 0.698113 | 0.554754 | 76.027308 | 53.248500 | 56.538440 |
| e4_dense_bm25_rrf_rerank | 0.739518 | 0.811321 | 0.755734 | 3040.798540 | 3008.877900 | 3309.305280 |

## E4 分类型结果

| 问题类型（E4 Dev） | 可回答题数 | Recall@5 | MRR |
|---|---:|---:|---:|
| cross_lingual | 2 | 1.0000 | 1.0000 |
| experimental_setup | 17 | 0.7647 | 0.7255 |
| keyword_exact | 3 | 1.0000 | 0.7500 |
| local_fact | 5 | 0.4000 | 0.4222 |
| multi_section | 4 | 1.0000 | 1.0000 |
| semantic_explanation | 22 | 0.8636 | 0.7500 |

## 基于 Dev 的结论与决策

- E1 的 MRR 相比 E0 的变化：+0.1677；说明当前中文问英文论文场景中，稀疏词项匹配是必要通道。
- E3 的 Recall@5 相比 E2 的变化：+0.1404；本轮用 RRF 替代简单并集具有可测的召回收益。
- E4 的 MRR 相比 E3 的变化：+0.2101，但 P95 延迟增加 58.5 倍；Reranker 应作为高精度模式，而不是默认交互链路。
- 后续 P1-D 默认采用 E3（Dense + BM25 + RRF）作为可交互基线；E4 保留为深度精读/离线评测模式。该决策仅来自 Dev，未使用 Test 调参。

## 一次性留出 Test 结果

E4 配置在完成 Dev 复核后冻结；下表为对 24 条留出论文题的首次运行结果，未据此修改任何参数。由于题目由人工编写，它是论文级留出测试，而不是对标注者完全不可见的盲测。

| 指标 | 数值 |
|---|---:|
| 可回答问题数 | 20 |
| MRR | 0.405556 |
| Recall@5 | 0.550000 |
| nDCG@5 | 0.437204 |
| P95 检索耗时（ms） | 3356.948260 |

## 待人工复核的失败案例

以下是自动筛出的最多 20 条失败/误召回案例。复核时需区分解析错误、Gold 页码问题、分块问题、召回问题与重排问题。

| Case ID | 类型 | 问题 | 原因 |
|---|---|---|---|
| ai_cs_v01_q003 | local_fact | PPO 对每一批通过与环境交互采样得到的数据允许采用什么更新方式？ | 未召回 Gold 页 |
| ai_cs_v01_q005 | experimental_setup | 这篇 RAG 论文训练时使用了什么 GPU 配置？ | 未召回 Gold 页 |
| ai_cs_v01_q010 | experimental_setup | 论文在 ImageNet 上如何设置 ResNet 的训练过程？请说明输入处理、Batch Normalization、优化器、batch size、初始学习率、weight decay 和 momentum。 | 未召回 Gold 页 |
| ai_cs_v01_q015 | semantic_explanation | Transformer 的 encoder 和 decoder 分别由什么结构组成？两者最主要的结构差异是什么？ | 未召回 Gold 页 |
| ai_cs_v01_q018 | semantic_explanation | Transformer 为什么需要 positional encoding？论文采用的 positional encoding 是如何构造的，为什么最终选择这种方法而不是 learned positional embedding？ | 未召回 Gold 页 |
| ai_cs_v01_q020 | experimental_setup | Transformer 在训练过程中使用什么优化器和学习率策略？base model 和 big model 的训练时间及硬件配置分别是什么？ | 未召回 Gold 页 |
| ai_cs_v01_q027 | experimental_setup | 论文在 MuJoCo 连续控制实验中如何比较不同 surrogate objectives？哪种设置取得了最高的平均归一化得分？ | 未召回 Gold 页 |
| ai_cs_v01_q040 | local_fact | LoRA 中矩阵 (A) 和 (B) 如何初始化？为什么训练开始时 LoRA 不会改变预训练模型原本的输出？ | 未召回 Gold 页 |
| ai_cs_v01_q042 | semantic_explanation | 为什么 LoRA 相比 adapter 不会增加额外的推理延迟？在部署时 LoRA 权重如何与原模型权重合并？ | 未召回 Gold 页 |

逐题 E4 运行记录：`data/runtime/evaluations/e4_dense_bm25_rrf_rerank_dev_20260817T112707Z.jsonl`。

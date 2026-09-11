# P1-C：检索消融实验协议

## 目标

在相同语料、解析器、分块、Embedding 与 Top-K 下，分别测量 BM25、Set Union、RRF 与 Reranker 的收益和代价。

## 固定条件

- 语料：`ai_cs_eval_v0.1`，10 篇已冻结 SHA-256 的 AI/CS 论文；
- 解析与分块：`pdfplumber` + `page_parent_child_v1`；
- Embedding：`BAAI/bge-small-zh-v1.5`；
- 子 Chunk 候选数：Dense 与 BM25 各 Top-10；最终父 Chunk Top-10；
- Query Expansion 与生成模型：全部关闭；
- RRF：仅 E3/E4，`k=60`；Reranker：仅 E4，`BAAI/bge-reranker-base`。

## 实验边界

| 实验 | 策略 | 融合 | 精排 |
|---|---|---|---|
| E0 | Dense | 无 | 无 |
| E1 | BM25 | 无 | 无 |
| E2 | Dense + BM25 | Set Union | 无 |
| E3 | Dense + BM25 | RRF | 无 |
| E4 | Dense + BM25 | RRF | BGE Reranker |

当前产品链路 `H0-current-hybrid` 含 Query Rewrite，不属于以上公平消融。

## 数据使用协议

- E0--E4 的实现、调试、选择和错误分析只看 53 条 Dev；
- Test 的 24 条题只在配置冻结后运行并记录；不得据此调整参数；
- 评测记录必须保留配置哈希、BM25 Manifest 指纹、候选分数和阶段耗时；
- Parent-Child 在本轮保持不变，其效果在 P1-D 通过固定分块与结构感知分块单独评估。

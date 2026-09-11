# P1-D：结构感知解析与 E5 实验协议

## 实验问题

在固定评测语料、Embedding、BM25、RRF、Reranker 与 Top-K 的条件下，结构感知解析和分块是否能改善 AI/CS 论文的页级证据召回？

## 对照边界

| 实验 | Parser / Chunker | 检索链路 | 数据用途 |
|---|---|---|---|
| E4 | `pdfplumber` + `page_parent_child_v1` | Dense + BM25 + RRF + BGE Reranker | 已冻结对照 |
| E5 | Docling 结构 Block + `structure_aware_v1` | 与 E4 相同 | Dev 调整，Test 单次汇报 |

P1-C 的 E4 基线已经冻结为 Git Tag `p1c-final`：Dev MRR `0.740`、Recall@5 `0.811`；Test MRR `0.406`、Recall@5 `0.550`。

## Parser 接口与质量报告

所有 Parser 都必须输出页码可追溯的 `DocumentBlock`，并记录：

- `parser_name`、`parser_version`、`ocr_used`；
- 页数、文本页数、空页数、字符数、标题数、表格数、解析耗时；
- `normal`、`low_text` 或 `failed` 的解析质量状态。

扫描件空文本不能建立索引。OCR 回退属于 P1-D.1，不纳入 E5 主实验，以免同时改变 Parser、文字准确率和分块变量。

## 数据使用纪律

- E5 的实现、参数选择、解析质量分析和失败案例分析只使用 53 条 Dev；
- 只有 E5 配置冻结后，才对 24 条 Test 运行一次；
- Test 由人工编写，属于论文级留出集，不宣称是完全盲测；
- 改 Parser 或 Chunker 必须重建独立 Scope，不能覆盖 E4 索引。

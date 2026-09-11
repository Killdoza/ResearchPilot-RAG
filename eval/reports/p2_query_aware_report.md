# P2：Query-aware Evidence-Constrained Retrieval

## 研究问题

在不改变 E5 Parser/Chunker、Embedding、BM25、RRF、Reranker 和生成模型的前提下，论文级 Scope、均衡候选、Coverage Gate、一次扩检与 G2 引用协议能否改善跨论文证据覆盖与可信回答？

## 协议

- 数据：P2 v0.2，Dev 24、论文级留出 Test 12；Test 仅一次。
- Static：固定全局 Top-K；Oracle：Gold Scope 上界；Scoped：用户显式选择论文；End-to-End：标题/别名规则解析。
- Gold 只用于离线评分；Planner 的运行时输入不含 Gold。
- E6：每论文 quota=3、G1 阈值 0.15、最多一次扩检、G2 Claim-Citation-Context 验证。

## Test 结果

| 模式 | Per-paper Recall | Evidence Coverage | Citation F1 | 正确拒答率 | P50 E2E ms |
|---|---:|---:|---:|---:|---:|
| Static | 0.125 | 0.000 | 0.146 | 0.500 | 6159 |
| Oracle（上界） | 0.562 | 0.375 | 0.449 | 0.250 | 10480 |
| Scoped（可部署） | 0.562 | 0.375 | 0.347 | 0.250 | 10298 |
| End-to-End（可部署） | 0.562 | 0.375 | 0.424 | 0.250 | 11605 |

## 回答

- **E6 是否改善来源覆盖？** 是。Scoped/End-to-End 相对 Static 明显改善 Per-paper Recall、Evidence Coverage 与 Citation F1。
- **Oracle 与真实模式的差距？** Oracle 比 Scoped Citation F1 高 0.102，说明 Scope/类型上界仍有价值；End-to-End 在这批 Test 上接近 Oracle，但样本只有 12 题，不能推广为稳定结论。
- **拒答是否更安全？** G2 消除了违规比较，但正确拒答率下降；当前阈值/Gate 不能宣称已解决拒答。
- **代价？** E6 的 P50 延迟约为 Static 的 1.7--1.9 倍，且 Prompt Token 更高。
- **产品决策？** 仅作为跨论文严格证据模式 Feature Flag，不替换普通单论文链路。

## 局限

P2 v0.2 没有严格同条件的 `comparison_allowed=true` 性能比较题；结论针对来源覆盖、引用与拒答。答案语义正确性未用 LLM Judge 伪造，仍需新的人工复核或校准后的 Judge 研究。

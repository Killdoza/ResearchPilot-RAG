# P2 Test 后失败审计

Test 在 `docs/p2_dev_freeze.md` 冻结后每模式仅运行一次；本文件只解释结果，不据此调参。

## 结论

1. E6 **显著改善了跨论文页级证据覆盖**：Static Test 的 Per-paper Recall / Evidence Coverage 为 `0.125 / 0.000`，Scoped 为 `0.562 / 0.375`，End-to-End 为 `0.562 / 0.375`。
2. E6 也提升了页级引用指标：Static Citation F1 `0.146`，Scoped `0.347`，End-to-End `0.424`。Oracle `0.449` 只是 Gold Scope 上界，不能当作产品结果。
3. E6 没有解决拒答可靠性：Scoped / End-to-End 正确拒答率仅 `0.250`，低于 Static `0.500`；Test 中 Gate 全部通过、没有触发扩检，说明阈值 0.15 在该留出集上仍过宽。
4. G2 继续阻止无证据的比较/伪造引用；四模式违规比较率为 0。但 G2 拦截的格式错误仍会损失可回答题。

## 主要失败类型

| 类型 | 证据 | 后续处理（非本次 Test 调参） |
|---|---|---|
| 关键页召回不足 | E6 覆盖仍仅 0.375 | P2.5 单变量表示/重排实验，或 query decomposition 对照 |
| 拒答阈值泛化不足 | Test Gate 通过 1.000，拒答率低 | 新版 Dev 数据上校准分数与 margin，不重用本 Test |
| 生成 JSON/G2 不稳定 | 各模式 1--6 次协议拦截 | 后续独立 Prompt/结构化输出版本实验 |
| 端到端 Scope | End-to-End 与 Scoped 覆盖相同，但延迟更高 | 保留为显式 Scope 来源 Trace，不默认猜测论文 |

## 产品结论

不将 E6 设为普通单论文页面默认链路；可作为“跨论文严格证据模式”Feature Flag。普通单论文继续使用 P1 已验证的 E5/A0。下一主线进入 P2.8 工程闭环；P2.5 或拒答校准必须用新的 Dev/Test 版本开始。

# P2.8.4-D product-rc2 最终验收

## 验收说明

- 验收日期：2026-09-09～2026-09-10；
- 数据集：`eval/datasets/product_trust_regression_v0.3.jsonl`；
- 产品 Profile：`high_precision`；
- 测试论文：LightRAG、EfficientRAG；
- TR017、TR018 按测试设计主动使用 EfficientRAG，不视为错误论文切换；
- 最终结果由完整 18 条运行与受影响用例的后续专项复测合并而成；
- `scripts/analyze_product_acceptance.py` 对合并结果执行确定性门禁。

## 18 条用例结论

| ID | 结论 | 实际 Route / 关键 Trace | Citation 页码 | 复核备注 |
| --- | --- | --- | --- | --- |
| TR001 | 通过 | `experimental_results`；结构化结果选择 | 7 | Legal 四组 Overall 对比由表格结构确定性生成，数值归属与方向正确。 |
| TR002 | 通过 | `experimental_results`；结构化结果选择 | 7 | GraphRAG/LightRAG 四个数据集 Overall 结果完整；Mix 上 GraphRAG 50.4% 高于 LightRAG 49.6%，未再声称全面胜出。 |
| TR003 | 通过 | `experimental_results`；表格与邻近正文映射 | 8 | `-High/-Low/-Origin` 含义与数值方向通过结构化回放，不再把消融含义写反。 |
| TR004 | 通过 | `experimental_results`；成本字段结构化回答 | 9 | 区分 610,000 tokens 与 API 调用公式，没有改写成 610,000 次调用。 |
| TR005 | 通过 | `experimental_results`；结构化结果选择 | 7 | 三个代表性数字的方法、数据集、指标和值均可回放，胜负方向正确。 |
| TR006 | 通过 | 首次 `summary_context_merge`；重复请求 `paper_summary_cache_hit` | 1、3、7、8、9、10 | 第二次命中完整 AnswerArtifact 缓存，正文、Citation、Claim 与数字警告一致。 |
| TR007 | 通过 | `document_summary`；结构校验与本地安全修复 | 1、3、5、7、8、9、10 | 无空章节、无空表格；证据不足部分明确标记，不进行第二次模型生成。 |
| TR008 | 通过 | `document_summary`；独立摘要意图缓存键 | 1、3、5、7、8、10 | 未错误复用 TR006/TR007；贡献、证据、局限和未明确边界结构完整。 |
| TR009 | 通过 | `multi_section`；`role_cited:method/results` | 1、2、5、7 | 方法与结果角色均经过 retrieved→selected→cited；最终 Claim 均有 Citation。 |
| TR010 | 通过 | `document_detail` | 2、4、9 | 增量更新机制回答无 unsupported Claim。 |
| TR011 | 通过 | `multi_section`；`role_cited:problem/method` | 1、2、5 | 问题与方法角色均进入最终引用，不再用“召回过”冒充“已引用”。 |
| TR012 | 通过 | A→B 两步 `document_detail` | B：1、2、4 | LightRAG 切换至 EfficientRAG 后无论文事实或工作记忆污染。 |
| TR013 | 通过 | B→A `document_detail` | A：2、9 | EfficientRAG 切回 LightRAG 后无污染；事实 Claim 全部有 Citation。 |
| TR014 | 通过 | `general_chat` | 无 | 普通问候不进行论文检索，也不显示 Citation。 |
| TR015 | 通过 | `document_detail`；批注保存 | 2、9 | 论文原文与用户观点分开，批注不冒充论文结论。 |
| TR016 | 通过 | `document_detail`；论文级批注召回 | 2、8 | 即使检索到不同 Chunk，仍可根据论文、章节、页码与相关性找回 TR015 批注。 |
| TR017 | 通过 | `metadata_unsupported_field` | 无 | EfficientRAG 未提供个人年龄；结构化拒答，`refused=true`、Citation=0。 |
| TR018 | 通过 | `full_document_scan`；扫描 316 Blocks，命中 1 | 1 | 在 EfficientRAG 全文找到 `aka.ms/efficientrag`，返回原文页码；没有把 Top-K 缺失误判为全文缺失。 |

## 产品工作流与视觉

| 项目 | 结果 | 说明 |
| --- | --- | --- |
| 首 token 前瞬时失败恢复 | 通过 | 自动故障注入覆盖只重试一次；首 token 后不自动重试。 |
| 失败轮次进入记忆 | 通过 | 失败与未完成消息被 `memory_eligible_messages` 排除。 |
| 重复保存研究笔记 | 通过 | 相同回答使用稳定 note_id，不重复增加；可删除和 Markdown 导出。 |
| Citation 显示证据编号与页码 | 通过 | 多种模型 Citation 写法统一为产品格式，再裁剪未使用证据。 |
| 390×844 | 通过 | Sidebar 默认折叠，Logo、副标题、提示条与输入框无重叠。 |
| 768×1024 | 通过 | 主区完整，交互无阻塞。 |
| 1366×900 | 通过 | Sidebar、品牌区与聊天输入布局正常。 |
| 1440×900 @ 125% | 通过 | 缩放后无覆盖或裁切。 |
| 1920×1080 | 通过 | 宽屏布局比例正常。 |

## 冻结门禁

| 门禁 | 结果 |
| --- | ---: |
| Route 错误 | 0 |
| 错误论文 | 0 |
| 表格行列映射错误 | 0 |
| 数字归属错误 | 0 |
| 胜负方向错误 | 0 |
| 编造 URL / 代码地址 | 0 |
| 摘要空章节 / 空表格 | 0 |
| 缓存前后答案变化 | 0 |
| 跨论文记忆或批注污染 | 0 |
| 事实 Claim 无 Citation | 0 |
| Citation 无页码 | 0 |
| 390 px 布局阻塞 | 0 |
| 18 条回归失败 | 0 |
| Pytest | 274 passed |
| Ruff | 全仓通过 |
| 编译、依赖、密钥扫描 | 全部通过 |

结论：满足《下一步方案 9.9》的 `product-rc2` 冻结条件，可以创建标签。

## 仍需诚实保留的边界

1. 预期页覆盖为 36/42（85.7%），并非所有 Gold 页都进入最终证据；
2. 自然语言 Claim 当前完成格式、数字与引用存在性约束，尚未实现模型级语义蕴含验证；
3. 结构化表格主要在当前两篇文本型英文论文上验证，不能外推为任意复杂表格均可靠；
4. 扫描件、图片表格、流程图与曲线图尚未进入多模态证据链；
5. 费用字段需由部署者配置当期单价，否则只报告 Token，不估算金额；
6. 当前是研究原型与公开演示候选，不应表述为生产级多租户系统。

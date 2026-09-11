# P2.8.4 可信回答与记忆边界协议

## 1. 目标与非目标

本阶段只解决产品可信度边界：Claim 与 Citation 对齐、定量数字忠实度、论文级工作记忆隔离、用户批注与论文事实分离，以及产品运行时统一编排。它不更换 Parser、Embedding、Reranker 或向量库，因此不能把指标变化解释为新的检索算法收益。

本阶段不调用第二个 LLM Judge。数字校验采用确定性规则，避免增加费用和不可复现性。

## 2. 回答契约

每个论文事实 Claim 保存 `claim_id`、`text`、`citation_ids`、`support_status` 与标准化数字。Citation 只能引用本轮最终上下文，编号 `e1...eN` 与页面展示的“证据 1...N”一一对应。

支持状态：

- `supported`：引用存在，且 Claim 中的数字可在对应证据中找到；
- `unsupported`：没有合法 Citation；
- `numeric_unsupported`：引用存在，但具体数字未在该引用原文中出现；
- `not_applicable`：拒答或不需要论文事实的文本。

若数字未通过校验，产品删除包含该数字的整条输出行，并显示核验提示。第一版只验证“数字是否出现在绑定证据”，不声称验证数字之间的因果或比较逻辑。

## 3. 记忆边界

工作记忆键从 `user_id + session_id` 升级为：

```text
user_id + session_id + scope_key
```

`scope_key` 只能是单论文 `paper:<paper_id>`、多论文 `crossdoc:<stable_hash>` 或普通聊天 `general`。Query Rewrite 与近期对话只读取同一 Scope 的消息。旧消息没有 Scope 时，仅允许在 `general` 中读取，不能猜测属于某篇论文。

长期记忆遵循同一边界：用户语言/格式偏好可以跨论文复用；论文结论和遗留问题只有 `source_doc` 命中当前 `paper_id` 时才能注入，即使论文 A 与论文 B 位于同一会话也不能互相读取。普通聊天只读取 `source_doc IS NULL` 的当前会话结论。

## 4. 批注边界

批注存入独立的 `rp_annotations`，固定标记 `verification_status=user_note`。默认“仅按论文原文回答”；用户主动选择“并列参考我的批注”后，系统才会把批注作为个人观点注入，且冲突时必须并列展示。

## 5. 回归协议

当前冻结数据集为 `eval/datasets/product_trust_regression_v0.3.jsonl`，共 18 条：覆盖定量结果、全文总结、跨页 Claim、记忆隔离、批注冲突与不可回答问题。自动测试只验证协议、数据结构和确定性边界；真实模型答案仍需产品回归复核，不能把单元测试写成“语义答案全部正确”。

冻结前 Dry-run 结果为 `18/18` Route 匹配、`0` 次模型调用。该结果证明路由契约稳定，不代表 18 条真实生成答案已经人工通过。

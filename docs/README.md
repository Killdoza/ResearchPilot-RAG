# ResearchPilot 文档导航

本目录只保留理解、运行和复现当前公开版本所需的文档。内部 Sprint 草稿、逐轮冒烟日志和过期路线图未进入公开仓库；完整实验结论以 `eval/reports/` 中的四份最终报告为准。

## 第一次了解项目

1. [`architecture.md`](architecture.md)：模块、数据流与运行时边界；
2. [`product_pipeline_alignment.md`](product_pipeline_alignment.md)：实验能力如何进入真实 Streamlit/FastAPI 产品链路；
3. [`product_runtime_orchestration.md`](product_runtime_orchestration.md)：任务路由、论文 Scope、记忆与拒答编排；
4. [`api_contract.md`](api_contract.md)：API 请求、响应和错误契约；
5. [`deployment_linux.md`](deployment_linux.md)：本地与 Docker 部署；
6. [`release_v1.1.0.md`](release_v1.1.0.md)：当前公开版本的能力和限制；
7. [`demo_script_v1.1.md`](demo_script_v1.1.md)：2--3 分钟项目演示与简历表述。

## 当前产品可信度

- [`p2_8_4_trustworthy_answer_protocol.md`](p2_8_4_trustworthy_answer_protocol.md)：Citation、Claim、数字、表格和摘要契约；
- [`p2_8_4_d_product_rc2_review.md`](p2_8_4_d_product_rc2_review.md)：18 条真实回归与视口验收；
- [`p2_8_5_research_workflow_mvp.md`](p2_8_5_research_workflow_mvp.md)：PDF 页核验、研究笔记与 Markdown 导出。

当前门禁事实为：274 项自动测试通过；真实产品回归 18/18；固定预期页覆盖 36/42（85.7%）。后两个数字只适用于冻结回归集，不代表任意论文 100% 正确。

## 核心实验

- [`../eval/reports/p1c_retrieval_ablation.md`](../eval/reports/p1c_retrieval_ablation.md)：Dense、BM25、Union、RRF、Rerank 消融；
- [`../eval/reports/p1d_parser_chunking_report.md`](../eval/reports/p1d_parser_chunking_report.md)：E4/E5 解析与分块对照；
- [`../eval/reports/p1e_long_context_report.md`](../eval/reports/p1e_long_context_report.md)：Long-context 与 RAG 的质量/成本取舍；
- [`../eval/reports/p2_query_aware_report.md`](../eval/reports/p2_query_aware_report.md)：无 Oracle 泄漏的跨论文 Scope 与证据 Gate 实验。

## 数据与标注协议

- [`annotation_guide.md`](annotation_guide.md)：单论文评测标注；
- [`crossdoc_annotation_guide.md`](crossdoc_annotation_guide.md)：跨论文评测标注；
- [`p2_annotation_guide.md`](p2_annotation_guide.md)：P2 Query-aware 数据集标注；
- [`p2_protocol.md`](p2_protocol.md)：P2 Gold/Runtime 输入边界与 Test 纪律。

## 工程与发布

- [`p2_8_performance.md`](p2_8_performance.md)：仅记录已实际测量的性能边界；
- [`release_v1.1.0.md`](release_v1.1.0.md)：公开版本的功能、指标和已知限制；
- [`demo_script_v1.1.md`](demo_script_v1.1.md)：公开演示脚本。

## 历史审计如何阅读

`p1d_page_audit.md`、`p1e_dev_answer_review.md` 和 `p1e_a0_dev_spot_review.md` 是最终报告直接引用的人工复核附件；`p2_failure_audit.md` 是跨论文实验的最终失败分类。它们用于支撑结论，不是产品运行日志。

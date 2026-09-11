# P1-D：E5 页面级解析人工复核

## 目的

本复核确认 E5 的 Dev 提升不是由错误页码、阅读顺序异常或过宽页码范围造成。每页应同时核对原 PDF 与对应的结构 Block JSON：

```text
data/raw/ai_cs_eval_v0.1/<论文文件>
data/runtime/parsed_blocks/e5_structure_aware/<论文 SHA-256>.json
```

Block JSON 是本地运行产物，不提交 Git；复核只需确认其 `page`、`section_path`、`block_type` 与原页面一致。

## 判定规则

- `通过`：标题层级、正文阅读顺序、页码和表格/图注归属均可接受；
- `可接受瑕疵`：有少量文字或格式噪声，但不影响页面证据定位；
- `失败`：关键正文顺序错乱、表格文本不可用、页码错配或漏掉关键内容；
- `需修订 Gold`：原有 Gold 页码或问题本身不正确，必须说明原因。

## 建议复核页

| 论文 | PDF 页 | 检查重点 | 结论 | 备注 |
|---|---:|---|---|---|
| ResNet | 4 | §3.4 训练配置、连续正文 | 通过 | PDF 第 4 页的 Fig. 3 图注、Residual Network 段落、`3.4 Implementation`、测试设置与 `4.1` 标题均在对应页 Block 中；标题层级和左图—右栏正文的阅读顺序可用。图像本身不进入文本索引，但图注被单独保留。 |
| ResNet | 8 | 实验表格与图注 | 可接受瑕疵 | Fig. 6/7、Table 7/8、`4.3` 标题与后续正文均定位在 PDF 第 8 页；两张表以 Markdown 表格 Block 保留，图/表图注也单独保留。图注同时出现在相邻的图片占位段落和 caption Block 中，存在重复，但未造成页码或归属错误。 |
| Transformer | 3 | Encoder/Decoder 标题与阅读顺序 | 可接受瑕疵 | `3 Model Architecture`、`3.1 Encoder and Decoder Stacks`、Encoder、Decoder 和 `3.2 Attention` 的顺序与原页一致；Fig. 1 只保留图片占位与图注，未提取图内文字，故不把该页作为视觉问答能力的证据。 |
| Transformer | 6 | §3.5 位置编码公式/正文 | 可接受瑕疵 | 初检发现 Docling 的 FormulaItem 已识别但适配器忽略其 `orig`，导致位置编码公式丢失；已修复并重建 E5 Scope。复核后的 Block 含独立 `formula`（正弦/余弦位置编码）及 `3.5 Positional Encoding` 正文，页码和章节正确。线性化公式的空格、上标格式不完全等同于排版公式。 |
| Transformer | 7 | 训练配置、表格与页码 | 可接受瑕疵 | `5 Training`、`5.1`、`5.2`、`5.3`、`5.4` 的层级和 WMT 数据、8×P100、Adam/学习率正文均在第 7 页且顺序正确；学习率公式在修复后作为独立公式 Block 保留，数学排版降为线性文本。 |
| PPO | 6 | MuJoCo 实验表格与结论 | 通过 | `6.1` 的 7 个 MuJoCo 任务、评分方式、Table 1 与 `6.2` 均被顺序解析；Table 1 为独立 Markdown 表格并绑定同页 `6.1` 路径，数值和图注可定位。 |
| RAG | 17 | Appendix C、GPU 配置 | 通过 | 附录标题、`C Training setup Details` 与“8 张 32GB NVIDIA V100 GPU”的训练配置均在 PDF 第 17 页；链接脚注另列为段落，但不影响 GPU 证据的页码和章节归属。 |
| LoRA | 4 | §4.1、公式与初始化描述 | 可接受瑕疵 | `4 OUR METHOD`、`4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES`、随机高斯初始化 A/零初始化 B 的正文均准确保留；公式 (3) 在修复后成为独立公式 Block。其线性化结果有符号/空格轻微失真，因此精确公式抄写应回看原 PDF，不能把它当作数学公式无损转写。 |
| SimCLR | 4 | 方法段落与图注 | 可接受瑕疵 | Fig. 4、Fig. 5 图注、数据增强段落和 `3.1` 标题均保留在第 4 页；双栏页面以左栏优先方式输出，Fig. 5 的右栏说明相对视觉顺序略后。文字证据和章节归属仍可用，图片内容本身不进入当前文本索引。 |
| MapReduce | 4 | locality optimization 正文 | 需修订 Gold | 复核确认 PDF 第 4 页是 `3.1 Execution Overview`、`3.2 Master Data Structures` 和 `3.3 Fault Tolerance`，不含 locality optimization。评测集 q050 的 Gold 已正确标为第 5 页，因此**无需改评测数据**；需修订的是本审计表原先“第 4 页 / locality”这一抽查项，后续应抽查第 5 页。第 4 页本身的双栏列表与标题顺序可用。 |

## 复核结论与已修正项

- 10 个指定 PDF 页面均已以原 PDF 的视觉版式和 E5 Block JSON 交叉核验；未发现系统性的页码错配或整页正文丢失。
- 本次发现并修复 1 个会影响证据完整性的适配器问题：Docling `FormulaItem` 的文本位于 `orig` 字段。修复后，公式以独立 `formula` Block 进入结构化分块；E5 Scope 和 Dev 结果均已重新生成。
- 当前可接受的边界是：图像内容不做视觉识别、图注或图片占位可能重复、双栏页存在少量自然阅读顺序瑕疵、公式为近似线性化文本。这些限制将在后续的多模态/OCR 阶段单独评测，不能被表述为“无损文档理解”。
- `MapReduce` 行暴露的是**审计抽样页写错**，不是 q050 的 Gold 页码错误：q050 仍以 PDF 第 5 页为准。

根据上述复核，E5 的 Parser/Chunker 配置可冻结；接下来只运行一次 Test，Test 的结果不会再用于修改本 Sprint 的配置。

## 复核后的行动

1. 若全部为“通过”或“可接受瑕疵”，E5 配置冻结；随后仅运行一次 Test。
2. 若出现“失败”，先只在 Dev 上修复 Parser/Chunker 或排除不可信 Block，再重新跑 E5 Dev。
3. Test 结果不得用于修改结构化分块参数。

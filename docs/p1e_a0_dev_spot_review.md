# P1-E A0 Dev 代表性答案复核包（10 条）

本复核包用于在冻结 Test 协议前抽查 A0 Advanced RAG 的五条单论文与五条跨论文答案。自动 Citation 指标仅检查论文键与页码是否覆盖 Gold，不等于答案事实正确性。

## 01. A0 Advanced RAG · ai_cs_v01_q006

**问题**：论文所说的深层神经网络的“degradation problem（退化问题）”具体指什么？作者为什么认为它不是由过拟合造成的？

**模型回答**：

退化问题指的是：随着网络层数增加，训练误差不降反升的现象。作者认为这不是过拟合造成的，因为过拟合通常表现为训练误差降低而验证误差升高，而退化问题中训练误差本身也更高。作者通过实验（如增加3倍训练迭代次数）观察到退化问题仍然存在，说明单纯增加迭代次数无法解决，从而排除了欠拟合或优化不充分的可能性。

**模型引用**：resnet_2015 p.3-3; resnet_2015 p.3-3; resnet_2015 p.5-5; resnet_2015 p.4-4

**自动页级结果**：Citation P/R/F1 = 0.0 / 0.0 / 0.0；正确拒答 = None。

**人工核验基准**：

预期答案：退化问题是指：随着网络深度增加，模型的准确率先趋于饱和，随后反而迅速下降；具体表现为更深的网络不仅测试误差更高，**训练误差本身也更高**。作者因此认为这种退化不是过拟合造成的，因为如果是过拟合，通常应表现为训练误差降低而测试误差升高；这里增加网络层数后连训练性能都变差，说明主要存在优化困难。
Gold 证据：
- resnet_2015 第 1-1 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[ ] 正确  [x] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：退化问题及“训练误差也升高，所以并非过拟合”的核心判断正确；但把“三倍迭代仍存在”表述为排除优化不足或欠拟合，超出了该题 Gold 直接支持的结论，且论文的要点恰是更深模型存在优化困难。

---

## 02. A0 Advanced RAG · ai_cs_v01_q010

**问题**：论文在 ImageNet 上如何设置 ResNet 的训练过程？请说明输入处理、Batch Normalization、优化器、batch size、初始学习率、weight decay 和 momentum。

**模型回答**：

根据提供的论文上下文，无法回答关于 ImageNet 上 ResNet 训练过程的具体设置（输入处理、Batch Normalization、优化器、batch size、初始学习率、weight decay 和 momentum）的问题。上下文中的 R01 描述的是 CIFAR-10 上的训练设置，而其他上下文（R02-R10）来自 SimCLR 论文，涉及自监督学习或迁移学习，并非 ResNet 在 ImageNet 上的标准监督训练。因此，证据不足，无法给出准确答案。

**模型引用**：无

**自动页级结果**：Citation P/R/F1 = 0.0 / 0.0 / 0.0；正确拒答 = None。

**人工核验基准**：

预期答案：ImageNet 训练时，首先将图像短边随机缩放到 ([256,480]) 范围，然后从原图或其水平翻转图像中随机裁剪 (224\times224) 区域，并减去逐像素均值，同时使用颜色增强。

每个卷积层之后、激活函数之前使用 Batch Normalization。网络从头训练，采用 SGD，mini-batch size 为 **256**；初始学习率为 **0.1**，当误差进入平台期时将学习率除以 10；weight decay 为 **0.0001**，momentum 为 **0.9**，并且**不使用 dropout**。
Gold 证据：
- resnet_2015 第 4-4 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[ ] 正确  [ ] 部分正确  [x] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：错误拒答。Gold 位于 ResNet 第 4 页，完整覆盖输入增强、BN、SGD、batch size、学习率、weight decay 与 momentum；A0 还暴露出检索候选混入 SimCLR 的噪声，但未把 SimCLR 事实冒充为 ResNet 结论。

---

## 03. A0 Advanced RAG · ai_cs_v01_q019

**问题**：论文从哪三个方面比较 Self-Attention、Recurrent 和 Convolutional layer？Self-Attention 在顺序计算次数和最长路径长度方面有什么优势？

**模型回答**：

论文从三个方面比较 Self-Attention、Recurrent 和 Convolutional layer：1) 每层的总计算复杂度；2) 可并行化的计算量，以所需的最小顺序操作数衡量；3) 网络中长距离依赖的路径长度。Self-Attention 在顺序计算次数方面具有优势，因为它以常数数量的顺序操作连接所有位置（O(1)），而 Recurrent 层需要 O(n) 个顺序操作。在最长路径长度方面，Self-Attention 的最大路径长度为 O(1)，而 Recurrent 层为 O(n)，Convolutional 层为 O(log_k(n))，因此 Self-Attention 能更轻松地学习长距离依赖。

**模型引用**：transformer_2017 p.6-6; transformer_2017 p.6-6; transformer_2017 p.6-6

**自动页级结果**：Citation P/R/F1 = 1.0 / 1.0 / 1.0；正确拒答 = None。

**人工核验基准**：

预期答案：作者主要从三个方面进行比较：

1. 每层的总计算复杂度；
2. 可并行计算程度，即最少需要多少次 sequential operations；
3. 任意两个位置之间的最大路径长度。

对于序列长度为 (n)、表示维度为 (d) 的情况：

Self-Attention 每层复杂度为

[
O(n^2d),
]

最少顺序操作次数为

[
O(1),
]

最大路径长度也是

[
O(1).
]

相比之下，Recurrent layer 的顺序操作次数和最大路径长度均为

[
O(n).
]

因此 Self-Attention 可以一次直接连接序列中的所有位置，不需要像 RNN 一样沿序列逐步传播信息，这既有利于并行计算，也缩短了长距离依赖之间的信息传播路径。论文还指出，当通常的 NLP 场景满足 (n<d) 时，Self-Attention 的计算复杂度也低于 recurrent layer。
Gold 证据：
- transformer_2017 第 6-7 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：完整给出 Table 1 的三个比较维度，以及 Self-Attention 相对 recurrent / convolution 的 O(1)、O(n)、O(log_k(n)) 路径与顺序操作结论；页码命中正确。

---

## 04. A0 Advanced RAG · ai_cs_v01_q026

**问题**：在 Actor-Critic 形式的 PPO 中，如果 policy 和 value function 共享网络参数，最终优化目标由哪三部分组成？PPO 每轮更新的数据采集和优化流程是什么？

**模型回答**：

根据提供的论文上下文，无法回答该问题。上下文仅包含 PPO 论文的摘要、算法伪代码片段和关于裁剪替代目标的图注，未提及 Actor-Critic 形式下 policy 和 value function 共享网络参数时的最终优化目标组成，也未描述 PPO 每轮更新的数据采集和优化流程。

**模型引用**：无

**自动页级结果**：Citation P/R/F1 = 0.0 / 0.0 / 0.0；正确拒答 = None。

**人工核验基准**：

预期答案：当 policy 和 value function 共享参数时，论文使用的联合目标包括三部分：

1. **Clipped policy surrogate objective** (L^{CLIP})；
2. **Value function loss** (L^{VF})；
3. **Entropy bonus** (S[\pi_\theta])，用于鼓励探索。

整体目标为：

[
L_t^{CLIP+VF+S}(\theta)
=======================

\hat{\mathbb E}_t
\left[
L_t^{CLIP}(\theta)
------------------

c_1L_t^{VF}(\theta)
+
c_2S[\pi_\theta](s_t)
\right].
]

其中 value function loss 是预测状态价值与目标价值之间的平方误差。

在 Actor-Critic PPO 中，每轮迭代时，首先让 **(N) 个并行 actor 各运行 (T) 个时间步**，因此总共收集 (NT) 个样本；然后计算 advantage estimates，并基于这些样本构造 surrogate loss；最后使用 minibatch SGD，通常实际采用 Adam，对同一批数据进行 **(K) 个 epoch** 的优化，再将当前策略设置为新的旧策略进入下一轮采样。
Gold 证据：
- ppo_2017 第 5-5 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[ ] 正确  [ ] 部分正确  [x] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：错误拒答。PPO 第 5 页明确给出共享参数时的 CLIP、value loss、entropy 三部分目标及 N 个 actor 收集 NT 样本、K 个 epoch 的更新流程；说明 A0 未召回关键局部块。

---

## 05. A0 Advanced RAG · ai_cs_v01_q035

**问题**：RAG 在四个 open-domain QA 数据集上的测试表现如何？RAG-Sequence 和 RAG-Token 分别取得了哪些 Exact Match 分数？

**模型回答**：

根据论文上下文，RAG 在四个 open-domain QA 数据集（NQ、TQA、WQ、CT）上的测试表现如下：RAG-Token 在 NQ 上取得 44.1，TQA 上 55.2（标准测试集）或 66.1（TQA-Wiki），WQ 上 45.5，CT 上 50.0；RAG-Sequence 在 NQ 上取得 44.5，TQA 上 56.8（标准测试集）或 68.0（TQA-Wiki），WQ 上 45.2，CT 上 52.2。这些分数均为 Exact Match 分数。

**模型引用**：rag_2020 p.6-6

**自动页级结果**：Citation P/R/F1 = 1.0 / 1.0 / 1.0；正确拒答 = None。

**人工核验基准**：

预期答案：论文在 Natural Questions（NQ）、TriviaQA（TQA）、WebQuestions（WQ）和 CuratedTrec（CT）上进行实验，使用 Exact Match 作为主要评价指标。

测试集结果为：

| 模型           |       NQ |             TQA |   WQ |       CT |
| ------------ | -------: | --------------: | ---: | -------: |
| RAG-Token    |     44.1 |     55.2 / 66.1 | 45.5 |     50.0 |
| RAG-Sequence | **44.5** | **56.8 / 68.0** | 45.2 | **52.2** |

TQA 的两个数分别对应常规 open-domain QA test set 和 TQA-Wiki test set。

作者指出，RAG 在这些任务上获得了非常强的结果，并证明生成式模型不一定需要 extractive reader 或文档 reranker 才能达到 state-of-the-art 水平。值得注意的是，在 NQ 中，即使正确答案没有出现在任何检索文档中，RAG 仍有 **11.8%** 的准确率，而严格 extractive 模型在这种情况下只能得到 0%。
Gold 证据：
- rag_2020 第 5-6 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：四个数据集、RAG-Token / RAG-Sequence 的 Exact Match 数值和 TQA 两种测试集均与 RAG 第 6 页表格一致。

---

## 06. A0 Advanced RAG · CD001

**问题**：Transformer 论文中的模型架构设计，与 LoRA 论文中的参数高效适配方法分别解决什么问题？两者作用的位置和训练目标有什么差异？

**模型回答**：

Transformer 论文中的模型架构设计解决的是序列建模问题，通过自注意力机制捕捉序列内元素间的依赖关系，并采用多头注意力、位置编码等设计来提升模型表达能力。LoRA 论文中的参数高效适配方法解决的是大规模预训练模型在下游任务微调时计算和存储成本过高的问题，通过冻结预训练权重并注入低秩分解矩阵，大幅减少可训练参数量。两者作用位置不同：Transformer 架构设计作用于整个模型结构，包括自注意力模块和 MLP 模块；LoRA 方法则作用于 Transformer 的注意力权重矩阵（如 Wq、Wk、Wv、Wo），冻结 MLP 模块。训练目标也不同：Transformer 架构设计旨在学习通用的序列表示，而 LoRA 旨在以低秩更新矩阵逼近全量微调的参数更新，从而在保持模型质量的同时降低训练成本。

**模型引用**：lora_2021 p.1-1; lora_2021 p.4-4; lora_2021 p.5-5

**自动页级结果**：Citation P/R/F1 = 0.3333333333333333 / 0.5 / 0.4；正确拒答 = None。

**人工核验基准**：

预期答案：C1：Transformer 的目标是设计一种新的序列转换模型架构，通过完全基于注意力机制替代循环和卷积结构，提高序列建模的并行能力。
C2：LoRA 的目标是在保持预训练模型权重冻结的情况下，通过训练低秩矩阵表示参数更新，从而降低大模型下游适配的训练参数量和存储成本。
C3：两者不是同一层面的替代方法，Transformer 关注基础模型架构设计，而 LoRA 关注已有 Transformer 模型的任务适配方式。
Gold 证据：
- transformer_2017 第 1-2 页（Abstract / Introduction）
- lora_2021 第 1-2 页（Abstract / Introduction）

**请填写**：

- 答案语义：`[ ] 正确  [x] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：正确区分架构设计与参数高效适配；但只引用了 LoRA，缺少 Transformer 论文证据，并将“LoRA 冻结 MLP”说成一般性机制，超出了 Gold 对低秩更新、冻结预训练权重的表述。

---

## 07. A0 Advanced RAG · CD003

**问题**：ResNet 与 Transformer 分别把 residual connection 放在什么结构位置，并直接说明了什么作用？能否仅据两篇论文断言二者的主要设计动机完全相同？

**模型回答**：

根据提供的论文上下文，ResNet 将 residual connection（shortcut connection）添加在每对 3×3 卷积层之后，即“a shortcut connection is added to each pair of 3×3 filters”（CTX:R01），其作用是解决退化问题（degradation problem），使得更深的网络能够获得更低的训练误差和更好的泛化（CTX:R01, R05）。Transformer 论文中并未直接描述 residual connection 的具体位置和作用，仅在背景部分提到其架构基于注意力机制，未提及残差连接（CTX:R02, R08）。因此，无法仅依据这两篇论文断言二者的主要设计动机完全相同，因为 Transformer 论文中未提供相关设计动机的说明。

**模型引用**：resnet_2015 p.5-5; resnet_2015 p.5-5; transformer_2017 p.2-2; transformer_2017 p.2-2

**自动页级结果**：Citation P/R/F1 = 0.0 / 0.0 / 0.0；正确拒答 = None。

**人工核验基准**：

预期答案：C1：ResNet 使用 shortcut connection 将输入 x 直接加入网络输出，使网络学习残差函数 F(x)，用于缓解深层网络优化困难。
C2：Transformer 的 encoder 和 decoder 子层也使用 residual connection，并结合 LayerNorm。
C3：两者都采用残差连接，但 ResNet 明确将其用于缓解深层网络的优化/退化问题；Transformer 论文将其描述为子层的架构组件，未直接把其主要动机表述为与 ResNet 完全相同，因此不能据此作完全等同的断言。
Gold 证据：
- resnet_2015 第 2-2 页（Residual Learning Framework）
- transformer_2017 第 3-3 页（Encoder and Decoder Stacks）

**请填写**：

- 答案语义：`[ ] 正确  [ ] 部分正确  [x] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：A0 错称 Transformer 未描述 residual connection；原文第 3 页明确说明每个 encoder / decoder 子层使用 residual connection 后再 LayerNorm。ResNet 的论证也没有回到其核心的 x + F(x) 结构，属于跨论文证据漏召回。

---

## 08. A0 Advanced RAG · CD005

**问题**：PPO 和 LoRA 分别优化什么对象？二者如何通过限制或减少更新规模提高训练效率？

**模型回答**：

根据提供的论文上下文，PPO 和 LoRA 分别优化不同的对象。PPO（近端策略优化）是一种强化学习算法，其优化对象是策略（policy），通过裁剪的替代目标函数来限制策略更新的幅度，从而提高训练稳定性（见 [CTX:R03]）。LoRA（低秩适应）则优化注入的低秩矩阵（即 A 和 B），而冻结预训练模型的权重，通过仅优化这些较小的矩阵来减少更新规模，从而提高训练效率（见 [CTX:R02]）。关于二者如何通过限制或减少更新规模提高训练效率：PPO 通过裁剪目标函数限制策略更新的幅度，避免过大的策略变化；LoRA 通过冻结大部分参数，仅优化低秩矩阵，从而减少需要计算梯度和维护优化器状态的参数数量，降低存储和计算开销（见 [CTX:R02]）。

**模型引用**：ppo_2017 p.6-6; lora_2021 p.2-2

**自动页级结果**：Citation P/R/F1 = 0.5 / 0.5 / 0.5；正确拒答 = None。

**人工核验基准**：

预期答案：C1：PPO 优化强化学习中的策略，通过 clipped surrogate objective 限制策略更新幅度。
C2：LoRA 不直接更新全部模型参数，而优化低秩矩阵表示的参数变化。
C3：二者都对更新施加约束，但效率来源不同：PPO 控制新旧策略之间的变化范围，LoRA 减少需要训练的参数规模。
Gold 证据：
- ppo_2017 第 3-3 页（Clipped Surrogate Objective）
- lora_2021 第 2-3 页（Introduction / Problem Statement）

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：准确说明 PPO 的 clipped objective 约束策略更新幅度，以及 LoRA 冻结预训练参数、只训练低秩矩阵以减少可训练参数与优化状态。

---

## 09. A0 Advanced RAG · CD007

**问题**：ResNet 和 PPO 是否在同一个公开数据集、同一个评价指标下进行了直接实验比较？哪一个方法取得更高性能？

**模型回答**：

根据提供的论文上下文，无法确认 ResNet 和 PPO 在同一个公开数据集、同一个评价指标下进行了直接实验比较。ResNet 相关上下文（如 CTX:R03、R05、R06）主要涉及 ImageNet 分类任务，使用 top-1/top-5 准确率或错误率作为指标；PPO 相关上下文（如 CTX:R01、R10）涉及连续控制（如 MuJoCo）和 Atari 游戏，使用平均得分等指标。两者任务、数据集和指标均不同，且没有上下文显示它们在同一条件下进行了直接比较。因此，无法判断哪个方法取得更高性能。

**模型引用**：resnet_2015 p.6-6; simclr_2020 p.7-7; simclr_2020 p.7-7; ppo_2017 p.6-6; ppo_2017 p.12-12

**自动页级结果**：Citation P/R/F1 = None / None / None；正确拒答 = False。

**人工核验基准**：

预期答案：论文没有提供 ResNet 和 PPO 在同一任务、同一数据集、同一指标下的直接实验比较，因此不能判断哪个方法性能更高。
Gold 证据：
- 无；此题要求拒答，不能强行比较。

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[ ] 否  [x] 是`
- 主要错误类型 / 备注：正文正确说明两篇论文任务、数据集和指标不同，因而不能比较“谁更高”；但结构化输出仍附带引用且没有把 refused 置为 true，自动正确拒答为 false，属于拒答协议违反而非事实判断错误。

---

## 10. A0 Advanced RAG · CD009

**问题**：RAG 和 MapReduce 是否都使用“外部数据”？两篇论文中外部数据分别承担什么作用？

**模型回答**：

根据提供的论文上下文，RAG 和 MapReduce 都涉及使用外部数据，但具体含义和作用不同。RAG（检索增强生成）模型结合了参数化记忆（预训练seq2seq模型）和非参数化记忆（维基百科的密集向量索引），外部数据（维基百科）作为非参数化记忆，通过预训练的神经检索器访问，用于在生成过程中提供事实性知识，增强生成内容的准确性和具体性。MapReduce 是一种编程模型，用于处理和生成大规模数据集，其输入数据（如爬取的文档、网络请求日志等）是外部数据，通过 map 和 reduce 函数分布式处理，外部数据作为计算的对象，用于生成派生数据（如倒排索引、图结构表示等）。因此，两者都使用外部数据，但 RAG 中外部数据用于检索增强生成，而 MapReduce 中外部数据是待处理的大规模数据集。

**模型引用**：rag_2020 p.1-1; mapreduce_2004 p.1-1; mapreduce_2004 p.1-1

**自动页级结果**：Citation P/R/F1 = 1.0 / 1.0 / 1.0；正确拒答 = None。

**人工核验基准**：

预期答案：C1：MapReduce 中输入数据集是待处理的大规模数据，Map 和 Reduce 对其进行分布式计算。
C2：RAG 中外部 Wikipedia 索引作为非参数记忆，用于检索知识增强生成。
C3：二者都涉及外部数据，但 MapReduce 的数据是计算对象，RAG 的数据是生成过程中的知识来源。
Gold 证据：
- mapreduce_2004 第 1-1 页（Abstract）
- rag_2020 第 1-2 页（Abstract / Introduction）

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：正确区分 RAG 中 Wikipedia 索引作为非参数记忆、服务知识增强生成，与 MapReduce 将大规模输入数据作为分布式计算对象的角色差异。

---

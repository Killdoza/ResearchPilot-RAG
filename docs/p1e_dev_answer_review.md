# P1-E Dev 人工答案复核包（20 条）

本复核包记录已完成的 LC1、LC2、N0 共 20 条代表性人工复核。A0 的 10 条独立复核见 [p1e_a0_dev_spot_review.md](p1e_a0_dev_spot_review.md)。自动 Citation 指标仅检查论文键与页码是否覆盖 Gold，不等于答案事实正确性；以下结论已根据原论文和 Gold 证据填写。

## 01. LC1 单论文全文 · ai_cs_v01_q006

**问题**：论文所说的深层神经网络的“degradation problem（退化问题）”具体指什么？作者为什么认为它不是由过拟合造成的？

**模型回答**：

退化问题指的是：随着网络深度增加，准确率趋于饱和后迅速下降，且这种退化并非由过拟合引起，因为更深的模型在训练集上的误差反而更高。作者认为这不是过拟合，而是优化困难：即使理论上可以通过恒等映射构造出与浅层网络训练误差相同的深层网络，现有求解器（如SGD）也难以找到这样的解。实验表明，使用BN后前向/反向信号均未消失，梯度范数正常，因此也不是梯度消失；作者推测深层网络可能具有指数级低的收敛率，影响训练误差的降低。

**模型引用**：resnet_2015 p.1-1; resnet_2015 p.1-1; resnet_2015 p.5-5; resnet_2015 p.5-5

**自动页级结果**：Citation P/R/F1 = 0.5 / 1.0 / 0.6666666666666666；正确拒答 = None。

**人工核验基准**：

预期答案：退化问题是指：随着网络深度增加，模型的准确率先趋于饱和，随后反而迅速下降；具体表现为更深的网络不仅测试误差更高，**训练误差本身也更高**。作者因此认为这种退化不是过拟合造成的，因为如果是过拟合，通常应表现为训练误差降低而测试误差升高；这里增加网络层数后连训练性能都变差，说明主要存在优化困难。
Gold 证据：
- resnet_2015 第 1-1 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：核心结论与 ResNet 第 1 页的退化问题定义一致；补充的优化困难说明没有改变结论。

---

## 02. N0 Dense RAG · ai_cs_v01_q006

**问题**：论文所说的深层神经网络的“degradation problem（退化问题）”具体指什么？作者为什么认为它不是由过拟合造成的？

**模型回答**：

根据提供的论文上下文，无法回答该问题。上下文仅包含ResNet论文的摘要和部分章节，未涉及对“degradation problem”的具体定义及其与过拟合关系的讨论。

**模型引用**：无

**自动页级结果**：Citation P/R/F1 = 0.0 / 0.0 / 0.0；正确拒答 = None。

**人工核验基准**：

预期答案：退化问题是指：随着网络深度增加，模型的准确率先趋于饱和，随后反而迅速下降；具体表现为更深的网络不仅测试误差更高，**训练误差本身也更高**。作者因此认为这种退化不是过拟合造成的，因为如果是过拟合，通常应表现为训练误差降低而测试误差升高；这里增加网络层数后连训练性能都变差，说明主要存在优化困难。
Gold 证据：
- resnet_2015 第 1-1 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[ ] 正确  [ ] 部分正确  [x] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：错误拒答；Gold 证据在 ResNet 第 1 页，Dense Top-K 未召回该页的完整定义。

---

## 03. LC1 单论文全文 · ai_cs_v01_q010

**问题**：论文在 ImageNet 上如何设置 ResNet 的训练过程？请说明输入处理、Batch Normalization、优化器、batch size、初始学习率、weight decay 和 momentum。

**模型回答**：

根据论文，ImageNet 上 ResNet 的训练设置如下：
- 输入处理：图像短边在 [256, 480] 范围内随机采样进行尺度增强，然后随机裁剪 224×224 区域（或水平翻转），并减去每个像素的均值；使用标准颜色增强。
- Batch Normalization：在每个卷积之后、激活之前采用 BN。
- 优化器：使用 SGD（随机梯度下降）。
- batch size：256。
- 初始学习率：0.1，当误差趋于平稳时除以 10。
- weight decay：0.0001。
- momentum：0.9。
- 不使用 dropout。

**模型引用**：resnet_2015 p.4-4

**自动页级结果**：Citation P/R/F1 = 1.0 / 1.0 / 1.0；正确拒答 = None。

**人工核验基准**：

预期答案：ImageNet 训练时，首先将图像短边随机缩放到 ([256,480]) 范围，然后从原图或其水平翻转图像中随机裁剪 (224\times224) 区域，并减去逐像素均值，同时使用颜色增强。

每个卷积层之后、激活函数之前使用 Batch Normalization。网络从头训练，采用 SGD，mini-batch size 为 **256**；初始学习率为 **0.1**，当误差进入平台期时将学习率除以 10；weight decay 为 **0.0001**，momentum 为 **0.9**，并且**不使用 dropout**。
Gold 证据：
- resnet_2015 第 4-4 页（人工 Markdown 草稿标注）

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：ResNet 第 4 页直接给出输入增强、BN、SGD、256 batch、学习率、weight decay、momentum 与不使用 dropout；回答完整对应。

---

## 04. N0 Dense RAG · ai_cs_v01_q010

**问题**：论文在 ImageNet 上如何设置 ResNet 的训练过程？请说明输入处理、Batch Normalization、优化器、batch size、初始学习率、weight decay 和 momentum。

**模型回答**：

根据提供的论文上下文，无法完整回答关于 ImageNet 上 ResNet 训练过程的全部细节（输入处理、Batch Normalization、优化器、batch size、初始学习率、weight decay 和 momentum）。上下文仅提及了训练误差曲线、模型对比等，未包含具体的超参数设置。

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
- 主要错误类型 / 备注：错误拒答；第 4 页包含题目要求的完整训练设置，属于局部事实漏召回。

---

## 05. LC1 单论文全文 · ai_cs_v01_q019

**问题**：论文从哪三个方面比较 Self-Attention、Recurrent 和 Convolutional layer？Self-Attention 在顺序计算次数和最长路径长度方面有什么优势？

**模型回答**：

论文从三个方面比较 Self-Attention、Recurrent 和 Convolutional layer：一是每层的总计算复杂度，二是可并行化的计算量（以所需的最小顺序操作数衡量），三是网络中长距离依赖的路径长度。Self-Attention 在顺序计算次数方面，能以常数次顺序操作连接所有位置，而 Recurrent 层需要 O(n) 次顺序操作；在最长路径长度方面，Self-Attention 的最长路径长度为 O(1)，而 Recurrent 为 O(n)，Convolutional 为 O(log_k(n))。

**模型引用**：transformer_2017 p.6-6; transformer_2017 p.6-6; transformer_2017 p.6-6; transformer_2017 p.6-6; transformer_2017 p.6-6

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
- 主要错误类型 / 备注：与 Transformer 第 6 页 Table 1 和 Why Self-Attention 的三个比较维度、O(1)/O(n) 路径结论一致。

---

## 06. N0 Dense RAG · ai_cs_v01_q019

**问题**：论文从哪三个方面比较 Self-Attention、Recurrent 和 Convolutional layer？Self-Attention 在顺序计算次数和最长路径长度方面有什么优势？

**模型回答**：

论文从三个方面比较 Self-Attention、Recurrent 和 Convolutional layer：1) 每层的总计算复杂度；2) 可并行化的计算量，以最小顺序操作数衡量；3) 网络中长距离依赖的路径长度。Self-Attention 在顺序计算次数方面，以常数次顺序操作连接所有位置，而 Recurrent 层需要 O(n) 次顺序操作；在最长路径长度方面，Self-Attention 为 O(1)，而 Recurrent 为 O(n)，Convolutional 为 O(log_k(n))。因此，Self-Attention 在顺序计算和路径长度上具有优势，更易于学习长距离依赖。

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
- 主要错误类型 / 备注：回答事实完整；模型把 context_id 写成 `CTX:Rxx`，是输出格式瑕疵，不影响本题语义与页码命中。

---

## 07. LC1 单论文全文 · ai_cs_v01_q026

**问题**：在 Actor-Critic 形式的 PPO 中，如果 policy 和 value function 共享网络参数，最终优化目标由哪三部分组成？PPO 每轮更新的数据采集和优化流程是什么？

**模型回答**：

在 Actor-Critic 形式的 PPO 中，如果 policy 和 value function 共享网络参数，最终优化目标由三部分组成：策略替代目标（如 L_CLIP）、价值函数误差项（L_VF）和熵奖励项（S）。具体公式为：L_CLIP+VF+S_t(θ) = E_t[L_CLIP_t(θ) - c1 * L_VF_t(θ) + c2 * S[π_θ](s_t)]，其中 c1、c2 是系数，L_VF_t 是平方误差损失 (V_θ(s_t) - V_t^targ)^2。

PPO 每轮更新的数据采集和优化流程为：每次迭代，N 个并行 actor 各自在环境中运行策略 π_θ_old 收集 T 个时间步的数据，并计算优势估计 A_hat_1, ..., A_hat_T；然后基于这些 NT 个时间步的数据构建替代损失，使用 minibatch SGD（或 Adam）进行 K 个 epoch 的优化，最后更新 θ_old ← θ。

**模型引用**：ppo_2017 p.4-5; ppo_2017 p.5-5; ppo_2017 p.5-5; ppo_2017 p.5-5; ppo_2017 p.5-5; ppo_2017 p.5-5; ppo_2017 p.5-5; ppo_2017 p.5-5; ppo_2017 p.5-5; ppo_2017 p.5-5

**自动页级结果**：Citation P/R/F1 = 1.0 / 1.0 / 1.0；正确拒答 = None。

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

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：PPO 第 5 页明确给出三项联合目标、N×T 采样与 K 个 epoch；回答的符号排版略简化但内容正确。

---

## 08. N0 Dense RAG · ai_cs_v01_q026

**问题**：在 Actor-Critic 形式的 PPO 中，如果 policy 和 value function 共享网络参数，最终优化目标由哪三部分组成？PPO 每轮更新的数据采集和优化流程是什么？

**模型回答**：

根据提供的论文上下文，无法回答该问题。上下文中仅包含 PPO 论文的图注、与其他算法的对比段落，以及 ResNet、SimCLR、Transformer、MapReduce 等无关内容，未涉及 Actor-Critic 形式 PPO 中 policy 和 value function 共享网络参数时的优化目标组成，也未描述 PPO 每轮更新的数据采集和优化流程。

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
- 主要错误类型 / 备注：错误拒答；第 5 页同时包含共享参数目标与 Actor-Critic 更新算法，Dense 检索没有提供该页。

---

## 09. LC1 单论文全文 · ai_cs_v01_q035

**问题**：RAG 在四个 open-domain QA 数据集上的测试表现如何？RAG-Sequence 和 RAG-Token 分别取得了哪些 Exact Match 分数？

**模型回答**：

根据论文上下文，RAG 在四个 open-domain QA 数据集（Natural Questions、TriviaQA、WebQuestions、CuratedTrec）上的测试表现如下：RAG-Token 的 Exact Match 分数分别为 NQ 44.1、TQA 55.2（标准测试集）/66.1（TQA-Wiki 测试集）、WQ 45.5、CT 50.0；RAG-Sequence 的 Exact Match 分数分别为 NQ 44.5、TQA 56.8（标准测试集）/68.0（TQA-Wiki 测试集）、WQ 45.2、CT 52.2。

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
- 主要错误类型 / 备注：RAG 第 6 页 Table 1 的 NQ/TQA/WQ/CT Exact Match 数值与回答逐项一致。

---

## 10. N0 Dense RAG · ai_cs_v01_q035

**问题**：RAG 在四个 open-domain QA 数据集上的测试表现如何？RAG-Sequence 和 RAG-Token 分别取得了哪些 Exact Match 分数？

**模型回答**：

根据论文上下文，RAG 在四个 open-domain QA 数据集（NQ、TQA、WQ、CT）上的测试表现如下：RAG-Token 在 NQ 上 Exact Match 为 44.1，TQA 上为 55.2（标准测试集）/66.1（TQA-Wiki），WQ 上为 45.5，CT 上为 50.0；RAG-Sequence 在 NQ 上为 44.5，TQA 上为 56.8/68.0，WQ 上为 45.2，CT 上为 52.2。

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
- 主要错误类型 / 备注：RAG 四个数据集与两种模型的 Exact Match 数值均正确，Dense 上下文已覆盖关键表格。

---

## 11. LC2 跨论文全文 · CD001

**问题**：Transformer 论文中的模型架构设计，与 LoRA 论文中的参数高效适配方法分别解决什么问题？两者作用的位置和训练目标有什么差异？

**模型回答**：

Transformer 论文提出了一种完全基于注意力机制的序列转换模型架构，旨在解决循环神经网络固有的顺序计算限制，从而提升并行化能力并减少训练时间。LoRA 论文则针对大规模预训练语言模型全量微调成本过高的问题，提出一种参数高效的适配方法，通过冻结预训练权重并注入低秩分解矩阵来减少可训练参数量。

两者作用的位置不同：Transformer 架构在编码器和解码器堆栈中，通过多头自注意力子层和前馈网络子层（以及解码器中的编码器-解码器注意力子层）来构建模型；LoRA 则作用于 Transformer 的权重矩阵（如自注意力模块中的 Wq、Wv 等），通过低秩分解来更新这些权重。

训练目标也不同：Transformer 的训练目标是最大化序列转换任务（如机器翻译）的条件概率，通过标准的交叉熵损失进行端到端训练；LoRA 的训练目标是在保持预训练权重不变的情况下，优化低秩矩阵以最大化下游任务的条件概率，从而实现对特定任务的适配。

**模型引用**：transformer_2017 p.2-2; transformer_2017 p.3-3; transformer_2017 p.3-3; lora_2021 p.1-1; lora_2021 p.4-4; lora_2021 p.5-5

**自动页级结果**：Citation P/R/F1 = 0.4 / 1.0 / 0.5714285714285715；正确拒答 = None。

**人工核验基准**：

预期答案：C1：Transformer 的目标是设计一种新的序列转换模型架构，通过完全基于注意力机制替代循环和卷积结构，提高序列建模的并行能力。
C2：LoRA 的目标是在保持预训练模型权重冻结的情况下，通过训练低秩矩阵表示参数更新，从而降低大模型下游适配的训练参数量和存储成本。
C3：两者不是同一层面的替代方法，Transformer 关注基础模型架构设计，而 LoRA 关注已有 Transformer 模型的任务适配方式。
Gold 证据：
- transformer_2017 第 1-2 页（Abstract / Introduction）
- lora_2021 第 1-2 页（Abstract / Introduction）

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：正确区分基础架构与参数高效适配；Transformer 与 LoRA 的职责、作用位置和训练对象没有混淆。

---

## 12. N0 Dense RAG · CD001

**问题**：Transformer 论文中的模型架构设计，与 LoRA 论文中的参数高效适配方法分别解决什么问题？两者作用的位置和训练目标有什么差异？

**模型回答**：

Transformer 论文提出的模型架构主要解决序列转导任务中循环网络训练慢、难以并行的问题，其核心是使用完全基于注意力机制的多头自注意力替代循环层，从而提升训练速度并取得更好的翻译效果。LoRA 论文提出的参数高效适配方法则解决大规模语言模型微调成本过高的问题，即硬件需求大、存储和任务切换开销高，其通过冻结预训练权重并注入低秩矩阵来高效适配下游任务。

在作用位置上，Transformer 架构中的自注意力模块包含四个权重矩阵（Wq、Wk、Wv、Wo），LoRA 方法主要应用于这些注意力权重矩阵，而冻结 MLP 模块（见 CTX:R03）。训练目标上，Transformer 的原始训练目标是完成序列转导任务（如机器翻译），而 LoRA 的训练目标是高效适配下游任务，仅优化注入的低秩矩阵，不更新预训练权重，且推理时可将低秩矩阵合并回原权重，不引入额外延迟（见 CTX:R07）。

**模型引用**：transformer_2017 p.10-10; lora_2021 p.12-12; lora_2021 p.5-5; lora_2021 p.4-4

**自动页级结果**：Citation P/R/F1 = 0.0 / 0.0 / 0.0；正确拒答 = None。

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
- 主要错误类型 / 备注：主比较结论正确，但将 LoRA 实验中未适配 MLP 的选择表述为“冻结 MLP”，且引用页未覆盖 Gold，论证不够严谨。

---

## 13. LC2 跨论文全文 · CD003

**问题**：ResNet 与 Transformer 分别把 residual connection 放在什么结构位置，并直接说明了什么作用？能否仅据两篇论文断言二者的主要设计动机完全相同？

**模型回答**：

根据提供的论文上下文，ResNet 将残差连接（shortcut connection）放置在“每个少数堆叠层”的构建块中，具体为：y = F(x, {Wi}) + x，其中 F 是待学习的残差映射，x 是输入，残差连接通过恒等映射（identity mapping）和逐元素相加实现，且不增加额外参数或计算复杂度。其直接作用是解决深度网络的退化问题（degradation problem），使网络更容易优化，并可从显著增加的深度中获益。Transformer 将残差连接放置在编码器和解码器每个子层（子层包括多头自注意力机制和前馈网络）周围，具体为：LayerNorm(x + Sublayer(x))，即每个子层的输出先与输入相加，再进行层归一化。其直接作用是便于训练深层模型（引用[11]）。关于两篇论文的主要设计动机是否完全相同，仅凭提供的上下文无法断言。ResNet 的动机是解决深度网络的退化问题，而 Transformer 的动机是减少序列计算、提高并行化能力，并完全依赖注意力机制。两篇论文的动机不同，因此不能断言完全相同。

**模型引用**：resnet_2015 p.3-3; resnet_2015 p.3-3; resnet_2015 p.3-3; resnet_2015 p.2-2; resnet_2015 p.2-2; transformer_2017 p.3-3; transformer_2017 p.3-3

**自动页级结果**：Citation P/R/F1 = 0.6666666666666666 / 1.0 / 0.8；正确拒答 = None。

**人工核验基准**：

预期答案：C1：ResNet 使用 shortcut connection 将输入 x 直接加入网络输出，使网络学习残差函数 F(x)，用于缓解深层网络优化困难。
C2：Transformer 的 encoder 和 decoder 子层也使用 residual connection，并结合 LayerNorm。
C3：两者都采用残差连接，但 ResNet 明确将其用于缓解深层网络的优化/退化问题；Transformer 论文将其描述为子层的架构组件，未直接把其主要动机表述为与 ResNet 完全相同，因此不能据此作完全等同的断言。
Gold 证据：
- resnet_2015 第 2-2 页（Residual Learning Framework）
- transformer_2017 第 3-3 页（Encoder and Decoder Stacks）

**请填写**：

- 答案语义：`[ ] 正确  [x] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[x] 否  [ ] 是`
- 主要错误类型 / 备注：两种残差连接的位置和“不能完全等同动机”的结论正确；但“Transformer 的直接作用是便于训练深层模型”不是该页直接陈述，应收敛为架构描述。

---

## 14. N0 Dense RAG · CD003

**问题**：ResNet 与 Transformer 分别把 residual connection 放在什么结构位置，并直接说明了什么作用？能否仅据两篇论文断言二者的主要设计动机完全相同？

**模型回答**：

根据提供的论文上下文，ResNet 将残差连接（shortcut connection）置于堆叠的非线性层之间，具体为“跳过一层或多层”，其输出与堆叠层的输出相加（F(x)+x），并明确指出这种恒等快捷连接的作用是“不增加额外参数或计算复杂度”，且使网络更易优化（将残差映射推向零比拟合恒等映射更容易）。Transformer 论文的上下文中未直接描述残差连接的具体结构位置或作用，仅提及注意力机制和整体架构。因此，无法仅依据这两篇论文断言二者的主要设计动机完全相同。

**模型引用**：resnet_2015 p.2-2

**自动页级结果**：Citation P/R/F1 = 1.0 / 0.5 / 0.6666666666666666；正确拒答 = None。

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
- 主要错误类型 / 备注：错误遗漏 Transformer 第 3 页已明确给出的 residual connection 与 LayerNorm，未完成跨论文比较。

---

## 15. LC2 跨论文全文 · CD005

**问题**：PPO 和 LoRA 分别优化什么对象？二者如何通过限制或减少更新规模提高训练效率？

**模型回答**：

PPO 优化的是强化学习中的策略（policy）参数 θ，通过最大化一个裁剪的替代目标函数（clipped surrogate objective）来更新策略，同时限制每次更新的幅度，避免过大的策略更新。LoRA 优化的是预训练模型权重矩阵的低秩分解矩阵 A 和 B，通过冻结原始权重 W0，仅训练注入的低秩矩阵，从而减少可训练参数的数量。二者都通过限制或减少更新规模来提高训练效率：PPO 通过裁剪概率比率（clipping probability ratio）限制策略更新的幅度，防止破坏性的大更新；LoRA 通过低秩分解减少可训练参数，从而降低计算和存储开销。

**模型引用**：ppo_2017 p.3-3; lora_2021 p.4-4; lora_2021 p.2-2

**自动页级结果**：Citation P/R/F1 = 0.6666666666666666 / 1.0 / 0.8；正确拒答 = None。

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
- 主要错误类型 / 备注：PPO 第 3 页的 clipped objective 与 LoRA 第 2 页的低秩参数更新均被正确解释，比较维度合理。

---

## 16. N0 Dense RAG · CD005

**问题**：PPO 和 LoRA 分别优化什么对象？二者如何通过限制或减少更新规模提高训练效率？

**模型回答**：

根据提供的论文上下文，PPO 和 LoRA 分别优化不同的对象：PPO 优化策略（policy），通过裁剪的替代目标函数限制策略更新的幅度，从而提高训练稳定性；LoRA 优化注入的低秩矩阵 A 和 B，通过冻结预训练模型并仅更新这些小型矩阵，减少需要计算梯度和维护优化器状态的参数数量，从而降低存储和计算开销。二者均通过限制或减少更新规模来提高训练效率：PPO 通过裁剪目标函数限制策略更新步长，LoRA 通过低秩分解减少可训练参数。

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
- 主要错误类型 / 备注：内容正确；虽然 PPO 引用页不在 Gold 第 3 页，但其“裁剪限制策略更新、LoRA 减少可训练参数”的结论没有错误。

---

## 17. LC2 跨论文全文 · CD007

**问题**：ResNet 和 PPO 是否在同一个公开数据集、同一个评价指标下进行了直接实验比较？哪一个方法取得更高性能？

**模型回答**：

根据提供的论文上下文，ResNet 论文（resnet_2015）和 PPO 论文（ppo_2017）没有在同一个公开数据集、同一个评价指标下进行直接实验比较。ResNet 论文在 ImageNet、CIFAR-10、PASCAL VOC、MS COCO 等数据集上评估图像识别、目标检测等任务，使用 top-1/top-5 错误率、mAP 等指标；PPO 论文在 MuJoCo 连续控制任务和 Atari 游戏上评估强化学习性能，使用平均奖励等指标。两者任务类型、数据集和指标均不同，因此无法判断哪个方法性能更高。

**模型引用**：resnet_2015 p.1-1; resnet_2015 p.4-4; resnet_2015 p.7-7; resnet_2015 p.8-8; ppo_2017 p.1-1; ppo_2017 p.6-6; ppo_2017 p.8-8

**自动页级结果**：Citation P/R/F1 = None / None / None；正确拒答 = False。

**人工核验基准**：

预期答案：论文没有提供 ResNet 和 PPO 在同一任务、同一数据集、同一指标下的直接实验比较，因此不能判断哪个方法性能更高。
Gold 证据：
- 无；此题要求拒答，不能强行比较。

**请填写**：

- 答案语义：`[ ] 正确  [x] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[ ] 否  [x] 是`
- 主要错误类型 / 备注：文字结论正确地拒绝了跨任务性能排序，但结构化输出 `refused=false` 且附带引用；按本题协议应显式拒答。

---

## 18. N0 Dense RAG · CD007

**问题**：ResNet 和 PPO 是否在同一个公开数据集、同一个评价指标下进行了直接实验比较？哪一个方法取得更高性能？

**模型回答**：

根据提供的论文上下文，无法回答该问题。上下文中仅涉及 SimCLR 论文中 ResNet 架构在 ImageNet 等数据集上的自监督与监督学习对比，以及 RAG 论文中关于开放域问答的评估设置，均未提及 PPO 方法，也没有 ResNet 与 PPO 在相同数据集和指标下的直接比较。

**模型引用**：无

**自动页级结果**：Citation P/R/F1 = None / None / None；正确拒答 = True。

**人工核验基准**：

预期答案：论文没有提供 ResNet 和 PPO 在同一任务、同一数据集、同一指标下的直接实验比较，因此不能判断哪个方法性能更高。
Gold 证据：
- 无；此题要求拒答，不能强行比较。

**请填写**：

- 答案语义：`[x] 正确  [ ] 部分正确  [ ] 不正确`
- 是否混淆论文来源：`[x] 否  [ ] 是`
- 是否应拒答：`[ ] 否  [x] 是`
- 主要错误类型 / 备注：正确以拒答结束，未输出性能排序；其召回上下文不充分，但没有凭空给出比较结论。

---

## 19. LC2 跨论文全文 · CD009

**问题**：RAG 和 MapReduce 是否都使用“外部数据”？两篇论文中外部数据分别承担什么作用？

**模型回答**：

根据提供的论文上下文，RAG（检索增强生成）和 MapReduce 是两种不同的技术，它们对“外部数据”的使用方式和作用不同。

RAG 论文中明确提到，其模型结合了“参数化记忆”（parametric memory）和“非参数化记忆”（non-parametric memory）。其中，非参数化记忆是一个基于 Wikipedia 的密集向量索引，通过预训练的神经检索器访问，作为外部知识源。RAG 使用外部数据（Wikipedia 文档）来增强生成过程，使模型能够访问和利用外部知识，从而在知识密集型任务中生成更准确、更具体的回答。

MapReduce 论文中，其编程模型处理的是“输入数据”（input data），这些数据通常是大型数据集（如爬取的文档、网络请求日志等），但论文并未将其称为“外部数据”。MapReduce 的作用是并行化处理这些大规模数据，通过 Map 和 Reduce 函数对数据进行转换和聚合，以生成派生数据（如倒排索引、网页图结构等）。

因此，RAG 和 MapReduce 都处理外部数据，但作用不同：RAG 利用外部数据作为知识来源来增强生成，而 MapReduce 利用外部数据作为计算对象进行分布式处理。

**模型引用**：rag_2020 p.1-1; rag_2020 p.2-2; rag_2020 p.3-3; mapreduce_2004 p.1-1

**自动页级结果**：Citation P/R/F1 = 0.75 / 1.0 / 0.8571428571428571；正确拒答 = None。

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
- 主要错误类型 / 备注：正确区分 MapReduce 输入数据的计算对象角色和 RAG Wikipedia 索引的非参数知识来源角色。

---

## 20. N0 Dense RAG · CD009

**问题**：RAG 和 MapReduce 是否都使用“外部数据”？两篇论文中外部数据分别承担什么作用？

**模型回答**：

根据提供的论文上下文，RAG 和 MapReduce 都涉及外部数据，但作用不同。RAG 论文（rag_2020）明确提到其模型结合了参数化记忆（预训练 seq2seq 模型）和非参数化记忆（维基百科的密集向量索引），外部数据（维基百科）作为非参数化记忆，通过神经检索器访问，用于增强生成模型的知识访问和事实性。MapReduce 论文（mapreduce_2004）中，外部数据指的是输入文件（如爬取的文档、Web 请求日志等），这些数据被分割成多个片段，由 map 函数处理生成中间键值对，再经过 reduce 函数合并，外部数据是计算处理的原始输入。因此，两者都使用外部数据，但 RAG 中外部数据作为知识来源，MapReduce 中外部数据作为待处理的数据集。

**模型引用**：rag_2020 p.1-1; mapreduce_2004 p.1-1

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
- 主要错误类型 / 备注：Dense 召回的两篇论文页码已足以支持比较，回答没有混淆“知识来源”与“待处理输入”。

---

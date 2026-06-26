# VLM 轻量化

> **问题：** 面向低轨卫星及低轨算力星座部署视觉—语言模型（VLM）时，如何围绕算力、显存、功耗、热控与存储约束，对模型进行有效轻量化？剪枝、知识蒸馏、量化和轻量化架构设计分别压缩什么对象、适用于哪些星上任务，以及应如何组合？

对于视觉—语言模型，星上部署负担来自模型权重、运行时激活、视觉 token、KV Cache、跨模态连接器以及底层算子执行。仅降低参数量并不必然降低端到端时延、峰值显存或持续运行功耗；尤其在高分辨率遥感图像、多图输入和长任务指令场景下，视觉编码阶段的预填充计算与 KV Cache 往往会成为新的瓶颈。

因此，剪枝、知识蒸馏、量化和轻量化架构设计并非可互相替代的四个选项，而是对应不同压缩对象的互补路线：量化将模型以低比特形式表示，是最直接的部署手段；剪枝删除冗余权重或结构；知识蒸馏将地面大模型的能力迁移到更小的学生模型；轻量化架构设计则从源头控制参数规模、注意力计算、KV Cache 与视觉 token 开销。

| 技术      | 直接作用对象                       | 主要收益             | 对星上 VLM 的核心价值            | 不能单独解决的问题             |
| ------- | ---------------------------- | ---------------- | ------------------------ | --------------------- |
| 剪枝      | 权重、注意力头、FFN 通道、层、视觉 token    | 降低计算图规模与模型存储     | 使压缩后的模型在既定硬件上获得真实时延和功耗收益 | KV Cache 的线性增长、量化数值误差 |
| 知识蒸馏    | 学生模型的输出、隐藏表示、跨模态关系           | 将大模型能力迁移至小模型     | 构建面向遥感/星务任务的原生小型 VLM     | 已部署模型的即时低成本压缩         |
| 量化      | 权重、激活、KV Cache、量化尺度          | 降低存储、访存与部分计算开销   | 最直接的低比特部署手段              | 模型结构冗余、视觉 token 过多    |
| 轻量化架构设计 | 模型深宽比、注意力、视觉编码器、连接器、token 路径 | 从源头降低参数、缓存和预填充成本 | 决定星上 VLM 的资源下限与长期可维护性    | 已有大模型的快速无训练压缩         |

---

## 1. 星上 VLM 轻量化的对象与边界

### 1.1 星上约束并不等价于参数约束

在地面数据中心中，模型轻量化通常首先服务于 GPU 成本与吞吐提升；在星上，轻量化还必须同时满足固定硬件资源、能量预算、热控上限、有限存储和长时间无人值守运行等约束。对于单次 VLM 推理，其资源占用可概括为：

$$
M_{\mathrm{total}} = M_{\mathrm{weight}} + M_{\mathrm{activation}} + M_{\mathrm{KV}} + M_{\mathrm{vision}} + M_{\mathrm{runtime}},
$$

其中，$M_{\mathrm{weight}}$ 为模型权重，$M_{\mathrm{activation}}$ 为预填充和解码阶段的中间激活，$M_{\mathrm{KV}}$ 为随上下文长度增长的 KV Cache，$M_{\mathrm{vision}}$ 为视觉编码器输出及视觉 token 相关开销，$M_{\mathrm{runtime}}$ 为算子 workspace、通信缓存和系统运行时开销。

这一分解意味着：将 7B 语言主干从 FP16 量化为 W4 可以显著降低 $M_{\mathrm{weight}}$，但不能自动消除高分辨率影像带来的视觉 token 膨胀，也不能解决长指令和多轮交互造成的 $M_{\mathrm{KV}}$ 增长。因此，星上 VLM 的轻量化必须对模型权重、计算结构与输入表征进行联合设计。

### 1.2 VLM 相较纯文本 LLM 的额外负担

对于遥感 VLM，图像并非普通低分辨率自然图像。大幅遥感影像通常具有宽覆盖范围、小目标密集、多尺度区域和多谱段输入等特点。若将整幅图像直接切分为大量 patch 并送入语言模型，视觉 token 数会显著抬高预填充计算、显存峰值和后续 KV Cache 占用。

因此，面向星上 VLM 的轻量化不能只压缩语言主干，还需要同步考虑：视觉编码器是否具备分层或区域选择能力；跨模态连接器是否足够紧凑；是否可在视觉端完成云筛选、候选区域提取或变化区域定位；以及是否能将任务相关区域压缩为少量语义 token 再送入语言主干。

### 1.3 相关工作

近两年的 VLM 轻量化研究已经从“把 LLM 压小后接视觉编码器”转向更细的多模态系统优化。相关论文大体围绕六类问题展开：一是视觉 token 数量过多，导致预填充计算和显存峰值过高；二是高分辨率图像进入视觉编码器时成本急剧上升；三是跨模态 projector 或 resampler 会决定视觉信息压缩后的保真度；四是小型语言主干需要更强的数据、训练配方和蒸馏机制支撑；五是 VLM 量化必须考虑视觉 token、文本 token 和跨模态激活的差异；六是遥感场景还要求多尺度、多图、多粒度理解。

| 研究方向           | 代表论文                                                                        | 对星上 VLM 的启示                                          |
| -------------- | --------------------------------------------------------------------------- | ---------------------------------------------------- |
| 蒸馏与模态自适应剪枝     | EfficientVLM [18]                                                           | 压缩视觉、文本和跨模态融合层时，应按任务动态分配视觉/语言保留比例                    |
| 视觉 token 裁剪与合并 | LLaVA-PruMerge [19]、VisPruner [20]、VScan [21]、FlowCut [22]                  | 高分辨率遥感图像的核心瓶颈常在视觉 token，而不是语言主干参数                    |
| 高分辨率编码与连接器压缩   | LLaVA-UHD [23]、TokenPacker [24]、FastVLM [25]                                | 应同时压缩视觉编码延迟、视觉 token 数和 projector 信息损失               |
| 小型/端侧 VLM 架构   | MobileVLM [15]、MobileVLM V2 [26]、LLaVA-Phi [16]、TinyLLaVA [17]、SmolVLM [27] | 1B–4B 级模型不是简单缩小，而需要小语言骨干、轻量视觉编码器、训练数据和配方协同设计         |
| VLM 专用量化       | Q-VLM [28]、LUQ [29]、QIG [30]                                                | 多模态 token 与文本 token 的统计分布不同，量化校准需要保留模态差异和 token 级敏感性 |
| 多模态蒸馏          | EM-KD [31]、Align-TI [32]                                                    | 小型 VLM 需要蒸馏视觉—文本亲和关系、显著视觉区域和生成过程，而不能只对齐答案文本          |
| 遥感 VLM 紧凑架构    | RSUniVLM [33]                                                               | 星上模型应面向图像级、区域级、像素级和多图变化任务构建统一但受控的模型规模                |

---

## 2. 技术一：剪枝

### 2.1 内涵

剪枝通过删除对任务输出贡献较小的参数或结构，使模型以更小的计算图完成近似功能。按照删除对象不同，剪枝可分为非结构化剪枝、半结构化剪枝和结构化剪枝。

非结构化剪枝直接将单个权重置零，参数稀疏率可以较高，但只有在硬件与推理框架具备高效稀疏矩阵支持时，才可能转化为实际时延和能耗收益。半结构化剪枝将非零模式限制在固定块结构内，例如 2:4 稀疏，以换取更好的硬件可执行性。结构化剪枝则删除完整注意力头、FFN 通道、层、隐藏维度或视觉编码器通道；由于压缩后的矩阵仍保持稠密结构，通常更容易在 GPU、NPU 和 FPGA 上获得可测量的端到端加速。

从技术过程看，剪枝并不是简单删除小权重，而是包含“重要性评估—结构删减—能力恢复—硬件映射”四个环节。重要性评估可依据权重幅值、输入激活、梯度、Hessian 近似或任务损失敏感性；结构删减决定删除单个权重、固定稀疏模式、通道、注意力头、层或隐藏维度；能力恢复通常依赖少量校准数据、继续训练、LoRA/Adapter 修复或蒸馏；硬件映射则决定压缩后的模型是否真正减少矩阵规模、访存和端到端时延。

| 剪枝对象 | 具体机制 | 对星上 VLM 的技术含义 |
|---|---|---|
| 权重参数 | 删除低重要性权重或固定稀疏块 | 主要降低模型存储，是否加速取决于稀疏算子支持 |
| 注意力头与 FFN 通道 | 删除完整头、通道或中间维度 | 直接缩小稠密矩阵计算，更容易转化为时延和功耗收益 |
| Transformer 层与隐藏维度 | 减少层数或模型宽度 | 改变模型主体计算图，需要配合恢复训练保证能力 |
| 视觉编码器与视觉 token | 裁剪视觉层、通道、patch 或区域 token | 直接缓解高分辨率遥感图像带来的预填充峰值 |
| 跨模态 projector | 删除冗余投影维度或连接器通道 | 避免语言主干压缩后连接器成为新的资源瓶颈 |

### 2.2 代表性工作

- SparseGPT 提出针对大规模生成式 Transformer 的一次性剪枝方法，可在无需再训练的条件下对 OPT-175B、BLOOM-176B 等模型实现至少 50% 稀疏，并兼容半结构化稀疏和权重量化 [1]。其价值在于说明大模型存在可利用的参数冗余，但其非结构化稀疏能否转换为星上实际加速仍取决于硬件支持。
- Wanda 使用“权重幅值与输入激活”的组合作为重要性指标，不需要权重更新即可完成 LLaMA 系列的剪枝 [2]。该思路提示星上部署前的剪枝校准不应只依据权重大小，还应考虑任务输入分布。
- LLM-Pruner 面向 LLaMA、Vicuna 和 ChatGLM 等模型开展结构化剪枝，并以 LoRA 对压缩后模型进行恢复 [3]。其“删减耦合结构—小规模恢复”的范式更接近星上模型的部署需求。
- SliceGPT 通过删除矩阵行、列来缩小隐藏维度，得到更小的稠密矩阵，而非依赖稀疏存储格式；在 LLaMA2-70B、OPT-66B 等模型上展示了无需额外推理优化的实际计算缩减 [4]。这类维度级压缩对星上硬件更有现实意义。
- EfficientVLM 提出“先蒸馏、再模态自适应剪枝”的 VLM 压缩框架，将视觉层、文本层和跨模态融合层按任务重要性裁剪，得到 93M 参数模型，约为教师模型的 44.3%，同时保留 98.4% 教师性能并获得 2.2 倍推理加速 [18]。这说明 VLM 剪枝不应只针对语言主干，还应纳入视觉编码器和跨模态融合结构。
- LLaVA-PruMerge 面向 LLaVA 类大多模态模型提出自适应视觉 token 缩减，先利用视觉编码器中 class token 与图像 token 的注意力稀疏性选择关键 token，再按 key 相似度合并被裁剪 token 的信息；在 LLaVA-1.5 上平均可将视觉 token 压缩 14 倍，并保持多类视觉问答和推理性能 [19]。这类方法对遥感大图的预填充峰值有直接参考价值。
- VisPruner 指出语言模型内部 text-visual attention 并不总是可靠的视觉 token 重要性指标，转而使用视觉编码器的视觉注意力选择重要 token，并以相似度去除重复 token；该方法无需训练，在 LLaVA-1.5-7B 上报告 FLOPs 降低 91%、推理延迟降低 75% 且性能接近原模型 [20]。其启示是：视觉 token 裁剪最好优先利用视觉端自身的空间显著性。
- VScan 将视觉 token 缩减拆成两阶段：视觉编码阶段做全局/局部扫描与 token merging，语言模型中间层再做视觉 token pruning；在 LLaVA-NeXT-7B 上报告预填充加速 2.91 倍、FLOPs 降低 10 倍并保留 95.4% 原性能 [21]。这说明视觉 token 并非只能在 projector 前一次性裁剪，也可以沿视觉编码和语言推理过程分阶段压缩。
- FlowCut 从跨层信息流角度分析视觉 token 冗余，认为冗余会随层间注意力集中动态形成，单层注意力分数可能产生矛盾判断；其信息流感知剪枝在 LLaVA-1.5-7B 和 LLaVA-NeXT-7B 上分别报告 88.9% 与 94.4% token reduction，并带来 3.2 倍预填充加速 [22]。这对星上任务的意义是：高压缩率视觉 token 裁剪需要显式保护信息流，而不是只按单层分数排序。

### 2.3 适用边界

剪枝不能替代量化。即使删除了部分注意力头或 FFN 通道，剩余模型仍可能以 FP16/BF16 形式占据较大存储。剪枝也不能单独解决长上下文的 KV Cache 问题；若任务依赖多轮对话、长星务手册或多时相影像输入，还需要结合 GQA/MQA、滑动窗口注意力、KV Cache 量化或外部检索机制。

---

## 3. 技术二：知识蒸馏

### 3.1 内涵

知识蒸馏以能力更强的教师模型指导学生模型学习，使学生不仅拟合标注答案，还可以学习教师的输出分布、隐藏表征、注意力关系、推理轨迹或高质量合成数据。与剪枝和量化不同，蒸馏的目标不是将既有模型直接“变小”，而是在地面训练阶段构建一个原生更小、更加适配星上任务的学生模型。

对于生成式 LLM，蒸馏需要处理序列生成中的曝光偏差与长文本累积误差；对于 VLM，还需额外维持视觉特征、文本语义和跨模态对齐关系。因此，面向星上 VLM 的蒸馏应至少覆盖输出层蒸馏、跨模态表征蒸馏和任务级行为蒸馏三个层次。

更具体地说，蒸馏压缩的不是某一个存储对象，而是将教师模型的“能力分布”转移到更小结构中。输出蒸馏关注答案概率和生成文本；中间层蒸馏关注隐藏状态、注意力图和 FFN 表征；关系蒸馏关注 token-token、region-token 和 vision-text 的相似性结构；数据蒸馏利用教师生成遥感问答、灾害报告、星务诊断样本和拒答样本；推理蒸馏则将教师的步骤化判断、工具调用轨迹或结构化报告格式传递给学生模型。

| 蒸馏层次 | 学习对象 | 对星上任务的作用 |
|---|---|---|
| 输出蒸馏 | logits、答案分布、生成文本 | 保留问答、分类、报告生成和指令跟随能力 |
| 表征蒸馏 | hidden states、attention、视觉特征 | 保留语言理解、视觉编码和跨模态对齐能力 |
| 关系蒸馏 | 视觉区域与文本 token 的对应关系 | 降低小模型在目标定位、属性描述和 grounding 上的退化 |
| 数据蒸馏 | 教师生成的图文指令、任务报告、困难样本 | 为遥感、态势感知和星务任务构建低成本领域训练集 |
| 行为蒸馏 | 推理步骤、拒答边界、结构化输出格式 | 约束学生模型在高风险任务中的输出方式 |

对遥感 VLM 而言，只蒸馏最终文字答案通常不够。学生模型还需要学习教师如何选择图像区域、如何把小目标或变化区域与文本描述对齐、如何在云雾遮挡或低信噪比场景中表达不确定性，以及如何把检测、变化分析和自然语言解释组织成稳定的结构化告警。

### 3.2 代表性工作

- MiniLLM 针对生成式语言模型提出基于反向 KL 散度的蒸馏目标，以缓解传统前向 KL 蒸馏对低概率区域的过度覆盖问题，并在 120M–13B 学生模型上验证生成质量、校准性和长文本表现 [5]。该工作表明，生成式大模型蒸馏不能简单照搬分类模型的输出对齐方式。
- GKD 提供了面向大规模预训练语言模型的通用蒸馏框架，支持在有限显存条件下组织多种蒸馏策略 [6]。其系统层面的启示是：蒸馏计算负担应主要留在地面，星上只承担学生模型部署与有限适配。
- Minitron（*Compact Language Models via Pruning and Knowledge Distillation*）将深度、宽度、注意力和 MLP 剪枝与基于蒸馏的再训练结合，从 15B 模型派生出 8B 与 4B 模型 [7]。该工作说明“结构化裁剪 + 蒸馏恢复”比单独使用剪枝更适合构建一组不同资源等级的部署模型。
- EfficientVLM 在视觉—语言预训练阶段对缩小后的 VLM 进行知识蒸馏，再结合模态自适应剪枝，说明 VLM 蒸馏可以先获得任务无关的紧凑多模态表示，再根据下游任务裁剪视觉、文本和融合模块 [18]。这一范式比只在最终指令微调阶段模仿答案更适合构建星上基础学生模型。
- EM-KD 面向高效 MLLM 中“学生视觉 token 少、教师视觉 token 多”的不平衡问题，先用 Hungarian matching 对齐视觉 token，再蒸馏 vision-language affinity 和视觉语义分布 [31]。该工作提示：如果星上模型采用视觉 token 压缩，蒸馏目标必须先解决 token 数量与空间位置不一致，否则学生容易损失细粒度视觉理解能力。
- Align-TI 认为多模态蒸馏不能停留在 next-token 对齐，而应蒸馏 token interactions：一方面对齐视觉指令相关的显著区域，另一方面对齐响应 token 之间的动态转移概率；其 2B 蒸馏模型在实验中超过 LLaVA-1.5-7B [32]。这说明星上小模型的蒸馏重点应从“答案像不像”扩展为“看哪里、如何组织推理链和输出结构”。

### 3.3 适用边界

蒸馏不能替代领域数据与任务定义。若教师模型本身不具备遥感、态势感知或星务语境下的可靠能力，学生模型只会继承其误判、幻觉和偏差。对于高风险任务，蒸馏训练集应纳入负样本、拒答样本、边界案例和结构化评测样本，并以目标检测、变化检测等专用感知模型的输出作为可验证约束，而不应仅依赖自然语言生成质量。

---

## 4. 技术三：量化

### 4.1 内涵

量化将 FP16、BF16 或 FP32 的权重、激活和缓存映射为更低比特的整数或浮点表示，以降低模型存储、内存带宽和部分矩阵计算开销。对于参数量为 $N$ 的模型，若仅考虑权重存储，理论占用近似为：

$$
M_{\mathrm{weight}} \approx N \times \frac{b}{8},
$$

其中 $b$ 为每个权重的比特数。由此可见，权重从 16 bit 降至 4 bit，理论存储量可降至原来的四分之一；但实际部署中还需计入量化尺度、视觉编码器、激活、KV Cache 和推理 workspace。

从量化对象看，可区分为权重量化、权重—激活联合量化、混合精度量化和 KV Cache 量化。权重量化最易于部署；权重—激活联合量化更可能带来端到端加速，但对硬件算子支持和激活 outlier 处理要求更高；KV Cache 量化则直接针对长上下文推理中的显存瓶颈。

量化的技术核心包括量化粒度、量化尺度、离群值处理和硬件算子匹配。量化粒度决定按张量、通道、分组还是块来共享 scale；量化尺度决定浮点值如何映射到整数或低比特浮点；离群值处理用于保护少量对输出影响显著的通道或权重；硬件算子匹配决定 INT8、INT4、FP8 等表示能否真正转化为吞吐、功耗和热控收益。

| 量化类型 | 压缩对象 | 技术重点 | 对星上 VLM 的含义 |
|---|---|---|---|
| Weight-only PTQ | 语言主干、视觉编码器、projector 权重 | 无训练或少量校准，降低权重存储和访存 | 最适合作为已有模型的快速部署路径 |
| 权重—激活联合量化 | 权重与运行时激活 | 处理激活 outlier，匹配 INT8/FP8 算子 | 可能带来真实计算加速，但需要遥感数据校准 |
| 混合精度量化 | 敏感层高精度，其余模块低精度 | 保护视觉前端、连接器、输出头等关键模块 | 比全模型统一 W4 更适合多模态模型 |
| KV Cache 量化 | 解码阶段 Key/Value 缓存 | 控制长上下文和多轮交互显存增长 | 对星务手册问答、多图任务和 RAG 场景很关键 |
| 量化感知训练 | 训练阶段模拟低比特误差 | 提高极低比特部署稳定性 | 更适合地面完成训练后上星推理 |
| 量化 PEFT | 量化底座与 LoRA/Adapter | 冻结低比特底座，只更新小模块 | 便于后续任务适配和有限增量更新 |

在星上 VLM 中，量化还需要按模块区分敏感性。视觉编码器早期层负责保留纹理、边缘和小目标信息，跨模态 projector 负责视觉语义进入语言主干的接口，语言模型后部层影响生成稳定性和格式遵循；这些模块对低比特误差的容忍度并不相同。因此，量化策略应从“模型整体压到几 bit”转向“哪些模块、哪些通道、哪些缓存对象可以低比特表示”。

### 4.2 代表性工作

- GPTQ 使用近似二阶信息进行一次性权重量化，可将大规模 GPT 类模型量化至 3–4 bit，并在生成式推理中保持较小精度损失 [8]。这类后训练量化适合已训练模型的快速部署评估。
- SmoothQuant 通过等价变换将激活中的量化难度迁移至权重侧，实现 LLM 的 W8A8 量化，并在多类模型上报告内存降低与推理加速 [9]。其关键启示是：激活 outlier 是联合量化的主要障碍，不能仅压缩权重。
- AWQ 依据激活统计识别并保护少量显著权重通道，提出硬件友好的低比特权重量化；论文同时报告其方法可用于多模态语言模型，并配套实现面向 4-bit LLM/VLM 的推理框架 [10]。这使其对星上 VLM 的参考价值较高。
- QLoRA 冻结 4-bit 量化底座，仅对 LoRA 参数进行训练，显著降低微调显存需求 [11]。其主要意义不在于单纯推理压缩，也不是 VLM 专项实测结果，而是为“量化部署底座 + 参数高效更新”提供通用技术基础。
- Q-VLM 专门针对 LVLM 后训练量化，指出常规逐层 rounding 搜索忽略跨层依赖，提出以激活熵作为代理进行块划分，并优化视觉编码器以降低搜索空间耦合；在 13B LLaVA 上报告 2.78 倍内存压缩和 1.44 倍生成速度提升，且多模态推理任务无明显性能下降 [28]。这表明 VLM 量化需要把视觉编码器和语言主干作为耦合系统处理。
- LUQ 研究 MLLM 的低于 4-bit 超低比特量化，发现多模态 token 及其产生的中间层激活比文本 token 具有更高熵，导致其对极低比特量化更敏感；该方法按层输出激活熵选择可压缩层，并强调使用图像与文本混合校准数据 [29]。这一结论直接支持星上 VLM 采用分层、分模块混合精度，而不是统一低比特。
- QIG 进一步将 LVLM 量化敏感性从模态级推进到 token 级，利用 integrated gradients 评估 token 对量化误差的敏感性，在 W4A8 和 W3A16 设置下改善多个 LVLM 的量化精度 [30]。这对遥感 VLM 尤其重要，因为小目标、边缘区域和异常区域可能只对应少量高敏感视觉 token。

### 4.3 适用边界

量化不能解决模型结构过宽、视觉 token 过多或长上下文设计不当的问题。进一步地，低比特表示会改变数值容错余量；在空间软错误环境下，量化尺度、异常通道和关键高位比特的保护方式需要与后续辐射可靠性研究协同考虑。因此，量化部署后仍需评测正常输入、低照度/云雾等退化输入、异常输入以及故障注入条件下的任务性能。

---

## 5. 技术四：轻量化架构设计

### 5.1 内涵

轻量化架构设计不以“先训练超大模型、再进行事后压缩”为前提，而是在模型设计阶段控制参数规模、注意力复杂度、KV Cache、视觉 token 和跨模态交互成本。它决定了星上模型的资源下限，也是面向长期部署与多版本维护最具基础性的路线。

对语言模型主干而言，典型方向包括深而窄的网络设计、embedding 共享、分组查询注意力（GQA）、多查询注意力（MQA）、滑动窗口注意力、状态空间模型以及稀疏专家等。对 VLM 而言，还必须关注轻量视觉编码器、紧凑跨模态连接器、视觉 token 压缩与动态计算路径。

轻量化架构设计的本质是把资源约束前移到模型定义阶段，而不是在训练完成后被动压缩。它需要同时设计语言骨干、视觉骨干、跨模态连接器、上下文机制和任务执行路径：语言骨干决定参数量与解码开销；视觉骨干决定高分辨率图像的预填充成本；连接器决定视觉 token 如何进入语言空间；注意力和记忆机制决定 KV Cache 增长；动态计算路径决定不同任务是否需要调用完整 VLM。

| 架构方向 | 具体机制 | 主要控制的资源对象 |
|---|---|---|
| 紧凑语言骨干 | 深而窄网络、共享 embedding、缩小 FFN、块级权重共享 | 参数、权重存储、解码计算 |
| 高效注意力 | GQA/MQA、滑动窗口、分块上下文、线性注意力 | KV Cache、长上下文访存、解码时延 |
| 轻量视觉编码器 | 分层编码、小型 ViT/CNN、区域选择、多尺度池化 | 视觉激活、预填充计算、显存峰值 |
| 视觉 token 压缩 | patch 合并、区域 token、候选目标 token、动态 token 路由 | 视觉 token 数、语言主干 prefill 成本 |
| 紧凑跨模态连接器 | 小型 projector、低秩连接器、门控融合 | 视觉—语言接口计算和量化敏感性 |
| 任务级动态路径 | 小模型前筛、复杂样本转交、RAG/规则约束 | 平均功耗、热负载、通信与算力调度 |

这意味着星上 VLM 的架构轻量化不能只等同于“小语言模型”。对于遥感输入，视觉 token 数和视觉预填充往往比语言主干参数更早触及资源上限；对于星务问答和态势感知，长上下文和检索记忆会带来 KV Cache 压力；对于通信调度和告警生成，结构化输出、规则约束和小模型前筛可以减少对通用大模型能力的依赖。

### 5.2 代表性工作

- MobileLLM 针对亚十亿参数语言模型采用深而窄的架构、embedding 共享、GQA 与块级权重共享，可作为星上 VLM 语言主干设计参考，说明在小模型尺度下架构本身能够显著影响能力—资源比 [12]。
- Mistral 7B 采用 GQA 加速推理，并以滑动窗口注意力降低长序列的推理成本 [13]。作为高效注意力机制参考，该类设计可缓解星上长任务文本、任务日志和检索上下文中的 KV Cache 与解码访存压力。
- Mamba 以选择性状态空间模型构建不依赖标准注意力的序列骨干，在长序列场景中具有线性扩展特征 [14]。其更适合作为遥测时间序列、任务日志和长时序传感器信息处理的候选骨干；是否能替代遥感 VLM 中的视觉—语言推理主干仍需专门验证。
- MobileVLM 使用 1.4B/2.7B 级语言模型、轻量视觉模型和高效 projector 构建面向移动设备的 VLM，并在 NVIDIA Jetson AGX Orin 上完成推理测试 [15]。该工作说明“小语言模型 + 紧凑视觉编码器 + 高效连接器”是边缘多模态部署的可行组合，但其自然图像设定不能直接代替遥感任务验证。
- LLaVA-Phi 与 TinyLLaVA 分别展示了以 2.7B/3.1B 级小语言模型构建多模态助手的可能性，说明高质量数据、视觉编码器选择和连接器设计可部分弥补语言主干缩小造成的能力损失 [16,17]。
- MobileVLM V2 在 MobileVLM 基础上结合新的架构设计、面向移动 VLM 的训练方案和高质量数据整理，报告 1.7B 模型达到或超过部分 3B 级 VLM，3B 模型超过多种 7B+ VLM [26]。其价值在于说明端侧 VLM 的能力并不只由参数规模决定，数据配方与连接器设计同样关键。
- SmolVLM 系统探索小型多模态模型的架构配置、图像 tokenization 策略和数据整理，最小 256M 模型推理显存低于 1GB，2.2B 模型达到接近更大模型的性能并支持图像与视频任务 [27]。该工作为星上 1B 以下或 1B–3B 级轻量 VLM 提供了更明确的架构参照。
- LLaVA-UHD 面向任意宽高比和高分辨率图像，提出图像模块化切片、视觉 token 压缩模块和空间组织 schema；在 LLaVA-1.5 336x336 基础上支持 6 倍更大分辨率且推理计算约为 94%，并提升 TextVQA 表现 [23]。该思路对遥感大幅影像尤其相关，因为它把高分辨率输入处理成可控的视觉 token 组织问题。
- TokenPacker 将 visual projector 从简单 MLP 改为粗到细的视觉 token 压缩器，用低分辨率点查询承载整体表示，再通过区域到点注入吸收高分辨率多层区域线索；实验报告视觉 token 压缩 75%–89% 且保持或提升多项性能 [24]。这说明 projector 可以承担“压缩但保细节”的关键角色，而不只是模态维度对齐层。
- FastVLM 通过 FastViTHD 高效视觉编码器，在高分辨率下同时降低视觉编码延迟和送入 LLM 的视觉 token 数；在 LLaVA-1.5 设置中报告 time-to-first-token 提升 3.2 倍，相比高分辨率 LLaVA-OneVision 在同 0.5B LLM 下实现 85 倍 TTFT 提升且视觉编码器小 3.4 倍 [25]。这对星上交互式任务很关键，因为首 token 延迟直接影响告警和调度响应。
- RSUniVLM 面向遥感 VLM 提出 Granularity-oriented Mixture of Experts，使模型在约 1B 参数规模下覆盖图像级、区域级、像素级任务，并支持多图变化检测与变化描述 [33]。这提示星上遥感 VLM 应通过粒度路由和任务专家控制模型规模，而不是把所有任务都交给同一条通用大模型路径。

### 5.3 适用边界

轻量化架构设计的代价是需要重新训练、蒸馏或至少进行大规模领域适配，因此不适合用来解决已部署模型的即时资源不足。其更适合在地面完成模型族设计：以同一任务接口维护小、中、大多个模型版本，再依据不同卫星的算力板卡、功耗预算和任务等级选择部署版本。

---

## 6. 结论

近期 VLM 专项论文进一步说明，星上 VLM 轻量化的关键变量正在从“语言主干有多小”转向“视觉 token 如何生成、压缩、排序、蒸馏和量化”。视觉 token 裁剪与合并可直接降低预填充峰值；高效视觉编码器与 projector 决定高分辨率遥感图像能否进入模型；VLM 专用量化需要考虑多模态 token 的高熵和 token 级敏感性；多模态蒸馏则需要对齐视觉区域、文本指令和响应生成过程。

对星上 VLM 而言，最值得优先推进的不是单独比较四种技术的压缩率，而是建立“任务—模型组件—资源对象—硬件平台”的联合优化关系：闭集判别任务优先使用专用小模型；需要开放语义理解和自然语言重配置的任务采用紧凑 VLM；已有模型部署以量化和结构化剪枝为主；新模型研制以轻量架构和地面蒸馏为主；高分辨率遥感场景则必须将视觉 token 压缩与 KV Cache 管理纳入核心设计，而不能只关注语言模型参数量。

---

## 参考文献

[1] FRANTAR E, ALISTARH D. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot[EB/OL]. arXiv:2301.00774, 2023[2026-06-25]. https://arxiv.org/abs/2301.00774.

[2] SUN M, LIU Z, BAIR A, et al. A Simple and Effective Pruning Approach for Large Language Models[EB/OL]. arXiv:2306.11695, 2023[2026-06-25]. https://arxiv.org/abs/2306.11695.

[3] MA X, FANG G, WANG X. LLM-Pruner: On the Structural Pruning of Large Language Models[EB/OL]. arXiv:2305.11627, 2023[2026-06-25]. https://arxiv.org/abs/2305.11627.

[4] ASHKBOOS S, CROCI M L, GENNARI DO NASCIMENTO M, et al. SliceGPT: Compress Large Language Models by Deleting Rows and Columns[EB/OL]. arXiv:2401.15024, 2024[2026-06-25]. https://arxiv.org/abs/2401.15024.

[5] GU Y, DONG L, WEI F, et al. MiniLLM: On-Policy Distillation of Large Language Models[EB/OL]. arXiv:2306.08543, 2023[2026-06-25]. https://arxiv.org/abs/2306.08543.

[6] TAN S, TAM W L, WANG Y, et al. GKD: A General Knowledge Distillation Framework for Large-Scale Pre-Trained Language Model[EB/OL]. arXiv:2306.06629, 2023[2026-06-25]. https://arxiv.org/abs/2306.06629.

[7] MURALIDHARAN S, SREENIVAS S T, JOSHI R, et al. Compact Language Models via Pruning and Knowledge Distillation[EB/OL]. arXiv:2407.14679, 2024[2026-06-25]. https://arxiv.org/abs/2407.14679.

[8] FRANTAR E, ASHKBOOS S, HOEFLER T, et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers[EB/OL]. arXiv:2210.17323, 2022[2026-06-25]. https://arxiv.org/abs/2210.17323.

[9] XIAO G, LIN J, SEZNEC M, et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models[EB/OL]. arXiv:2211.10438, 2022[2026-06-25]. https://arxiv.org/abs/2211.10438.

[10] LIN J, TANG J, TANG H, et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration[EB/OL]. arXiv:2306.00978, 2023[2026-06-25]. https://arxiv.org/abs/2306.00978.

[11] DETTMERS T, PAGNONI A, HOLTZMAN A, et al. QLoRA: Efficient Finetuning of Quantized LLMs[EB/OL]. arXiv:2305.14314, 2023[2026-06-25]. https://arxiv.org/abs/2305.14314.

[12] LIU Z, ZHAO C, IANDOLA F, et al. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases[EB/OL]. arXiv:2402.14905, 2024[2026-06-25]. https://arxiv.org/abs/2402.14905.

[13] JIANG A Q, SABLAYROLLES A, MENSCH A, et al. Mistral 7B[EB/OL]. arXiv:2310.06825, 2023[2026-06-25]. https://arxiv.org/abs/2310.06825.

[14] GU A, DAO T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces[EB/OL]. arXiv:2312.00752, 2023[2026-06-25]. https://arxiv.org/abs/2312.00752.

[15] CHU X, QIAO L, LIN X, et al. MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices[EB/OL]. arXiv:2312.16886, 2023[2026-06-25]. https://arxiv.org/abs/2312.16886.

[16] ZHU Y, ZHU M, LIU N, et al. LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model[EB/OL]. arXiv:2401.02330, 2024[2026-06-25]. https://arxiv.org/abs/2401.02330.

[17] ZHOU B, HU Y, WENG X, et al. TinyLLaVA: A Framework of Small-Scale Large Multimodal Models[EB/OL]. arXiv:2402.14289, 2024[2026-06-25]. https://arxiv.org/abs/2402.14289.

[18] WANG T, ZHOU W, ZENG Y, et al. EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning[EB/OL]. arXiv:2210.07795, 2022[2026-06-25]. https://arxiv.org/abs/2210.07795.

[19] SHANG Y, CAI M, XU B, et al. LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models[EB/OL]. arXiv:2403.15388, 2024[2026-06-25]. https://arxiv.org/abs/2403.15388.

[20] ZHANG Q, CHENG A, LU M, et al. Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs[EB/OL]. arXiv:2412.01818, 2024[2026-06-25]. https://arxiv.org/abs/2412.01818.

[21] ZHANG C, MA K, FANG T, et al. VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models[EB/OL]. arXiv:2505.22654, 2025[2026-06-25]. https://arxiv.org/abs/2505.22654.

[22] TONG J, JIN W, QIN P, et al. FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models[EB/OL]. arXiv:2505.19536, 2025[2026-06-25]. https://arxiv.org/abs/2505.19536.

[23] XU R, YAO Y, GUO Z, et al. LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images[EB/OL]. arXiv:2403.11703, 2024[2026-06-25]. https://arxiv.org/abs/2403.11703.

[24] LI W, YUAN Y, LIU J, et al. TokenPacker: Efficient Visual Projector for Multimodal LLM[EB/OL]. arXiv:2407.02392, 2024[2026-06-25]. https://arxiv.org/abs/2407.02392.

[25] VASU P K A, FAGHRI F, LI C L, et al. FastVLM: Efficient Vision Encoding for Vision Language Models[EB/OL]. arXiv:2412.13303, 2024[2026-06-25]. https://arxiv.org/abs/2412.13303.

[26] CHU X, QIAO L, ZHANG X, et al. MobileVLM V2: Faster and Stronger Baseline for Vision Language Model[EB/OL]. arXiv:2402.03766, 2024[2026-06-25]. https://arxiv.org/abs/2402.03766.

[27] MARAFIOTI A, ZOHAR O, FARRE M, et al. SmolVLM: Redefining small and efficient multimodal models[EB/OL]. arXiv:2504.05299, 2025[2026-06-25]. https://arxiv.org/abs/2504.05299.

[28] WANG C, WANG Z, XU X, et al. Q-VLM: Post-training Quantization for Large Vision-Language Models[EB/OL]. arXiv:2410.08119, 2024[2026-06-25]. https://arxiv.org/abs/2410.08119.

[29] BHATNAGAR S, XU A, TAN K H, et al. LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models[EB/OL]. arXiv:2509.23729, 2025[2026-06-25]. https://arxiv.org/abs/2509.23729.

[30] XIANG Z, ZENG F, FANG H, et al. Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients[EB/OL]. arXiv:2603.17809, 2026[2026-06-25]. https://arxiv.org/abs/2603.17809.

[31] FENG Z, YANG S, DUAN B, et al. EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens[EB/OL]. arXiv:2511.21106, 2025[2026-06-25]. https://arxiv.org/abs/2511.21106.

[32] CHEN L, ZHAO X, DING K, et al. Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions[EB/OL]. arXiv:2602.09483, 2026[2026-06-25]. https://arxiv.org/abs/2602.09483.

[33] LIU X, LIAN Z. RSUniVLM: A Unified Vision Language Model for Remote Sensing via Granularity-oriented Mixture of Experts[EB/OL]. arXiv:2412.05679, 2024[2026-06-25]. https://arxiv.org/abs/2412.05679.

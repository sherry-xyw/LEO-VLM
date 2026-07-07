# 在轨持续学习与增量学习：星上 VLM 的必要性和技术可行性

## 1. 总体判断

星上 VLM 需要具备持续适应能力，但其必要性并不意味着对模型进行频繁或全量的再训练。低轨长期运行中，地域、季节、成像条件、传感器状态、任务指令与模态配置会持续变化，并共同作用于视觉表征、视觉—语言对齐和任务输出。CLeaRS 的长时程、任务增量与模态增量评测表明，现有 VLM 在这类连续变化下会发生明显遗忘，传统持续学习方法也难以稳定维持跨任务、跨指令与跨模态能力 [1]；RemoteCLIP 则说明，通用视觉—语言表征经过遥感领域对齐后，才能更可靠地支撑后续遥感任务 [2]。因此，静态部署的通用 VLM 难以覆盖卫星全寿命周期内的能力需求，需要建立面向变化的增量适应与版本更新机制。

星上更新应遵循“先非参数、后参数；先局部、后协同”的分层原则。区域知识、任务规则和历史案例等外部知识变化，优先通过 Prompt、案例记忆、RAG 或规则库更新处理；短期成像退化和轻度域偏移，优先通过统计量、原型或测试时适应进行可复位校准；只有当出现持续视觉域失配、新类别、新模态或复杂任务扩展，且轻量更新无法恢复性能时，才在地面或算力星开展带历史回放与回归评测的选择性 PEFT，并向星上发布 Adapter、LoRA、连接器、任务头或知识库增量。已有零样本在轨 VLM 演示表明，Prompt 可以支持部分低风险任务重配置 [3]；卫星—地面协同推理及在轨 GeoFM 工作也表明，模型部署与协同处理已具备初步可行性 [4,5]，但仍不能替代经验证的能力学习与安全发布。

---

## 2. 必要性

星上 VLM 的持续适应需求可归纳为三类：观测场景、任务与模态变化引起的表征和对齐失配，目标类别与能力边界扩展引起的结构化判别需求，以及外部知识和任务规则更新引起的上下文失效。

### 2.1 观测场景、任务与模态迁移

低轨遥感观测并非来自固定、同分布的数据源。地域、季节、太阳高度、云雾、海况、灾害阶段、观测几何、分辨率、传感器噪声以及可见光/SAR/红外/多光谱等模态差异，都会改变输入分布和视觉—语言对齐关系。在此基础上，星上任务还可能由封闭集识别扩展为图文描述、视觉问答、目标指代、事件解释和告警报告，或由光学图像接入 SAR、红外、多光谱或高光谱输入；相应的任务指令、输出字段、告警规则和格式约束也会变化。

CLeaRS 的长时程、模态增量和任务增量评测表明，传统持续学习策略不能简单外推到 RS-VLM：场景、任务、指令和模态变化会同时影响跨模态对齐、任务路由与旧能力保持 [1]。通用 MLLM 的连续学习研究也显示，面对新领域与新能力时，模型会发生灾难性干扰，需要结合参数隔离和任务路由维持旧能力 [6]；模态不一致持续学习研究进一步指出，模态类型和任务类型同时变化时，遗忘比单一类别增量更严重 [7]。

### 2.2 类别拓展

卫星在役周期内会遇到训练集中未定义的新目标、新型设施、突发灾害形态或细粒度属性。对灾害告警、目标监视和态势感知等高风险任务，模型不仅要“能描述”，还要能稳定地检测、定位或分类。现有遥感增量目标检测研究已围绕“以少量新类样本学习新类别，同时保持基类性能”开展探索。例如，InfRS 利用类别原型与校准机制缓解新旧类别间遗忘，并在 NWPU VHR-10 与 DIOR 上完成验证 [8]。这类工作虽然面向专用检测模型，而非 VLM，但说明当任务需要结构化、高置信输出时，新类别仍需要基于样本、原型和局部参数更新进行适配。

对于 VLM，类别拓展还会带来文本标签语义、视觉目标边界和生成描述之间的耦合问题。MLLM-CL 将新领域和新能力的连续引入作为不同设置，并通过参数隔离与路由降低干扰 [6]；面向 CLIP 的增量 Prompt 学习则利用稳定文本原型约束视觉 Prompt，以减少新类别学习造成的表征漂移 [9]。因此，开放词表识别适合用于候选发现、初步描述和困难样本筛选，但不应替代面向高风险类别的地面再训练与回归验证。

### 2.3 外部知识与任务规则更新

区域背景、任务规则、历史案例、目标属性、载荷手册和地面反馈均可能更新，但这类变化通常不意味着视觉表征已经失效。若直接修改 VLM 权重，不仅资源代价更高，还会引入遗忘和错误发布风险。RS-RAG 将遥感图像与文本知识编码到统一向量空间，并通过检索、重排序和知识增强提示改善图像描述、分类与 VQA [10]。因此，对于知识类变化，应优先维护具有来源、时间戳、地域和版本信息的检索记忆，而不是改动主模型权重。

由此，三类变化分别指向运行状态校准、外部知识更新与局部能力学习；当更新需要跨卫星共享或长期积累时，还需要进一步考虑协同聚合和反馈闭环。

---

## 3. 技术路线

以下五条路线依次对应运行状态校准、外部知识更新、局部参数适配、星座级协同和长期经验积累。它们并非互斥方案：前两类优先处理可逆、低成本变化；后三类在变化持续、影响能力边界或需要跨节点共享时逐步介入。

### 技术一：漂移检测 + 轻量自监督/测试时适应

**内涵：** 在轨 VLM 面临的是非平稳观测环境：训练或部署验证阶段形成的输入分布、表征分布和视觉—语言对应关系，会随轨道、季节、传感器状态和任务语境持续变化。漂移检测是对当前观测流与参考分布之间差异的持续估计。在缺少实时真值标签的星上环境中，这种估计通常不能直接依赖任务准确率，而应结合图像统计、表征距离、图文相似度、检索命中率、输出不确定性和少量地面审核反馈来判断模型假设是否仍然可信。它解决的是“是否已经偏离既有适用范围”的问题，而不是直接学习新类别或新任务 [1,11]。

轻量自监督适应与测试时适应（test-time adaptation, TTA）是源数据不可用或标签不可用条件下的运行期校正方法。自监督适应把未标注观测样本转化为重建、一致性或时相预测等辅助任务，以约束视觉表征不要随环境变化而失稳；TTA 则在推理时利用测试数据和模型自身参数进行在线更新，例如通过自监督任务或预测熵最小化调整少量状态 [29,33]。对于 VLM，这类方法适合修正成像条件、噪声或轻度域偏移造成的置信度和对齐偏差；若变化来自新目标语义、新任务规则或新模态结构，仅靠 TTA 通常不够，需要升级到知识更新或参数适配。

**适用变化：** 成像退化、季节/天气变化、传感器噪声、轻度地域偏移等短期或渐进式域迁移。

**可更新对象：** 类别原型、置信度温度参数，或视觉端的小型 Adapter/LoRA。

**相关研究：**
- 面向 VLM 的 SubTTA 通过对齐视觉与文本的语义子空间，并在未标注测试流上进行测试时适应，专门处理分布偏移造成的模态间隙和视觉噪声问题 [11]。这说明 VLM 的轻量适应不能仅关注视觉特征，还需要约束跨模态对齐关系。
- Black-CL 提出面向无法访问模型权重或梯度、且计算受限的 VLM 持续学习设定；其中 BETA 仅优化文本原型，并结合潜在分布回放与测试时原型适应完成增量更新。尽管并非卫星场景，其“极低可训练参数 + 无主干反传”的假设与星上受限环境具有较强可迁移性 [12]。
- CoSMAE 在遥感连续自监督学习中结合数据混合和模型混合蒸馏，在多个任务序列上相对既有持续学习基线最高提升 4.94%，为无标注遥感观测流维持视觉表征连续性提供了补充思路 [13]。
- 面向卫星影像在线域适应的工作表明，更新目标域 BatchNorm 统计量和全局类别中心可以避免反向传播，适合处理图像退化带来的快速失配 [14]；但该类方法属于视觉模型层面的轻量策略，不能替代 VLM 的跨模态适应。

### 技术二：案例记忆库 / RAG / 规则库更新

**内涵：** 案例记忆库、RAG 与规则库是 VLM 的外部知识层。RAG 的基本思想是让生成模型同时依赖参数化知识和可检索的非参数记忆，从而把部分事实更新、证据来源和上下文约束从模型权重中分离出来 [10,30]。在星上任务中，这一层主要处理区域背景、任务规则、历史案例、目标属性和处置经验等时效性知识，因为这些知识发生变化时，视觉—语言表征本身未必失效。

案例记忆库保存“在哪种观测条件和任务背景下出现过什么情况、如何判断、如何处置”的经验条目；RAG 将当前图像、指令或遥测状态与外部证据关联起来，使回答具备可追溯依据；规则库则规定告警阈值、输出格式、任务边界和禁止行为。三者共同解决知识更新和证据约束问题，但不直接修复视觉编码器的域偏移，也不能保证检索到的内容一定正确。因此，星上 RAG 更应被视为带来源、时间、地域、载荷和版本元数据的证据调用机制，而不是简单的“加一个知识库”。

**适用变化：** 区域背景、任务规则、历史事件、故障案例、目标属性、地面审核反馈等知识变化。

**可更新对象：** 向量索引、案例摘要、规则文件、目标原型、检索缓存和可信度元数据。

**相关研究：**
- RS-RAG 在遥感图像描述、分类与 VQA 实验中展示了多模态检索增强的收益 [10]。
- Self-RAG 将检索、生成与反思统一在语言模型框架中，使模型能够按需检索并评估检索内容和生成内容的质量，为高可信知识调用提供了可借鉴的机制 [15]。
- Reflexion 将环境反馈以文本式 episodic memory 的形式保留并用于后续决策，说明经验积累可以优先通过非参数记忆实现，而不必频繁修改模型权重 [16]。

### 技术三：局部任务头、Adapter 与 LoRA 更新

**内涵：** 局部任务头、Adapter 与 LoRA 是在冻结大部分基座模型的前提下引入小规模可训练模块的参数高效适配方法。Adapter 通过少量插入模块为新任务提供额外变换，LoRA 通过低秩矩阵表示权重更新方向，局部任务头则把共享表征映射到检测、分类、分割或结构化输出空间 [17,18,31]。它们的共同作用不是重新训练一个 VLM，而是把新区域、新类别、新任务或新模态引起的能力差异限制在可单独发布、比较和回滚的增量模块中。

在星上 VLM 中，PEFT 的关键不是“参数少”本身，而是更新对象要与变化来源对应。视觉域失配更可能影响视觉编码器或视觉 Adapter，新谱段或新传感器更可能影响模态接入层和视觉—语言连接器，新任务输出则更适合通过任务头或指令侧适配表达。这样做可以降低上注包大小和训练成本，但不能自动避免遗忘；每个增量模块仍需要旧任务回归、代表样本或原型回放、版本路由和安全回滚。

**适用变化：** 持续域失配、新类别、新型任务能力、跨区域适配，以及新载荷/新谱段带来的局部模块失配。

**可更新对象：** 分类/检测/分割任务头、视觉—语言连接器、视觉 Adapter、语言主干 LoRA，或任务专属 Adapter bank。

**相关研究：**
- LoRA 最初面向大语言模型提出，通过冻结基座权重并注入低秩更新矩阵，大幅减少可训练参数和训练显存需求，为大模型选择性微调提供了基础方法 [17]；LLaMA-Adapter V2 进一步将参数高效适配扩展到视觉指令模型，说明视觉 token 融合、跨模态对齐和指令能力可以通过少量新增参数进行针对性调整 [18]。
- TPPT 利用稳定的文本原型引导视觉与文本 Prompt 的增量学习，体现了“以语言语义锚点约束跨模态表征漂移”的轻量持续学习思路 [9]。MoE Adapter 通过为新任务增添专家 Adapter、保留原始 CLIP 并对输入进行自动路由，缓解连续学习中的参数漂移与零样本能力退化；LLaVA-DyMoE 则指出动态 MoE 仍会因路由漂移发生遗忘，并通过 token 级分配约束减少旧任务能力损失 [19,20]。
- 当新增的是传感器或模态而非单一类别时，PathWeave 提出将单模态 Adapter 与跨模态 Adapter 组合，并通过 MoE 门控进行路径选择，在连续新增图像、视频、音频、深度和点云模态时降低训练负担 [21]。这一机制尤其值得用于讨论光学、SAR、红外或高光谱持续接入时的模块化适配。
- CLeaRS 显示，顺序 LoRA 对 RS-VLM 的任务、指令和模态转换仍不足以保证稳定性 [1]。LoRA-Det 在星载遥感有向目标检测中报告，仅更新 12.4% 参数即可达到全量微调约 97%–100% 的性能 [22]。前者说明 VLM 的 PEFT 不能只在语言侧附加 LoRA，后者可作为专用感知模型快速适配的参考，但不能直接外推为生成式 VLM 的训练成本或遗忘表现。

**必须配套的遗忘控制：** 代表样本/原型回放、旧模型或旧 Adapter 蒸馏、多地域多任务的 Adapter/LoRA bank，以及新旧版本联合评测。LoRA 只降低可训练参数与上注包大小，**并不自动消除遗忘**。

### 技术四：星地/星间异步联邦 PEFT

**内涵：** 星地/星间异步联邦 PEFT 是把联邦学习的数据本地化原则与 PEFT 的小规模增量更新结合起来。联邦学习使训练数据保留在产生节点本地，通过聚合本地计算得到的模型更新形成共享能力 [32]；PEFT 则把共享对象从完整模型参数或梯度缩小为 Adapter、LoRA、任务头、原型或检索索引差分。二者结合后，多颗卫星可以在不集中下传原始观测数据的情况下，将各自区域、季节、载荷和任务经验转化为可验证的增量能力。

“异步”指星座节点不需要在同一时间完成训练、通信和聚合，而是按照星地可见窗口、星间链路状态、能源预算和任务空闲期逐步上报更新。由于各星的数据通常非独立同分布，且 LoRA rank、模态配置和版本状态可能不同，聚合不能简单理解为平均参数，而应按任务簇、模态、版本依赖和可信度选择性整合 [23-26]。

**适用变化：** 多星面对不同地域、传感器、任务或季节数据，既需要共享能力又不适合频繁下传原始数据的场景。

**可同步对象：** LoRA/Adapter 增量、局部任务头、类别原型、漂移摘要、检索索引差分和可信度统计。

**相关研究：**
- Satellite Federated Fine-Tuning 提出卫星—地面协同的基础模型微调框架。论文明确指出，传统卫星联邦学习下卫星算力不足以有效微调大基础模型，因而需要将模型组件在卫星、地面或其他节点之间分配；其仿真结果报告训练时间约缩短 33% [23]。该工作是星上基础模型微调的重要系统参考，但并非 RS-VLM 飞行实证。
- FediLoRA 面向异构 LoRA rank 和缺失模态下的联邦多模态微调，采用逐维聚合与选择性模型编辑，说明星座中不能直接平均不同卫星的 LoRA 因子 [24]。该工作不处于卫星场景，但可作为异构载荷和不同 LoRA rank 聚合的机制参考。
- SFL-LEO 将 split learning 与 federated learning 结合，利用低轨周期性断连期间的本地训练和异步聚合；在 Starlink 带宽轨迹驱动的实验中取得了与常规 split learning 接近的准确率 [25]。
- Sparse Incremental Aggregation in Satellite Federated Learning 表明，在轨内 ISL 上使用稀疏化与逐跳增量聚合可提升带宽效率 [26]。

### 技术五：VLM 自进化

**内涵：** VLM 自进化是从持续观测、模型预测、任务反馈和多源一致性中生成候选经验，并将其用于后续知识更新或能力更新的过程。现有自进化 VLM 研究通常依赖模型自生成问题、伪答案、反思反馈或奖励信号，从未标注图像中构造可训练的中间监督 [27,28]。在星上场景中，自进化更准确地说是“候选经验生产机制”：它把观测流转化为待验证案例、困难样本、候选问答或 PEFT 训练样本，而不是让模型直接自主重训。

**适用变化：** 标签稀缺、观测流持续到来、存在地面反馈或多源一致性信号，希望逐步积累经验的长期运行场景。

**相关研究：**
- RISE 提出从未标注图像中通过“提问者—求解者”闭环进行 VLM 自进化，并引入问题质量监督和技能分布平衡以降低伪标签退化和模式塌缩风险 [27]。需要双角色生成、多轮采样、质量评估和再训练，计算与验证开销较大。
- VisPlay 将单一基础 VLM 划分为 Image-Conditioned Questioner 与 Multimodal Reasoner，并采用 GRPO、难度奖励和多样性奖励，在未标注图像上生成问题与 silver responses，从而在视觉推理、组合泛化和幻觉抑制上提升性能 [28]。
- Self-RAG 与 Reflexion 分别提供了“检索—生成—自检”与“记忆驱动反思”的非参数化经验积累范式 [15,16]。

---

## 参考文献

[1] WENG X, NI R, PANG C, et al. Continual Vision-Language Learning for Remote Sensing: Benchmarking and Analysis[EB/OL]. arXiv:2604.00820, 2026[2026-06-25]. https://arxiv.org/abs/2604.00820.

[2] LIU F, CHEN D, GUAN Z, et al. RemoteCLIP: A Vision Language Foundation Model for Remote Sensing[EB/OL]. arXiv:2306.11029, 2023[2026-06-25]. https://arxiv.org/abs/2306.11029.

[3] DELFA VICTORIA J M, JOHN T C, HERSON A W. NAVI-Orbital: First In-Orbit Demonstration of a Zero-Shot Vision-Language Model for Autonomous Earth Observation[EB/OL]. arXiv:2606.18271, 2026[2026-06-25]. https://arxiv.org/abs/2606.18271.

[4] ZHANG Y, YANG J, CHEN Z, et al. A Satellite-Ground Synergistic Large Vision-Language Model System for Earth Observation[EB/OL]. arXiv:2507.05731, 2025[2026-06-25]. https://arxiv.org/abs/2507.05731.

[5] DU A, DEL PRETE R, MOUSIST A, et al. First On-Orbit Demonstration of a Geospatial Foundation Model[EB/OL]. arXiv:2512.01181, 2025[2026-06-25]. https://arxiv.org/abs/2512.01181.

[6] ZHAO H, ZHU F, WANG R, et al. MLLM-CL: Continual Learning for Multimodal Large Language Models[EB/OL]. arXiv:2506.05453, 2025[2026-06-25]. https://arxiv.org/abs/2506.05453.

[7] PIAN W, DENG S, MO S, et al. Modality-Inconsistent Continual Learning of Multimodal Large Language Models[EB/OL]. arXiv:2412.13050, 2024[2026-06-25]. https://arxiv.org/abs/2412.13050.

[8] LI W, ZHOU J, LI X, et al. InfRS: Incremental Few-Shot Object Detection in Remote Sensing Images[EB/OL]. arXiv:2405.11293, 2024[2026-06-25]. https://arxiv.org/abs/2405.11293.

[9] LU H, ZHANG X, MOORE K, et al. Continual Learning on CLIP via Incremental Prompt Tuning with Intrinsic Textual Anchors[EB/OL]. arXiv:2505.20680, 2025[2026-06-25]. https://arxiv.org/abs/2505.20680.

[10] WEN C, LIN Y, QU X, et al. RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model[EB/OL]. arXiv:2504.04988, 2025[2026-06-25]. https://arxiv.org/abs/2504.04988.

[11] ZENG Z, BAO W, LIN X, et al. Subspace Alignment for Vision-Language Model Test-time Adaptation[EB/OL]. arXiv:2601.08139, 2026[2026-06-25]. https://arxiv.org/abs/2601.08139.

[12] LI Y, FANG W, GAO H, et al. Black-Box Continual Learning for Vision-Language Models[EB/OL]. arXiv:2606.22999, 2026[2026-06-25]. https://arxiv.org/abs/2606.22999.

[13] MÖLLENBROK L, RASTI B, DEMIR B. Continual Self-Supervised Learning with Masked Autoencoders in Remote Sensing[EB/OL]. arXiv:2506.21312, 2025[2026-06-25]. https://arxiv.org/abs/2506.21312.

[14] NILOY F F, BHAUMIK K K, WOO S S. Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images under Image Degradation[EB/OL]. arXiv:2401.02113, 2024[2026-06-25]. https://arxiv.org/abs/2401.02113.

[15] ASAI A, WU Z, WANG Y, et al. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection[EB/OL]. arXiv:2310.11511, 2023[2026-06-25]. https://arxiv.org/abs/2310.11511.

[16] SHINN N, CASSANO F, BERMAN E, et al. Reflexion: Language Agents with Verbal Reinforcement Learning[EB/OL]. arXiv:2303.11366, 2023[2026-06-25]. https://arxiv.org/abs/2303.11366.

[17] HU E J, SHEN Y, WALLIS P, et al. LoRA: Low-Rank Adaptation of Large Language Models[EB/OL]. arXiv:2106.09685, 2021[2026-06-25]. https://arxiv.org/abs/2106.09685.

[18] GAO P, HAN J, ZHANG R, et al. LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model[EB/OL]. arXiv:2304.15010, 2023[2026-06-25]. https://arxiv.org/abs/2304.15010.

[19] YU J, ZHUGE Y, ZHANG L, et al. Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters[EB/OL]. arXiv:2403.11549, 2024[2026-06-25]. https://arxiv.org/abs/2403.11549.

[20] ZHAO C, LI M, LU H, et al. On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models[EB/OL]. arXiv:2603.27481, 2026[2026-06-25]. https://arxiv.org/abs/2603.27481.

[21] YU J, XIONG H, ZHANG L, et al. LLMs Can Evolve Continually on Modality for X-Modal Reasoning[EB/OL]. arXiv:2410.20178, 2024[2026-06-25]. https://arxiv.org/abs/2410.20178.

[22] PU X, XU F. Low-Rank Adaption on Transformer-based Oriented Object Detector for Satellite Onboard Processing of Remote Sensing Images[EB/OL]. arXiv:2406.02385, 2024[2026-06-25]. https://arxiv.org/abs/2406.02385.

[23] ZHU Y, ZHU J, WANG T, et al. Satellite Federated Fine-Tuning for Foundation Models in Space Computing Power Networks[EB/OL]. arXiv:2504.10403, 2025[2026-06-25]. https://arxiv.org/abs/2504.10403.

[24] YANG L, NGUYGEN N K, HU P, et al. FediLoRA: Heterogeneous LoRA for Federated Multimodal Fine-tuning under Missing Modalities[EB/OL]. arXiv:2509.06984, 2025[2026-06-25]. https://arxiv.org/abs/2509.06984.

[25] WU J, ZHANG J, LIN Z, et al. SFL-LEO: Asynchronous Split-Federated Learning Design for LEO Satellite-Ground Network Framework[EB/OL]. arXiv:2504.13479, 2025[2026-06-25]. https://arxiv.org/abs/2504.13479.

[26] RAZMI N, MUKHERJEE S, MATTHIESEN B, et al. Sparse Incremental Aggregation in Satellite Federated Learning[EB/OL]. arXiv:2501.11385, 2025[2026-06-25]. https://arxiv.org/abs/2501.11385.

[27] XU C, MIAO Y, ZHANG P, et al. RISE: Reliable Improvement in Self-Evolving Vision-Language Models[EB/OL]. arXiv:2605.20914, 2026[2026-06-25]. https://arxiv.org/abs/2605.20914.

[28] HE Y, HUANG C, LI Z, et al. VisPlay: Self-Evolving Vision-Language Models from Images[EB/OL]. arXiv:2511.15661, 2025[2026-06-29]. https://arxiv.org/abs/2511.15661.

[29] WANG D, SHELHAMER E, LIU S, et al. Tent: Fully Test-Time Adaptation by Entropy Minimization[EB/OL]. OpenReview, 2021[2026-06-29]. https://openreview.net/forum?id=uXl3bZLkr3c.

[30] LEWIS P, PEREZ E, PIKTUS A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks[EB/OL]. arXiv:2005.11401, 2020[2026-06-29]. https://arxiv.org/abs/2005.11401.

[31] HOULSBY N, GIURGIU A, JASTRZEBSKI S, et al. Parameter-Efficient Transfer Learning for NLP[EB/OL]. arXiv:1902.00751, 2019[2026-06-29]. https://arxiv.org/abs/1902.00751.

[32] MCMAHAN H B, MOORE E, RAMAGE D, et al. Communication-Efficient Learning of Deep Networks from Decentralized Data[EB/OL]. arXiv:1602.05629, 2016[2026-06-29]. https://arxiv.org/abs/1602.05629.

[33] SUN Y, WANG X, LIU Z, et al. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts[EB/OL]. arXiv:1909.13231, 2019[2026-06-29]. https://arxiv.org/abs/1909.13231.

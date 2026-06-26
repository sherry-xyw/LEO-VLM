# 在轨持续学习与增量学习：星上 VLM 的必要性和技术可行性

> **问题：** 面向低轨卫星及低轨算力星座部署视觉—语言模型（VLM）时，是否有必要在地面投入时间进行再训练？哪些变化可由星上 Prompt、RAG 或轻量适应解决，哪些变化必须通过地面选择性再训练/参数高效微调（PEFT）解决？

## 1. 总体判断

有必要在地面为星上 VLM 进行再训练，但通常不是重新预训练完整 VLM。对于已经完成遥感领域预训练或指令微调的 VLM，更合理的范式是：针对在轨发现的持续域失配、新类别、新模态或复杂任务扩展，在地面开展带历史回放和回归评测的选择性 PEFT，再上注 Adapter、LoRA、连接器、任务头或知识库增量。

为系统考察 RS-VLM 在连续变化环境下的适应能力，CLeaRS 构建了包含 10 个遥感子集、超过 20.7 万图文对的基准，并设计了长时程、模态增量和任务增量三种评测协议。实验结果表明，多种 VLM 在这些连续变化场景下均会出现灾难性遗忘，代表性持续学习方法也难以稳定应对任务切换、自然语言指令变化与模态转换 [1]。

地面再训练的价值不仅在于补充新数据，也在于完成遥感领域对齐。RemoteCLIP 通过遥感图文数据预训练视觉—语言模型，在零样本分类、图文检索等任务上相较原始 CLIP 获得稳定提升，说明通用视觉—语言表征需要经过遥感领域适配，才能更可靠地服务后续任务 [2]。

与此同时，零样本 VLM 能够覆盖部分低风险变化。NAVI-Orbital 报告了在 LEO 卫星上部署 Gemma 3，通过自然语言 Prompt 完成场景分类、图像描述和任务重配置；其飞行载荷未进行微调 [3]。SpaceVerse 则提出以星上紧凑 LVLM 处理轻量任务、地面常规 LVLM 处理复杂任务的卫星—地面协同推理系统，表明 VLM 已开始进入面向低轨遥感的系统部署讨论，但其重点仍是协同推理与数据选择，而非持续参数更新 [4]。与之相邻的在轨 GeoFM 工作已经展示了压缩和领域适配后的视觉基础模型可在 ISS 载荷上完成可靠推理，但该工作不涉及 VLM 的在线学习 [5]。

从模型层面看，VLM 的参数高效适配具有可行性。LLaMA-Adapter V2 通过视觉 Adapter、早期视觉 token 融合和少量可学习参数完成视觉指令适配，说明视觉—语言能力可以在不全量更新模型的条件下进行针对性调整 [6]。但在星上场景中，是否训练、训练哪些模块以及如何发布，仍需由任务风险、视觉域变化程度、可用资源和历史能力保持共同决定。

综合现有研究，可以形成以下判断：RS-VLM 的长期、模态与任务变化确实需要持续适应机制；遥感领域对齐能够改善通用 VLM 的下游适用性；零样本 VLM 已可在轨完成部分语义理解与自然语言重任务化，但公开飞行演示尚未覆盖在线微调；RAG、VLM/LLM 参数高效微调、星地协同微调与受控自进化分别对应不同层级的更新需求，不能相互替代。

---

## 2. 必要性

### 2.1 场景迁移

低轨遥感观测并非来自固定、同分布的数据源。地域、季节、太阳高度、云雾、海况、灾害阶段、观测几何、分辨率、传感器噪声以及可见光/SAR/红外/多光谱等模态差异，都会改变输入分布和视觉—语言对齐关系。

VLM 在轨运行时可能从封闭集检测扩展为图文描述、视觉问答、目标指代、事件解释和告警报告；也可能从光学图像接入 SAR、红外、多光谱或高光谱输入。任务指令、输出字段、告警规则和格式约束也可能变化。

CLeaRS 的长时程、模态增量和任务增量评测表明，传统持续学习策略不能简单外推到 RS-VLM：任务、指令和模态变化会同时影响跨模态对齐、任务路由与旧能力保持 [1]。通用 MLLM 的连续学习研究也显示，模型面对新领域与新能力时会发生灾难性干扰，需要结合参数隔离和任务路由来维持旧能力 [7]；而模态不一致持续学习研究进一步指出，模态类型和任务类型同时变化时，遗忘会比单一类别增量更严重 [8]。

因此，星上 VLM 的持续适应不仅是恢复视觉精度，还需要维持图文语义对齐、指令遵循、生成可靠性和拒答边界。

### 2.2 类别拓展

卫星在役周期内会遇到训练集中未定义的新目标、新型设施、突发灾害形态或细粒度属性。对灾害告警、目标监视和态势感知等高风险任务，模型不仅要“能描述”，还要能稳定地检测、定位或分类。

现有遥感增量目标检测研究已围绕“以少量新类样本学习新类别，同时保持基类性能”开展探索。例如，InfRS 利用类别原型与校准机制缓解新旧类别间遗忘，并在 NWPU VHR-10 与 DIOR 上完成验证 [9]。这类工作虽然面向专用检测模型，而非 VLM，但说明当任务需要结构化、高置信输出时，新类别仍需要基于样本、原型和局部参数更新进行适配。

对于 VLM，类别拓展还会带来文本标签语义、视觉对象边界和生成描述之间的耦合问题。MLLM-CL 将新领域和新能力的连续引入作为不同设置，并通过参数隔离与路由降低干扰 [7]；面向 CLIP 的增量 Prompt 学习则利用稳定文本原型约束视觉 Prompt，以减少新类别学习造成的表征漂移 [18]。因此，开放词表识别适合用于候选发现、初步描述和困难样本筛选，但不应替代面向高风险类别的地面再训练与回归验证。

### 2.3 外部知识与任务规则更新

区域背景、任务规则、历史案例、目标属性、载荷手册和地面反馈均可能更新，但这些变化通常不意味着视觉表征已经失效。若直接修改 VLM 权重，不仅资源代价更高，还会引入遗忘和错误发布风险。

RS-RAG 将遥感图像与文本知识编码到统一向量空间，并通过检索、重排序和知识增强提示改善图像描述、分类与 VQA [10]。因此，对于知识类变化，应优先维护具有来源、时间戳、地域和版本信息的检索记忆，而不是改动主模型权重。

需要明确的是，RAG 的适用边界在于更新“模型可访问的知识”，而不是修复“模型如何看懂图像”。若性能下降源于视觉编码器失配、跨模态连接器偏移或新目标的细粒度区分不足，检索增强不能替代参数适配。

---

## 3. 技术路线

下面的技术路线按更新成本、风险和系统复杂度从低到高排列。

### 技术一：漂移检测 + 轻量自监督/测试时适应

**适用变化：** 成像退化、季节/天气变化、传感器噪声、轻度地域偏移等短期或渐进式域迁移。

**可更新对象：** 归一化统计量、类别原型、置信度温度参数，或视觉端的小型 Adapter/LoRA。

**相关研究与启示：**
- 面向 VLM 的 SubTTA 通过对齐视觉与文本的语义子空间，并在未标注测试流上进行测试时适应，专门处理分布偏移造成的模态间隙和视觉噪声问题 [11]。这说明 VLM 的轻量适应不能仅关注视觉特征，还需要约束跨模态对齐关系。
- Black-CL 面向无法访问模型权重或梯度、且计算受限的 VLM 持续学习设置，仅使用文本原型、潜在分布回放与测试时原型适应实现增量更新。尽管并非卫星场景，其“极低可训练参数 + 无主干反传”的假设与星上受限环境具有较强可迁移性 [12]。
- CoSMAE 在遥感连续自监督学习中结合数据混合和模型混合蒸馏，在多个任务序列上相对既有持续学习基线最高提升 4.94%，为无标注遥感观测流维持视觉表征连续性提供了补充思路 [13]。
- 面向卫星影像在线域适应的工作表明，更新目标域 BatchNorm 统计量和全局类别中心可以避免反向传播，适合处理图像退化带来的快速失配 [14]；但该类方法属于视觉模型层面的轻量策略，不能替代 VLM 的跨模态适应。

**面向 VLM 的可行性判断：中高。** 该路线适合处理短期和轻度失配，且应优先更新视觉侧统计量、原型或极小模块。它不宜替代面向长期域失配、新类别或新模态的专门适配。

### 技术二：案例记忆库 / RAG / 规则库更新

**适用变化：** 区域背景、任务规则、历史事件、故障案例、目标属性、地面审核反馈等知识变化。

**可更新对象：** 向量索引、案例摘要、规则文件、目标原型、检索缓存和可信度元数据。

**相关研究与启示：**
- RS-RAG 在遥感图像描述、分类与 VQA 实验中展示了多模态检索增强的收益 [10]。
- Self-RAG 将检索、生成与反思统一在语言模型框架中，使模型能够按需检索并评估检索内容和生成内容的质量，为高可信知识调用提供了可借鉴的机制 [15]。
- Reflexion 将环境反馈以文本式 episodic memory 的形式保留并用于后续决策，说明经验积累可以优先通过非参数记忆实现，而不必频繁修改模型权重 [16]。

**面向 VLM 的可行性判断：最高。** 记忆更新增量小、可追溯、可删除、易签名和回滚，适合优先处理知识性变化。

**边界：** 记忆/RAG 解决的是“模型可访问什么知识”，不能替代“模型是否已学会稳定视觉判别”。若新类别或新域导致视觉特征失配，仍需局部 PEFT 或地面再训练。

### 技术三：局部任务头、Adapter 与 LoRA 更新

**适用变化：** 持续域失配、新类别、新型任务能力、跨区域适配，以及新载荷/新谱段带来的局部模块失配。

**可更新对象：** 分类/检测/分割任务头、视觉—语言连接器、视觉 Adapter、语言主干 LoRA，或任务专属 Adapter bank。

**相关研究与启示：**
- LoRA 最初面向大语言模型提出，通过冻结基座权重并注入低秩更新矩阵，大幅减少可训练参数和训练显存需求，为大模型选择性微调提供了基础方法 [17]。
- LLaMA-Adapter V2 进一步将参数高效适配扩展到视觉指令模型，说明视觉 token 融合、跨模态对齐和指令能力可以通过少量新增参数进行针对性调整 [6]。
- TPPT 利用稳定的文本原型引导视觉与文本 Prompt 的增量学习，体现了“以语言语义锚点约束跨模态表征漂移”的轻量持续学习思路 [18]。
- MoE Adapter 方法通过为新任务增添专家 Adapter、保留原始 CLIP 并对输入进行自动路由，缓解连续学习中的参数漂移与零样本能力退化 [19]。更进一步的 LLaVA-DyMoE 面向大型视觉语言模型的连续指令微调，指出动态 MoE 仍会因路由漂移发生遗忘，并通过 token 级分配约束减少旧任务能力损失 [20]。
- 当新增的是传感器或模态而非单一类别时，PathWeave 提出将单模态 Adapter 与跨模态 Adapter 组合，并通过 MoE 门控进行路径选择，在连续新增图像、视频、音频、深度和点云模态时降低训练负担 [21]。这一机制尤其值得用于讨论光学、SAR、红外或高光谱持续接入时的模块化适配。
- CLeaRS 显示，顺序 LoRA 对 RS-VLM 的任务、指令和模态转换仍不足以保证稳定性 [1]。因此，VLM 的 PEFT 不能只在语言侧附加 LoRA，而应根据变化来源选择视觉端、连接器、语言端或任务头。
- LoRA-Det 在星载遥感有向目标检测中报告，仅更新 12.4% 参数即可达到全量微调约 97%–100% 的性能 [22]。该结果适合作为专用感知模型快速适配的参考，但不能直接外推为生成式 VLM 的训练成本或遗忘表现。

**面向 VLM 的可行性判断：**
- **在星上直接训练：中等。** 即使仅更新 LoRA，仍需承受视觉 token、反向传播激活、优化器状态、回放缓存和验证成本；因此不宜把它等同于检测器的轻量上注更新。
- **在地面开展选择性 PEFT、再上注增量：高。** 地面能够完成难例筛选、历史任务回放、跨模态消融、量化回归和幻觉/拒答评测，是 VLM 参数更新更可靠的主路径。

**模块选择原则：**
- 视觉域变化优先适配视觉端模块；
- 跨模态失配或新模态优先适配连接器/投影层和模态 Adapter；
- 指令与输出任务变化优先适配 Prompt、任务头或语言侧 LoRA；
- 高风险目标检测/定位任务可采用“专用感知模型负责结构化判别，VLM 负责解释与问答”的组合，而不让 VLM 单独承担全部判别责任。

**必须配套的遗忘控制：** 代表样本/原型回放、旧模型或旧 Adapter 蒸馏、多地域多任务的 Adapter/LoRA bank，以及新旧版本联合评测。LoRA 只降低可训练参数与上注包大小，**并不自动消除遗忘**。

### 技术四：星地/星间异步联邦 PEFT

**适用变化：** 多星面对不同地域、传感器、任务或季节数据，既需要共享能力又不适合频繁下传原始数据的场景。

**可同步对象：** LoRA/Adapter 增量、局部任务头、类别原型、漂移摘要、检索索引差分和可信度统计。

**相关研究与启示：**
- Satellite Federated Fine-Tuning 提出卫星—地面协同的基础模型微调框架。论文明确指出，传统卫星联邦学习下卫星算力不足以有效微调大基础模型，因而需要将模型组件在卫星、地面或其他节点之间分配；其仿真结果报告训练时间约缩短 33% [23]。该工作是星上基础模型微调的重要系统参考，但并非 RS-VLM 飞行实证。
- FediLoRA 面向异构 LoRA rank 和缺失模态下的联邦多模态微调，采用逐维聚合与选择性模型编辑，说明星座中不能直接平均不同卫星的 LoRA 因子 [24]。该工作不处于卫星场景，但可作为异构载荷和不同 LoRA rank 聚合的机制参考。
- SFL-LEO 将 split learning 与 federated learning 结合，利用低轨周期性断连期间的本地训练和异步聚合；在 Starlink 带宽轨迹驱动的实验中取得了与常规 split learning 接近的准确率 [25]。
- Sparse Incremental Aggregation in Satellite Federated Learning 表明，在轨内 ISL 上使用稀疏化与逐跳增量聚合可提升带宽效率 [26]。

**面向 VLM 的可行性判断：中等。** 围绕大模型微调、异构多模态 LoRA、卫星通信与异步聚合的研究已分别给出可行路径；但针对真实在轨 RS-VLM 的在线 LoRA 训练与长期安全发布，公开飞行验证仍较少。该路线更适合被视为远期星座级适应方向，而不是近期单星 VLM 参数更新的默认方案。

### 技术五：受控的 VLM 自进化

**适用变化：** 标签稀缺、观测流持续到来、存在地面反馈或多源一致性信号，希望逐步积累经验的长期运行场景。

**可接受的“自进化”定义：**
1. 从地面确认、人工反馈、多时相一致性、跨传感器一致性和高置信预测中提取经验；
2. 优先将经验写入案例记忆或检索索引；
3. 只有在监督信号经过可信筛选后，才将其用于局部 LoRA/Adapter 候选更新；
4. 候选版本必须经过新旧任务联合评测与资源检查后，才允许被上注或启用。

**相关研究与启示：**
- RISE 提出从未标注图像中通过“提问者—求解者”闭环进行 VLM 自进化，并引入问题质量监督和技能分布平衡以降低伪标签退化和模式塌缩风险 [27]。
- Self-RAG 与 Reflexion 分别提供了“检索—生成—自检”与“记忆驱动反思”的非参数化经验积累范式 [15,16]。

**面向 VLM 的可行性判断：有限且需严格约束。** RISE 等方法需要双角色生成、多轮采样、质量评估和再训练，计算与验证开销较大；现有结果主要来自地面 VLM 基准，而非空间飞行环境。因此，近期更适合研究“记忆驱动、反馈驱动、可验证的弱自进化”，不宜让单星进行无约束的闭环自训练或自动替换稳定版本。

---

## 4. 从变化类型到 VLM 更新路径的选择规则

| 变化来源 | 首选路径 | 是否需要地面 PEFT | 何时升级 | 不应优先采用 |
|---|---|---:|---|---|
| 短期成像退化、轻度域偏移 | 统计量、原型、TTA | 否/视情况 | 漂移持续且任务性能下降 | 直接全量微调 |
| 新区域知识、规则、案例 | RAG/记忆/规则更新 | 否 | 检索无法恢复任务质量 | 修改视觉编码器 |
| 新类别且风险较低 | 开放词表筛选 + 原型记忆 | 视情况 | 误报/漏报不可接受 | 仅靠自然语言描述作最终判别 |
| 新类别且需可靠定位/告警 | 局部感知任务头/视觉 PEFT | 是 | 新旧类冲突或性能不稳 | 只更新语言侧 LoRA |
| 长期地域、季节或成像链路失配 | 视觉 Adapter/LoRA + 回放蒸馏 | 是 | 轻量适应无法恢复性能 | 仅依赖 RAG |
| 新模态、新谱段 | 连接器/模态 Adapter 联合适配 | 是 | 跨模态对齐持续失败 | 对全模型使用统一 rank 的 LoRA |
| 新复杂 VQA、推理或报告任务 | 多任务指令 PEFT + 回归评测 | 是 | Prompt 无法满足准确性与可控性 | 直接将 Prompt 更改视为能力学习 |
| 多星异构数据和任务 | 异步联邦 PEFT/记忆差分 | 多数需地面或算力星验证 | 局部更新已通过验证且可共享 | 同步 FedAvg 或平均异构 LoRA 因子 |
| 标签稀缺但有可信反馈 | 受控记忆与候选 PEFT | 视监督质量而定 | 多源一致性、伪标签与验证均达标 | 自生成数据后直接自动发布 |

---

## 参考文献

[1] WENG X, NI R, PANG C, et al. Continual Vision-Language Learning for Remote Sensing: Benchmarking and Analysis[EB/OL]. arXiv:2604.00820, 2026[2026-06-25]. https://arxiv.org/abs/2604.00820.

[2] LIU F, CHEN D, GUAN Z, et al. RemoteCLIP: A Vision Language Foundation Model for Remote Sensing[EB/OL]. arXiv:2306.11029, 2023[2026-06-25]. https://arxiv.org/abs/2306.11029.

[3] DELFA VICTORIA J M, JOHN T C, HERSON A W. NAVI-Orbital: First In-Orbit Demonstration of a Zero-Shot Vision-Language Model for Autonomous Earth Observation[EB/OL]. arXiv:2606.18271, 2026[2026-06-25]. https://arxiv.org/abs/2606.18271.

[4] ZHANG Y, YANG J, CHEN Z, et al. A Satellite-Ground Synergistic Large Vision-Language Model System for Earth Observation[EB/OL]. arXiv:2507.05731, 2025[2026-06-25]. https://arxiv.org/abs/2507.05731.

[5] DU A, DEL PRETE R, MOUSIST A, et al. First On-Orbit Demonstration of a Geospatial Foundation Model[EB/OL]. arXiv:2512.01181, 2025[2026-06-25]. https://arxiv.org/abs/2512.01181.

[6] GAO P, HAN J, ZHANG R, et al. LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model[EB/OL]. arXiv:2304.15010, 2023[2026-06-25]. https://arxiv.org/abs/2304.15010.

[7] ZHAO H, ZHU F, WANG R, et al. MLLM-CL: Continual Learning for Multimodal Large Language Models[EB/OL]. arXiv:2506.05453, 2025[2026-06-25]. https://arxiv.org/abs/2506.05453.

[8] PIAN W, DENG S, MO S, et al. Modality-Inconsistent Continual Learning of Multimodal Large Language Models[EB/OL]. arXiv:2412.13050, 2024[2026-06-25]. https://arxiv.org/abs/2412.13050.

[9] LI W, ZHOU J, LI X, et al. InfRS: Incremental Few-Shot Object Detection in Remote Sensing Images[EB/OL]. arXiv:2405.11293, 2024[2026-06-25]. https://arxiv.org/abs/2405.11293.

[10] WEN C, LIN Y, QU X, et al. RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model[EB/OL]. arXiv:2504.04988, 2025[2026-06-25]. https://arxiv.org/abs/2504.04988.

[11] ZENG Z, BAO W, LIN X, et al. Subspace Alignment for Vision-Language Model Test-time Adaptation[EB/OL]. arXiv:2601.08139, 2026[2026-06-25]. https://arxiv.org/abs/2601.08139.

[12] LI Y, FANG W, GAO H, et al. Black-Box Continual Learning for Vision-Language Models[EB/OL]. arXiv:2606.22999, 2026[2026-06-25]. https://arxiv.org/abs/2606.22999.

[13] MÖLLENBROK L, RASTI B, DEMIR B. Continual Self-Supervised Learning with Masked Autoencoders in Remote Sensing[EB/OL]. arXiv:2506.21312, 2025[2026-06-25]. https://arxiv.org/abs/2506.21312.

[14] NILOY F F, BHAUMIK K K, WOO S S. Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images under Image Degradation[EB/OL]. arXiv:2401.02113, 2024[2026-06-25]. https://arxiv.org/abs/2401.02113.

[15] ASAI A, WU Z, WANG Y, et al. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection[EB/OL]. arXiv:2310.11511, 2023[2026-06-25]. https://arxiv.org/abs/2310.11511.

[16] SHINN N, CASSANO F, BERMAN E, et al. Reflexion: Language Agents with Verbal Reinforcement Learning[EB/OL]. arXiv:2303.11366, 2023[2026-06-25]. https://arxiv.org/abs/2303.11366.

[17] HU E J, SHEN Y, WALLIS P, et al. LoRA: Low-Rank Adaptation of Large Language Models[EB/OL]. arXiv:2106.09685, 2021[2026-06-25]. https://arxiv.org/abs/2106.09685.

[18] LU H, ZHANG X, MOORE K, et al. Continual Learning on CLIP via Incremental Prompt Tuning with Intrinsic Textual Anchors[EB/OL]. arXiv:2505.20680, 2025[2026-06-25]. https://arxiv.org/abs/2505.20680.

[19] YU J, ZHUGE Y, ZHANG L, et al. Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters[EB/OL]. arXiv:2403.11549, 2024[2026-06-25]. https://arxiv.org/abs/2403.11549.

[20] ZHAO C, LI M, LU H, et al. On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models[EB/OL]. arXiv:2603.27481, 2026[2026-06-25]. https://arxiv.org/abs/2603.27481.

[21] YU J, XIONG H, ZHANG L, et al. LLMs Can Evolve Continually on Modality for X-Modal Reasoning[EB/OL]. arXiv:2410.20178, 2024[2026-06-25]. https://arxiv.org/abs/2410.20178.

[22] PU X, XU F. Low-Rank Adaption on Transformer-based Oriented Object Detector for Satellite Onboard Processing of Remote Sensing Images[EB/OL]. arXiv:2406.02385, 2024[2026-06-25]. https://arxiv.org/abs/2406.02385.

[23] ZHU Y, ZHU J, WANG T, et al. Satellite Federated Fine-Tuning for Foundation Models in Space Computing Power Networks[EB/OL]. arXiv:2504.10403, 2025[2026-06-25]. https://arxiv.org/abs/2504.10403.

[24] YANG L, NGUYGEN N K, HU P, et al. FediLoRA: Heterogeneous LoRA for Federated Multimodal Fine-tuning under Missing Modalities[EB/OL]. arXiv:2509.06984, 2025[2026-06-25]. https://arxiv.org/abs/2509.06984.

[25] WU J, ZHANG J, LIN Z, et al. SFL-LEO: Asynchronous Split-Federated Learning Design for LEO Satellite-Ground Network Framework[EB/OL]. arXiv:2504.13479, 2025[2026-06-25]. https://arxiv.org/abs/2504.13479.

[26] RAZMI N, MUKHERJEE S, MATTHIESEN B, et al. Sparse Incremental Aggregation in Satellite Federated Learning[EB/OL]. arXiv:2501.11385, 2025[2026-06-25]. https://arxiv.org/abs/2501.11385.

[27] XU C, MIAO Y, ZHANG P, et al. RISE: Reliable Improvement in Self-Evolving Vision-Language Models[EB/OL]. arXiv:2605.20914, 2026[2026-06-25]. https://arxiv.org/abs/2605.20914.

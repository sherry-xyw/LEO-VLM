# 在轨持续学习与增量学习：星上 VLM 的必要性和技术可行性

> **问题：** 面向低轨卫星及低轨算力星座部署视觉—语言模型（VLM）时，是否有必要在地面投入时间进行再训练？哪些变化可由星上 Prompt、RAG 或轻量适应解决，哪些变化必须通过地面选择性再训练/参数高效微调（PEFT）解决？

## 1. 总体判断

星上 VLM 需要具备持续适应能力，但其必要性并不意味着对模型进行频繁或全量的再训练。低轨长期运行中，地域、季节、成像条件、传感器状态、任务指令与模态配置会持续变化，并共同作用于视觉表征、视觉—语言对齐和任务输出。CLeaRS 的长时程、任务增量与模态增量评测表明，现有 VLM 在这类连续变化下会发生明显遗忘，传统持续学习方法也难以稳定维持跨任务、跨指令与跨模态能力 [1]；RemoteCLIP 则说明，通用视觉—语言表征经过遥感领域对齐后，才能更可靠地支撑后续遥感任务 [2]。因此，静态部署的通用 VLM 难以覆盖卫星全寿命周期内的能力需求，需要建立面向变化的增量适应与版本更新机制。

星上更新应遵循“先非参数、后参数；先局部、后协同”的分层原则。区域知识、任务规则和历史案例等外部知识变化，优先通过 Prompt、案例记忆、RAG 或规则库更新处理；短期成像退化和轻度域偏移，优先通过统计量、原型或测试时适应进行可复位校准；只有当出现持续视觉域失配、新类别、新模态或复杂任务扩展，且轻量更新无法恢复性能时，才在地面或算力星开展带历史回放与回归评测的选择性 PEFT，并向星上发布 Adapter、LoRA、连接器、任务头或知识库增量。已有零样本在轨 VLM 演示表明，Prompt 可以支持部分低风险任务重配置 [3]；卫星—地面协同推理及在轨 GeoFM 工作也表明，模型部署与协同处理已具备初步可行性 [4,5]，但仍不能替代经验证的能力学习与安全发布。

---

## 2. 必要性

星上 VLM 的持续适应需求可归纳为三类：观测场景、任务与模态变化引起的表征和对齐失配，目标类别与能力边界扩展引起的结构化判别需求，以及外部知识和任务规则更新引起的上下文失效。三类变化的作用对象不同，因此需要采用不同层级的更新机制。

### 2.1 观测场景、任务与模态迁移

低轨遥感观测并非来自固定、同分布的数据源。地域、季节、太阳高度、云雾、海况、灾害阶段、观测几何、分辨率、传感器噪声以及可见光/SAR/红外/多光谱等模态差异，都会改变输入分布和视觉—语言对齐关系。在此基础上，星上任务还可能由封闭集识别扩展为图文描述、视觉问答、目标指代、事件解释和告警报告，或由光学图像接入 SAR、红外、多光谱或高光谱输入；相应的任务指令、输出字段、告警规则和格式约束也会变化。

CLeaRS 的长时程、模态增量和任务增量评测表明，传统持续学习策略不能简单外推到 RS-VLM：场景、任务、指令和模态变化会同时影响跨模态对齐、任务路由与旧能力保持 [1]。通用 MLLM 的连续学习研究也显示，面对新领域与新能力时，模型会发生灾难性干扰，需要结合参数隔离和任务路由维持旧能力 [6]；模态不一致持续学习研究进一步指出，模态类型和任务类型同时变化时，遗忘比单一类别增量更严重 [7]。因此，星上 VLM 的持续适应不仅是恢复视觉精度，还需要维持图文语义对齐、指令遵循、生成可靠性和拒答边界。

### 2.2 类别拓展

卫星在役周期内会遇到训练集中未定义的新目标、新型设施、突发灾害形态或细粒度属性。对灾害告警、目标监视和态势感知等高风险任务，模型不仅要“能描述”，还要能稳定地检测、定位或分类。现有遥感增量目标检测研究已围绕“以少量新类样本学习新类别，同时保持基类性能”开展探索。例如，InfRS 利用类别原型与校准机制缓解新旧类别间遗忘，并在 NWPU VHR-10 与 DIOR 上完成验证 [8]。这类工作虽然面向专用检测模型，而非 VLM，但说明当任务需要结构化、高置信输出时，新类别仍需要基于样本、原型和局部参数更新进行适配。

对于 VLM，类别拓展还会带来文本标签语义、视觉对象边界和生成描述之间的耦合问题。MLLM-CL 将新领域和新能力的连续引入作为不同设置，并通过参数隔离与路由降低干扰 [6]；面向 CLIP 的增量 Prompt 学习则利用稳定文本原型约束视觉 Prompt，以减少新类别学习造成的表征漂移 [9]。因此，开放词表识别适合用于候选发现、初步描述和困难样本筛选，但不应替代面向高风险类别的地面再训练与回归验证。

### 2.3 外部知识与任务规则更新

区域背景、任务规则、历史案例、目标属性、载荷手册和地面反馈均可能更新，但这类变化通常不意味着视觉表征已经失效。若直接修改 VLM 权重，不仅资源代价更高，还会引入遗忘和错误发布风险。RS-RAG 将遥感图像与文本知识编码到统一向量空间，并通过检索、重排序和知识增强提示改善图像描述、分类与 VQA [10]。因此，对于知识类变化，应优先维护具有来源、时间戳、地域和版本信息的检索记忆，而不是改动主模型权重。

RAG 的适用边界在于更新“模型可访问什么知识”，而不是修复“模型如何看懂图像”。若性能下降源于视觉编码器失配、跨模态连接器偏移或新目标的细粒度区分不足，检索增强不能替代参数适配。由此，三类变化分别指向运行状态校准、外部知识更新与局部能力学习；当更新需要跨卫星共享或长期积累时，还需要进一步考虑协同聚合和反馈闭环。

---

## 3. 技术路线

以下五条路线依次对应运行状态校准、外部知识更新、局部参数适配、星座级协同和长期经验积累。它们并非互斥方案：前两类优先处理可逆、低成本变化；后三类在变化持续、影响能力边界或需要跨节点共享时逐步介入。

### 技术一：漂移检测 + 轻量自监督/测试时适应

**技术内涵：** 漂移检测是指持续比较当前在轨观测流与模型部署时的参考分布，以判断模型输入、内部表征或任务输出是否发生了足以影响可靠性的变化。其检测对象包括：输入分布漂移，如云雾、季节、太阳高度、观测几何和噪声造成的像素或谱段统计变化；特征与跨模态漂移，如视觉特征分布、图像—文本相似度和类别原型覆盖度发生偏移；以及任务表现漂移，如置信度校准变差、结构化输出错误增多或告警漏检/虚警上升。漂移检测本身不直接学习新能力，而是为后续“是否适应、适应什么、是否升级至参数更新”提供触发依据。

轻量自监督适应与测试时适应（test-time adaptation, TTA）是在当前未标注或弱标注观测流上进行局部校准的两类方法。前者通常利用掩码重建、时相一致性、多视角一致性或高置信样本等代理目标，维持视觉表征对当前观测条件的适配；后者则在推理阶段或一个任务窗口内，利用熵最小化、原型校正、伪标签一致性等目标更新少量状态。二者均强调只调整归一化统计量、类别原型、温度参数、Prompt 或极小 Adapter 等低风险对象，并保持基座模型主体不变：自监督适应更偏向连续表征校准，TTA 更偏向短时、可复位的运行时校准。

**适用变化：** 成像退化、季节/天气变化、传感器噪声、轻度地域偏移等短期或渐进式域迁移。

**可更新对象：** 归一化统计量、类别原型、置信度温度参数，或视觉端的小型 Adapter/LoRA。

**相关研究与启示：**
- 面向 VLM 的 SubTTA 通过对齐视觉与文本的语义子空间，并在未标注测试流上进行测试时适应，专门处理分布偏移造成的模态间隙和视觉噪声问题 [11]。这说明 VLM 的轻量适应不能仅关注视觉特征，还需要约束跨模态对齐关系。
- Black-CL 提出面向无法访问模型权重或梯度、且计算受限的 VLM 持续学习设定；其中 BETA 仅优化文本原型，并结合潜在分布回放与测试时原型适应完成增量更新。尽管并非卫星场景，其“极低可训练参数 + 无主干反传”的假设与星上受限环境具有较强可迁移性 [12]。
- CoSMAE 在遥感连续自监督学习中结合数据混合和模型混合蒸馏，在多个任务序列上相对既有持续学习基线最高提升 4.94%，为无标注遥感观测流维持视觉表征连续性提供了补充思路 [13]。
- 面向卫星影像在线域适应的工作表明，更新目标域 BatchNorm 统计量和全局类别中心可以避免反向传播，适合处理图像退化带来的快速失配 [14]；但该类方法属于视觉模型层面的轻量策略，不能替代 VLM 的跨模态适应。

**面向 VLM 的可行性判断：中高。** 该路线适合恢复已有能力，而不是学习新的稳定能力。星上应优先采用无梯度、可复位的统计量、原型与校准参数更新；当漂移持续、跨模态对齐无法恢复，或任务已涉及新类别、新模态和高风险告警时，应升级为局部 PEFT 与地面回归验证，而不应依赖连续伪标签自训练。

### 技术二：案例记忆库 / RAG / 规则库更新

**技术内涵：** 案例记忆库、RAG 与规则库共同构成模型外部的非参数知识层，其核心是在不修改 VLM 权重的前提下，改变模型在当前任务中可访问的事实、经验与约束。案例记忆库以结构化条目保存历史观测、事件背景、处置结论、人工反馈和任务结果；RAG 将当前图像、文本指令或遥测状态编码为查询，从外部知识库检索相关证据，再将证据组织到模型上下文中；规则库则用显式条件、阈值、任务流程和输出格式约束模型行为。三者分别承担“经验沉淀—证据调用—边界约束”的作用。

其完整机制包括知识采集、表征编码、索引构建、检索召回、重排序与时空过滤、上下文注入以及规则校验。关键不在于简单增加记忆数量，而在于为每条知识或案例建立来源、时间、地域、传感器、权限、置信度、版本和有效期等元数据，并处理过期知识、相互矛盾证据和检索错误。由于更新对象位于模型外部，记忆和规则可独立签名、增量同步、删除与回滚，具有较好的可控性。

**适用变化：** 区域背景、任务规则、历史事件、故障案例、目标属性、地面审核反馈等知识变化。

**可更新对象：** 向量索引、案例摘要、规则文件、目标原型、检索缓存和可信度元数据。

**相关研究与启示：**
- RS-RAG 在遥感图像描述、分类与 VQA 实验中展示了多模态检索增强的收益 [10]。
- Self-RAG 将检索、生成与反思统一在语言模型框架中，使模型能够按需检索并评估检索内容和生成内容的质量，为高可信知识调用提供了可借鉴的机制 [15]。
- Reflexion 将环境反馈以文本式 episodic memory 的形式保留并用于后续决策，说明经验积累可以优先通过非参数记忆实现，而不必频繁修改模型权重 [16]。

**面向 VLM 的可行性判断：最高。** 记忆更新增量小、可追溯、可删除、易签名和回滚，适合优先处理知识性变化。其边界在于：记忆/RAG 解决的是“模型可访问什么知识”，不能替代“模型是否已学会稳定视觉判别”。若新类别或新域导致视觉特征失配，仍需局部 PEFT 或地面再训练；对安全关键任务，检索结果也需经独立规则模块校验，不能直接等同于执行指令。

### 技术三：局部任务头、Adapter 与 LoRA 更新

**技术内涵：** 局部任务头、Adapter 与 LoRA 属于参数高效微调（PEFT）的不同实现形式，其共同目标是在冻结大部分基座模型参数的条件下，仅训练少量与新任务相关的增量参数，从而以较低的训练、存储和上注成本恢复或扩展能力。局部任务头位于模型输出端，用于将共享表征映射为特定任务所需的分类、检测、分割或结构化字段；Adapter 是插入网络层间的小型瓶颈模块，通过额外分支学习任务特定变换；LoRA 则将原始权重的更新限制为低秩矩阵分解，使新增参数只表示一个低维更新方向。Prompt/Prefix tuning 虽不直接改变主干权重，也可作为参数高效的任务条件化方式。

对于 VLM，PEFT 的重点不只是减少可训练参数，更在于使更新模块与变化来源匹配：视觉域失配主要影响视觉编码器或视觉 Adapter；新模态或跨模态对齐问题主要影响投影层、连接器和模态 Adapter；任务指令、输出格式或语言推理方式变化则更多涉及 Prompt、任务头与语言侧 LoRA。持续学习场景下，还需通过任务专属 Adapter/LoRA bank、模块路由、代表样本或原型回放、旧版本蒸馏和新旧任务联合评测，降低不同增量模块之间的干扰与灾难性遗忘。

**适用变化：** 持续域失配、新类别、新型任务能力、跨区域适配，以及新载荷/新谱段带来的局部模块失配。

**可更新对象：** 分类/检测/分割任务头、视觉—语言连接器、视觉 Adapter、语言主干 LoRA，或任务专属 Adapter bank。

**相关研究与启示：**
- LoRA 最初面向大语言模型提出，通过冻结基座权重并注入低秩更新矩阵，大幅减少可训练参数和训练显存需求，为大模型选择性微调提供了基础方法 [17]；LLaMA-Adapter V2 进一步将参数高效适配扩展到视觉指令模型，说明视觉 token 融合、跨模态对齐和指令能力可以通过少量新增参数进行针对性调整 [18]。
- TPPT 利用稳定的文本原型引导视觉与文本 Prompt 的增量学习，体现了“以语言语义锚点约束跨模态表征漂移”的轻量持续学习思路 [9]。MoE Adapter 通过为新任务增添专家 Adapter、保留原始 CLIP 并对输入进行自动路由，缓解连续学习中的参数漂移与零样本能力退化；LLaVA-DyMoE 则指出动态 MoE 仍会因路由漂移发生遗忘，并通过 token 级分配约束减少旧任务能力损失 [19,20]。
- 当新增的是传感器或模态而非单一类别时，PathWeave 提出将单模态 Adapter 与跨模态 Adapter 组合，并通过 MoE 门控进行路径选择，在连续新增图像、视频、音频、深度和点云模态时降低训练负担 [21]。这一机制尤其值得用于讨论光学、SAR、红外或高光谱持续接入时的模块化适配。
- CLeaRS 显示，顺序 LoRA 对 RS-VLM 的任务、指令和模态转换仍不足以保证稳定性 [1]。LoRA-Det 在星载遥感有向目标检测中报告，仅更新 12.4% 参数即可达到全量微调约 97%–100% 的性能 [22]。前者说明 VLM 的 PEFT 不能只在语言侧附加 LoRA，后者可作为专用感知模型快速适配的参考，但不能直接外推为生成式 VLM 的训练成本或遗忘表现。

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

**技术内涵：** 星地/星间异步联邦 PEFT 是将联邦学习的“数据留在本地、只交换模型更新”原则与参数高效微调结合起来，使多颗卫星能够基于各自观测数据训练小规模参数增量，并通过星地接触窗口或星间链路共享更新，而无需集中传输原始图像。每个节点保存同一基座模型或相容的模块版本，仅上传 LoRA、Adapter、任务头、原型或记忆索引的差分；聚合节点根据任务、模态、数据质量和版本依赖关系融合这些差分，再将通过验证的增量发布给相应卫星。

“异步”是其区别于传统同步联邦学习的关键：卫星无需在同一时刻完成训练和通信，而是在各自可见窗口、能量预算和任务空闲期独立上报。聚合端需要处理更新时效差、不同卫星数据分布不一致、LoRA rank 不同、模态缺失和版本陈旧等问题，因此不能简单采用同步 FedAvg，而应采用按模块、按维度、按任务簇或按可信度的选择性聚合。对资源较弱的观测星，还可将部分前向/反向计算迁移到地面或算力星，形成 split-federated 训练。

**适用变化：** 多星面对不同地域、传感器、任务或季节数据，既需要共享能力又不适合频繁下传原始数据的场景。

**可同步对象：** LoRA/Adapter 增量、局部任务头、类别原型、漂移摘要、检索索引差分和可信度统计。

**相关研究与启示：**
- Satellite Federated Fine-Tuning 提出卫星—地面协同的基础模型微调框架。论文明确指出，传统卫星联邦学习下卫星算力不足以有效微调大基础模型，因而需要将模型组件在卫星、地面或其他节点之间分配；其仿真结果报告训练时间约缩短 33% [23]。该工作是星上基础模型微调的重要系统参考，但并非 RS-VLM 飞行实证。
- FediLoRA 面向异构 LoRA rank 和缺失模态下的联邦多模态微调，采用逐维聚合与选择性模型编辑，说明星座中不能直接平均不同卫星的 LoRA 因子 [24]。该工作不处于卫星场景，但可作为异构载荷和不同 LoRA rank 聚合的机制参考。
- SFL-LEO 将 split learning 与 federated learning 结合，利用低轨周期性断连期间的本地训练和异步聚合；在 Starlink 带宽轨迹驱动的实验中取得了与常规 split learning 接近的准确率 [25]。
- Sparse Incremental Aggregation in Satellite Federated Learning 表明，在轨内 ISL 上使用稀疏化与逐跳增量聚合可提升带宽效率 [26]。

**面向 VLM 的可行性判断：中等。** 围绕大模型微调、异构多模态 LoRA、卫星通信与异步聚合的研究已分别给出可行路径；但针对真实在轨 RS-VLM 的在线 LoRA 训练与长期安全发布，公开飞行验证仍较少。该路线更适合被视为远期星座级适应方向，而不是近期单星 VLM 参数更新的默认方案。

### 技术五：受控的 VLM 自进化

**技术内涵：** VLM 自进化不是单一算法，而是一种利用持续到来的观测、任务反馈和外部监督信号逐步积累经验、构造增量监督并改进模型的长期适应范式。它将前述记忆、轻量适应、PEFT 与协同更新串联为“反馈获取—经验沉淀—候选更新—验证发布”的闭环：从未标注观测、人工审核、多时相一致性、跨传感器一致性和任务结果中提取候选经验；经质量评估、冲突检测与可信筛选后，优先写入案例记忆和检索索引；仅在数据质量、覆盖度和安全要求满足时，才将部分可信样本用于训练局部 PEFT 候选版本；最后通过新旧任务、资源与风险联合验证决定是否发布。

“受控”意味着模型自身的生成结果不能直接被视为长期训练标签，更不能自动替换稳定版本。其前提是来源可追溯、监督信号可验证、候选更新可回滚、发布前可评测；在星上场景中，案例记忆、错误样本队列和可信度统计通常应先于参数更新发生。因此，自进化的重点是把模型输出转化为经过筛选的经验资产，而不是形成“自生成数据—自主训练—自动部署”的无约束闭环。

**适用变化：** 标签稀缺、观测流持续到来、存在地面反馈或多源一致性信号，希望逐步积累经验的长期运行场景。

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

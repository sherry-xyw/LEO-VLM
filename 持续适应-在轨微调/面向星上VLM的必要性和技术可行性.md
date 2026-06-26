# 在轨持续学习与增量学习：面向星上 VLM 的必要性和技术可行性

> **问题：** 面向低轨卫星及低轨算力星座部署视觉—语言模型（VLM）时，是否有必要在地面投入时间进行再训练？哪些变化可由星上 Prompt、RAG 或轻量适应解决，哪些变化必须通过地面选择性再训练/参数高效微调（PEFT）解决？

## 1. 核心结论

### 1.1 总体判断

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

## 3. 技术可行性

下面的技术路线按更新成本、风险和系统复杂度从低到高排列。每条路线均围绕“技术内涵—代表性工作—面向星上 VLM 的适用性与边界”展开。对于 VLM，应始终区分“可在星上进行的低风险适应”与“应在地面或算力星完成的参数再训练”。

### 技术路线 1：漂移检测 + 轻量自监督/测试时适应

**技术内涵：** 漂移检测负责回答“当前失配是否已超出基座模型的稳定能力边界”，测试时适应（test-time adaptation, TTA）与轻量自监督适应则负责在未标注或弱标注观测流上进行短期校准。前者可通过输入质量、特征分布、视觉—文本相似度、类别原型覆盖度、置信度或任务损失的变化识别；后者通常仅更新归一化统计量、类别原型、温度参数、Prompt 或极小 Adapter，而不改变基座模型。

对于 VLM，域偏移不仅会使视觉特征退化，还会拉大视觉与文本表征之间的模态间隙。因此，轻量适应的关键不是单纯提高当前样本的预测置信度，而是以稳定文本原型、跨模态语义空间或时相一致性作为锚点，限制视觉侧更新的方向。轻量自监督可利用掩码重建、相邻时相一致性、多视角一致性或高置信样本进行表征校准；TTA 则更适合在单个任务窗口内进行可复位的快速调整。

**适用变化：** 成像退化、季节/天气变化、传感器噪声、轻度地域偏移等短期或渐进式域迁移。

**可更新对象：** 归一化统计量、类别原型、置信度温度参数，或视觉端的小型 Adapter/LoRA。

**代表性工作与启示：**
- SubTTA 通过对齐视觉与文本的语义子空间，并将视觉特征投影到任务相关文本子空间后再进行测试时适应，针对性缓解分布偏移引起的模态间隙与视觉噪声 [11]。该工作表明，VLM 的轻量适应必须显式保持跨模态对齐，而不能只沿用纯视觉 TTA 的伪标签自训练。
- Black-CL 面向无法访问模型权重或梯度、且计算受限的黑盒 VLM 设置，使用文本原型、潜在分布回放与测试时原型适应完成增量更新 [12]。其“无主干反传、仅维护低维状态”的思路与星上受限环境具有较强可迁移性。
- CoSMAE 在遥感连续自监督学习中以数据混合和模型混合蒸馏维持视觉表征连续性，为无标注遥感观测流提供了可借鉴的自监督机制 [13]；面向卫星影像退化的在线域适应工作则表明，更新目标域 BatchNorm 统计量和全局类别中心可避免反向传播 [14]。

**面向星上 VLM 的适用性与边界：** 该路线适合恢复已有能力，而非学习新的稳定能力。星上优先采用无梯度的统计量、原型和校准参数更新，并将其作为可复位的任务窗口状态；当漂移持续、跨模态对齐无法恢复，或任务已涉及新类别、新模态和高风险告警时，应升级为局部 PEFT 与地面回归验证，而不应依赖连续伪标签自训练。

### 技术路线 2：案例记忆库 / RAG / 规则库更新

**技术内涵：** 该路线通过更新模型外部的非参数知识，而不是修改模型权重来适应信息变化。其基本过程为：将图像、文本、遥测或历史案例编码为可检索表征；依据当前观测、任务指令和元数据检索候选证据；经重排序、时空过滤和规则校验后，将可信内容注入 VLM 的上下文。案例记忆用于沉淀历史事件与决策经验，RAG 用于从大规模知识库中按需调用相关证据，规则库则为任务边界、格式约束和安全限制提供确定性校验。

与参数更新相比，记忆更新具有增量小、可追溯、可删除和易回滚的特点。其核心技术问题不在于“存更多信息”，而在于建立带来源、时间、地域、传感器、权限、置信度与有效期的记忆条目，并处理证据冲突、过期知识和检索错误。

**适用变化：** 区域背景、任务规则、历史事件、故障案例、目标属性、地面审核反馈等知识变化。

**可更新对象：** 向量索引、案例摘要、规则文件、目标原型、检索缓存和可信度元数据。

**代表性工作与启示：**
- RS-RAG 将遥感影像与文本知识编码至统一向量空间，并通过检索、重排序和知识增强提示改善图像描述、分类与 VQA，说明多模态检索可为遥感 VLM 提供可扩展的外部知识接口 [10]。
- Self-RAG 将检索、生成与自我批判耦合，使模型能够按需检索并评价证据与生成结果的质量 [15]；其价值在于提示星上系统不应将“是否检索”视为固定流程，而应使证据需求与任务不确定性联动。
- Reflexion 以语言化 episodic memory 保存环境反馈并用于后续决策，说明经验积累可首先在非参数记忆层发生，而不必频繁修改模型权重 [16]。

**面向星上 VLM 的适用性与边界：** 该路线是优先级最高的长期更新机制。记忆条目应同时记录时空与传感器元数据，并通过版本、签名、可信度和有效期进行管理；多星场景可优先同步案例摘要、原型和索引差分，而非原始数据。其边界在于：RAG 解决的是“模型可访问什么知识”，不能修复视觉编码器失配、连接器偏移或新类别的细粒度视觉判别不足；对安全关键控制任务，检索结果还必须经独立规则模块校验，不能直接等同于执行指令。

### 技术路线 3：局部任务头、Adapter 与 LoRA 更新

**技术内涵：** 参数高效微调通过冻结基座模型，仅训练少量新增或低秩参数来恢复或扩展能力。LoRA 将权重更新表示为低秩增量；Adapter 在网络层间引入小型瓶颈分支；Prompt/Prefix tuning 通过少量可学习提示改变任务条件；任务头则针对检测、分类、分割或结构化生成输出进行局部重构。对 VLM 而言，视觉编码器、跨模态连接器、语言骨干和任务头承担不同功能，PEFT 的关键不只是降低训练参数量，而是使更新位置与变化来源相匹配。

持续适应中的主要风险是灾难性遗忘。因而，PEFT 还应与代表样本/原型回放、旧模型或旧 Adapter 蒸馏、参数隔离、Adapter bank 与路由机制结合：新任务尽量使用独立增量承载，旧任务通过回放和蒸馏保持，运行时再由任务、地域或模态路由选择合适的增量模块。

**适用变化：** 持续域失配、新类别、新型任务能力、跨区域适配，以及新载荷/新谱段带来的局部模块失配。

**可更新对象：** 分类/检测/分割任务头、视觉—语言连接器、视觉 Adapter、语言主干 LoRA，或任务专属 Adapter bank。

**代表性工作与启示：**
- LoRA 通过冻结基座权重并学习低秩更新矩阵，显著降低了大模型微调的可训练参数与优化器状态开销 [17]；LLaMA-Adapter V2 进一步表明，视觉 token 融合和视觉指令能力可由少量新增 Adapter 完成适配 [6]。
- TPPT 利用稳定文本原型约束视觉与文本 Prompt 的增量学习，为跨模态表征漂移提供了“语言语义锚点” [18]。
- MoE Adapter 通过为新任务增添专家 Adapter、保留原始 CLIP 并自动路由输入来降低参数干扰；LLaVA-DyMoE 进一步指出，动态 MoE 若发生 token 路由漂移，仍会损伤旧任务能力，需要对路由进行约束 [19,20]。
- PathWeave 将单模态 Adapter 与跨模态 Adapter 结合，并通过门控选择路径，为连续接入图像、视频、音频、深度和点云等模态提供了模块化适配思路 [21]。该机制可用于讨论光学、SAR、红外或高光谱持续接入时的连接器与模态 Adapter 设计。
- CLeaRS 显示，顺序 LoRA 对 RS-VLM 的任务、指令和模态转换仍不足以稳定保持旧能力 [1]；LoRA-Det 在星载遥感有向目标检测中展示了局部低秩适配的资源优势，但其结果不能直接外推至生成式 VLM [22]。

**面向星上 VLM 的适用性与边界：**
- **在星上直接训练：中等。** 即使仅训练 LoRA，仍需承担高分辨率视觉 token 的反向传播激活、优化器状态、回放缓存和验证成本；因此不能将 VLM LoRA 等同于专用检测器的轻量上注。
- **在地面或算力星开展选择性 PEFT、再上注增量：高。** 更适合完成难例筛选、历史任务回放、跨模态消融、量化回归和幻觉/拒答评测。
- **模块选择原则：** 视觉域变化优先适配视觉端；跨模态失配或新模态优先适配连接器/投影层和模态 Adapter；指令与输出变化优先适配 Prompt、任务头或语言侧 LoRA。高风险目标检测与定位任务仍宜采用“专用感知模型负责结构化判别，VLM 负责解释与问答”的组合，而不由 VLM 独立承担最终告警判定。

### 技术路线 4：星地/星间异步联邦 PEFT

**技术内涵：** 联邦 PEFT 旨在使多颗卫星利用本地数据训练小规模参数增量，并在不集中下传原始数据的条件下完成共享。其基本流程是：卫星在本地执行轻量更新或生成增量摘要；在星地接触窗口或可用 ISL 上上传 LoRA/Adapter、任务头、原型或记忆差分；聚合节点完成异构更新融合、历史能力验证与版本发布。异步机制允许不同卫星按各自可见窗口和资源状态参与，而不要求所有节点同步到达。

在星座中，联邦学习还需与模型切分、轨道窗口和通信调度耦合。对资源较弱的观测星，可将部分前向/反向计算迁移至地面或算力星，形成 split-federated 训练；对异构 LoRA rank、缺失模态或任务差异，则需按参数维度、模块或能力簇进行选择性聚合，而不能直接平均所有低秩因子。

**适用变化：** 多星面对不同地域、传感器、任务或季节数据，既需要共享能力又不适合频繁下传原始数据的场景。

**可同步对象：** LoRA/Adapter 增量、局部任务头、类别原型、漂移摘要、检索索引差分和可信度统计。

**代表性工作与启示：**
- Satellite Federated Fine-Tuning 提出卫星—地面协同的基础模型微调框架，通过分解并分配模型组件缓解单星计算不足，并针对间歇星地通信和不稳定 ISL 设计通信策略 [23]。该工作说明大模型在轨微调的可行性取决于计算—通信协同，而非仅取决于参数量是否减少。
- FediLoRA 面向异构 LoRA rank 和缺失模态下的联邦多模态微调，采用逐维聚合与选择性模型编辑，说明星座中不能直接平均不同卫星的 LoRA 因子 [24]。
- SFL-LEO 将 split learning 与 federated learning 结合，利用低轨周期性断连期间进行本地训练、在接触窗口完成异步聚合，为利用可预测轨道窗口安排训练与同步提供了参考 [25]。
- Sparse Incremental Aggregation in Satellite Federated Learning 以稀疏化和逐跳增量聚合提升在轨内 ISL 上的带宽利用效率，为更新差分的分块传输和逐跳融合提供了通信侧思路 [26]。

**面向星上 VLM 的适用性与边界：** 该路线具有中等可行性，更适合作为星座级中远期适应能力。近期可采用“观测星发现漂移并生成轻量增量—地面/算力星完成 PEFT 与回归验证—再向卫星灰度下发”的分工。应优先同步经验证的参数差分和高频状态摘要，并以窗口、带宽、任务时限和版本依赖关系调度传输；针对真实在轨 RS-VLM 的在线 LoRA 训练、异构聚合和长期安全发布，仍缺少直接飞行证据。

### 技术路线 5：受控的 VLM 自进化

**技术内涵：** VLM 自进化是指系统利用未标注观测、任务反馈、地面审核或多源一致性信号，持续产生可用于改进模型的经验。其合理实现不应是“自生成数据—自主训练—自动替换模型”的闭环，而应包括：可信反馈筛选、案例记忆沉淀、错误归因、候选训练数据构建、局部 PEFT 候选生成，以及新旧能力与资源约束下的验证发布。换言之，自进化的核心是对经验和监督信号进行受控积累，而不是让模型无约束地以自身输出作为长期训练标签。

**适用变化：** 标签稀缺、观测流持续到来、存在地面反馈或多源一致性信号，希望逐步积累经验的长期运行场景。

**可接受的更新对象：** 优先为案例记忆、检索索引、错误样本队列和可信度统计；只有在监督信号经过筛选后，才形成局部 LoRA/Adapter 的候选增量。

**代表性工作与启示：**
- RISE 通过“提问者—求解者”闭环从未标注图像中提升 VLM 能力，并引入问题质量监督和技能分布平衡以缓解伪标签退化与模式塌缩 [27]。该工作说明无标注图像可成为能力改进的来源，但也表明自进化依赖多轮采样、质量控制与再训练。
- Self-RAG 的检索—生成—自检机制与 Reflexion 的记忆驱动反思机制，分别提供了可验证知识调用和经验沉淀的非参数路径 [15,16]。二者更适合被理解为受控自进化的前置环节，而不是自动参数更新的充分条件。

**面向星上 VLM 的适用性与边界：** 近期更适合研究“记忆驱动、反馈筛选、候选上注、可验证发布”的弱自进化：
1. 从地面确认、多时相一致性、跨传感器一致性和高置信预测中提取候选经验；
2. 优先写入带来源和有效期的记忆库，并保留错误案例用于后续归因；
3. 仅将通过可信筛选的样本用于地面或算力星的局部 PEFT 候选训练；
4. 候选版本需通过新旧任务联合评测、资源检查和灰度对照后，才允许上注或启用。

因此，不宜让单星在缺少外部监督、回放集和回滚机制时执行无约束闭环自训练或自动替换稳定版本。

---

## 4. 从变化类型到 VLM 更新路径的选择规则

| 变化来源 | 首选路径 | 是否需要地面 PEFT | 何时升级 | 不应优先采用 |
| --- | --- | ---: | --- | --- |
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

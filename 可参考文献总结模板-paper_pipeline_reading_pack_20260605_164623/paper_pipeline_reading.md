# S级论文模型图与Pipeline精读

说明：本文档基于本地已下载 PDF 和文献调研总表整理。图片为论文候选模型图页/主实验表页整页截图，便于保留图题和上下文；如需汇报 PPT，可后续再精裁 Figure 或 Table。

## 总览

| 序号 | 论文 | 会议/年份 | 任务 | 方法 | 预训练模型/基础模型 | 本项目用途 |
|---|---|---|---|---|---|---|
| 1 | An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction | AAAI 2024 | Joint IE | Generative IE | BART/T5 类生成式预训练语言模型（具体 backbone 待精读确认） | joint entity-relation extraction；text-to-structure输出；图结构输出 |
| 2 | CROSSAGENTIE: Cross-Type and Cross-Task Multi-Agent LLM Collaboration for Zero-Shot Information Extraction | Findings of ACL 2025 | Joint IE | LLM-based IE | GPT-4/ChatGPT 类闭源 LLM（多智能体协作；具体版本待精读确认） | zero-shot抽取；prompt设计；joint entity-relation extraction；跨任务一致性校验 |
| 3 | GUIDEX: Guided Synthetic Data Generation for Zero-Shot Information Extraction | Findings of ACL 2025 | General IE | Generative IE; LLM-based IE | GPT-4/ChatGPT 类闭源 LLM 用于合成数据；下游抽取模型待精读确认 | schema设计；synthetic data生成；zero-shot抽取；标注指南设计 |
| 4 | InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction | arXiv 2023 | General IE | Generative IE; LLM-based IE | T5/Flan-T5 类 instruction-tuned text-to-text backbone | prompt设计；schema设计；text-to-structure输出；开源baseline |
| 5 | UIE: Unified Structure Generation for Universal Information Extraction | ACL 2022 | General IE | Generative IE | T5-base/T5-large text-to-text 预训练模型 | schema设计；text-to-structure输出；生成式IE baseline |
| 6 | A Survey of Generative Information Extraction | Findings of ACL 2024 | General IE | Generative IE; LLM-based IE | 不适用（综述） | 相关工作框架；评估指标设计；方法分类 |
| 7 | M-BRe: Discovering Training Samples for Relation Extraction from Unlabeled Texts with Large Language Models | EMNLP 2025 | RE | LLM-based IE | GPT-4/ChatGPT 类 LLM 用于无标注样本发现；RE backbone 待精读确认 | few-shot训练；关系样本自动标注；text-to-triplet输出 |
| 8 | REBEL: Relation Extraction By End-to-end Language generation | Findings of EMNLP 2021 | RE | Generative IE | BART-large | text-to-triplet输出；生成式RE baseline；开源baseline |
| 9 | DEGREE: A Data-Efficient Generation-Based Event Extraction Model | NAACL 2022 | EE | Generative IE | BART-base/BART-large 生成式预训练模型 | text-to-structure输出；few-shot训练；事件schema设计 |
| 10 | GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer | NAACL 2024 | NER | Traditional IE | DeBERTa-v3 / BERT 类双向 Transformer encoder | label description设计；zero-shot抽取；开源baseline |
| 11 | GLiREL - Generalist Model for Zero-Shot Relation Extraction | NAACL 2025 | RE | Traditional IE | DeBERTa-v3 / GLiNER-style encoder（具体 backbone 待精读确认） | label description设计；zero-shot抽取；开源baseline |
| 12 | GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models | NAACL 2024 | RE | Generative IE; LLM-based IE | 不适用或 LLM-as-evaluator/生成式 RE 评估；具体模型随实验设置 | 评估指标设计；生成式RE评测协议 |
| 13 | GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction | ICLR 2024 | General IE | Generative IE; LLM-based IE | CodeLLaMA/LLaMA 系列指令模型（GoLLIE checkpoints） | 标注指南设计；zero-shot抽取；schema设计；开源baseline |
| 14 | UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition | ICLR 2024 | NER | LLM-based IE | LLaMA/Alpaca 系列蒸馏小模型；ChatGPT 作为教师 | 可作为太空 QB 报文低资源/开放标签实体抽取 baseline；标签描述和蒸馏数据构建方式可迁移。 |

## 1. An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction

- 会议/期刊：AAAI 2024；类型：Top Conference。
- 任务/方法：Joint IE；Generative IE。
- 预训练模型/基础模型：BART/T5 类生成式预训练语言模型，具体 backbone 仍需依据 PDF 精读确认。
- 训练方式：在带实体与关系标注的数据集上监督微调，目标序列是文本对应的图结构表示。
- 数据构造：使用 CoNLL04、ACE05、SciERC 等联合实体关系抽取数据集；样本由文本和实体/关系标注构造成 text-to-graph 目标。
- 输入方式：输入为原始文本；输出为实体节点、类型和关系边构成的图。
- Pipeline 简要说明：把实体关系联合抽取建模为从文本自回归生成图结构：输入句子/文档，模型按预定义图表示生成实体节点、实体类型和关系边，最后解析为结构化三元组或图。
- 主实验表格与指标说明：主实验报告 Entity-F1、Relation-F1/Triplet-F1，用于比较联合抽取性能。PDF 未下载，具体表格待人工补。
- 与太空目标实体关系抽取的关系：joint entity-relation extraction；text-to-structure输出；图结构输出

模型图/方法示意页：未生成。原因：PDF 未下载或论文无明确模型图页。

主实验表格页：未生成。原因：PDF 未下载或综述/评估论文无传统主实验表。

## 2. CROSSAGENTIE: Cross-Type and Cross-Task Multi-Agent LLM Collaboration for Zero-Shot Information Extraction

- 会议/期刊：Findings of ACL 2025；类型：Findings。
- 任务/方法：Joint IE；LLM-based IE。
- 预训练模型/基础模型：GPT-3.5-Turbo 为主要 backbone，并额外测试 GPT-4o；框架基于 Microsoft AutoGen。
- 训练方式：主设置是 zero-shot prompting；另外评估 template fine-tuning，将 LLM 输出蒸馏成模板化训练数据。
- 数据构造：NER：CoNLL03、CoNLL04、OntoNotes4、SemEval、TACRED；RE：CoNLL04、SemEval、TACRED、NYT、SciERC。
- 输入方式：输入为文本和实体/关系标签集合；输出为严格 span-level matching 的实体或关系结果。
- Pipeline 简要说明：多智能体零样本 IE：每个实体类型或关系类型对应一个 LLM agent，先分类型抽取，再通过跨类型/跨任务协作检查遗漏、冲突和边界，最后汇总为实体列表或关系三元组。
- 主实验表格与指标说明：主实验用 micro-F1。NER 上相比 AEiO 平均提升 8.56 F1，相比 Type-Agents 提升 7.63 F1；RE 上相比 Direct-Prompting 平均提升 9.13 F1，相比 Type-Agents 提升 5.54 F1。
- 与太空目标实体关系抽取的关系：zero-shot抽取；prompt设计；joint entity-relation extraction；跨任务一致性校验

模型图/方法示意页：
![CROSSAGENTIE: Cross-Type and Cross-Task Multi-Agent LLM Collaboration for Zero-Shot Information Extraction 模型图](<论文图片摘录/02_CROSSAGENTIE_model_p4-04.png>)

主实验表格页：
![CROSSAGENTIE: Cross-Type and Cross-Task Multi-Agent LLM Collaboration for Zero-Shot Information Extraction 主实验表](<论文图片摘录/02_CROSSAGENTIE_table_re_p7-07.png>)

## 3. GUIDEX: Guided Synthetic Data Generation for Zero-Shot Information Extraction

- 会议/期刊：Findings of ACL 2025；类型：Findings。
- 任务/方法：General IE；Generative IE; LLM-based IE。
- 预训练模型/基础模型：使用 LLM 进行 guideline 与数据生成，具体闭源模型版本需精读确认；下游可训练 IE 模型。
- 训练方式：不是传统监督模型训练论文，核心是合成数据构造；生成的数据可用于 zero-shot/few-shot IE 训练或评估。
- 数据构造：从 FineWeb-Edu 随机采样约 10,000 篇教育文档，构造 GUIDEX 数据集；包含 28,677 个 unique labels，平均每篇 5.34 个不同标签、11.39 条 annotation。
- 输入方式：输入为长文档；输出为 guideline、schema/dataclass，以及符合 schema 的实体/属性实例。
- Pipeline 简要说明：先让 LLM 从文档生成 annotation guideline/schema，再根据 guideline 生成 Python dataclass 形式的结构化样本，最后执行代码级一致性检查过滤 hallucination 和不合法标注。
- 主实验表格与指标说明：主实验关注 label coverage 与合成数据有效性；已知对 35 个 IE 数据集标签空间覆盖约 42% 左右，部分 NER 数据集如 CoNLL03/BroadTwitter/HarveyNER/BC5CDR 达到 100% type overlap。
- 与太空目标实体关系抽取的关系：schema设计；synthetic data生成；zero-shot抽取；标注指南设计

模型图/方法示意页：
![GUIDEX: Guided Synthetic Data Generation for Zero-Shot Information Extraction 模型图](<论文图片摘录/03_GUIDEX_model_p3-03.png>)

主实验表格页：
![GUIDEX: Guided Synthetic Data Generation for Zero-Shot Information Extraction 主实验表](<论文图片摘录/03_GUIDEX_table_p7-07.png>)

## 4. InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction

- 会议/期刊：arXiv 2023；类型：Preprint。
- 任务/方法：General IE；Generative IE; LLM-based IE。
- 预训练模型/基础模型：基于 Flan-T5/Instruction-tuned T5 类 text-to-text 模型；原表中 ACL venue 已修正为 arXiv:2304.08085。
- 训练方式：多任务 instruction tuning：把不同 IE 数据集转换为自然语言指令 + 输入文本 + 结构化输出，进行监督微调。
- 数据构造：覆盖 NER、RE、EE、EAE 等多类 IE 数据集，论文特别比较 ChatGPT、Tk-Instruct、UIE 等。
- 输入方式：输入为 instruction + schema/任务描述 + 原文；输出为实体、关系或事件结构。
- Pipeline 简要说明：把不同 IE 任务统一为 instruction following：任务指令描述抽取目标，模型读取文本并生成结构化答案，实现 NER、RE、EE 等多任务统一抽取。
- 主实验表格与指标说明：摘要报告 gpt-3.5-turbo 在零样本 IE 上约 20.47 F1，而 InstructUIE 在零样本上达到约 38.69 F1；主表页截图中包含具体数据集分项。
- 与太空目标实体关系抽取的关系：prompt设计；schema设计；text-to-structure输出；开源baseline

模型图/方法示意页：
![InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction 模型图](<论文图片摘录/04_InstructUIE_model_p2-02.png>)

主实验表格页：
![InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction 主实验表](<论文图片摘录/04_InstructUIE_table_p6-06.png>)

## 5. UIE: Unified Structure Generation for Universal Information Extraction

- 会议/期刊：ACL 2022；类型：Top Conference。
- 任务/方法：General IE；Generative IE。
- 预训练模型/基础模型：T5-v1.1-base / T5-v1.1-large 初始化，继续进行 text-to-structure pre-training。
- 训练方式：先在大规模结构化 IE 任务上预训练，包括 pair、record 和 text span corruption 目标；再在具体 IE 数据集上微调。
- 数据构造：13 个 IE benchmark，覆盖实体、关系、事件和情感任务；包括 ACE05、CoNLL04、NYT、ADE、CASIE 等。
- 输入方式：输入为 schema prompt + text；输出为 SEL 结构化记录，例如实体 spot、关系 association 和 span。
- Pipeline 简要说明：通过 Structural Schema Instructor 指定 spot/association schema，将 NER、RE、EE、情感抽取等任务统一成结构生成语言 SEL，T5 根据文本生成统一结构。
- 主实验表格与指标说明：主实验报告不同任务的 F1，包括 Entity-F1、Relation-F1、Event-F1 等；统一结构生成在多任务上达到或接近 SOTA。
- 与太空目标实体关系抽取的关系：schema设计；text-to-structure输出；生成式IE baseline

模型图/方法示意页：
![UIE: Unified Structure Generation for Universal Information Extraction 模型图](<论文图片摘录/05_UIE_model_p4-04.png>)

主实验表格页：
![UIE: Unified Structure Generation for Universal Information Extraction 主实验表](<论文图片摘录/05_UIE_table_p7-07.png>)

## 6. A Survey of Generative Information Extraction

- 会议/期刊：Findings of ACL 2024；类型：Findings。
- 任务/方法：General IE；Generative IE; LLM-based IE。
- 预训练模型/基础模型：不适用；综述覆盖 T5、BART、GPT/LLM 等生成式 backbone。
- 训练方式：总结 supervised fine-tuning、prompting、instruction tuning、schema-guided generation 等训练范式。
- 数据构造：综述覆盖 NER、RE、EE、Joint IE、Open IE 等任务及常用基准。
- 输入方式：按任务可为文本、schema、prompt、instruction；输出为实体列表、关系三元组、事件结构或图。
- Pipeline 简要说明：综述，不提出单一 pipeline；核心框架是把 IE 任务从分类/序列标注转为文本序列、结构、三元组、事件或图的生成。
- 主实验表格与指标说明：无主实验表；适合作为论文综述部分的总框架来源。
- 与太空目标实体关系抽取的关系：相关工作框架；评估指标设计；方法分类

模型图/方法示意页：未生成。原因：本地 `Findings of ACL/2024_A Survey of Generative Information Extraction.pdf` 首页标题为 “Real World Conversational Entity Linking Requires More Than Zero-Shots”，与该综述不匹配，已移除错误截图。

主实验表格页：未生成。原因：综述论文无传统主实验表；且当前本地 PDF 源文件不匹配。

## 7. M-BRe: Discovering Training Samples for Relation Extraction from Unlabeled Texts with Large Language Models

- 会议/期刊：EMNLP 2025；类型：Top Conference。
- 任务/方法：RE；LLM-based IE。
- 预训练模型/基础模型：使用 LLM 发现样本；下游 RE 模型 backbone 待精读确认。
- 训练方式：弱监督/自训练式流程：无标注文本 -> LLM 生成候选关系样本 -> 过滤 -> 训练 RE 模型。
- 数据构造：用于 relation extraction 的无标注文本和标准 RE 测试集；表中主实验页包含具体数据集。
- 输入方式：输入为无标注句子/文档和关系标签；输出为训练样本或关系三元组。
- Pipeline 简要说明：面向无标注文本的 RE 样本发现：LLM 从无标注文本中发现可训练关系样本，经筛选/合并后构造训练集，再训练关系抽取模型。
- 主实验表格与指标说明：主实验报告 F1，重点验证自动发现训练样本能提升低资源 RE。具体逐项数值见截图表页。
- 与太空目标实体关系抽取的关系：few-shot训练；关系样本自动标注；text-to-triplet输出

模型图/方法示意页：
![M-BRe: Discovering Training Samples for Relation Extraction from Unlabeled Texts with Large Language Models 模型图](<论文图片摘录/07_M-BRe_model_p3-03.png>)

主实验表格页：
![M-BRe: Discovering Training Samples for Relation Extraction from Unlabeled Texts with Large Language Models 主实验表](<论文图片摘录/07_M-BRe_table_p5-05.png>)

## 8. REBEL: Relation Extraction By End-to-end Language generation

- 会议/期刊：Findings of EMNLP 2021；类型：Findings。
- 任务/方法：RE；Generative IE。
- 预训练模型/基础模型：BART-large。
- 训练方式：在关系三元组数据上监督微调；目标是线性化三元组文本序列。
- 数据构造：主要使用 WebNLG、NYT 等关系抽取/三元组生成数据集，并评估零样本/迁移能力。
- 输入方式：输入为句子或文本片段；输出为一个或多个关系三元组。
- Pipeline 简要说明：把关系抽取转为端到端序列生成：输入文本，BART 直接生成包含 subject、relation、object 的线性化三元组序列。
- 主实验表格与指标说明：主实验报告 triplet-level precision/recall/F1；生成式 RE 在多个三元组抽取基准上优于 pipeline 或分类式基线。
- 与太空目标实体关系抽取的关系：text-to-triplet输出；生成式RE baseline；开源baseline

模型图/方法示意页：
![REBEL: Relation Extraction By End-to-end Language generation 模型图](<论文图片摘录/08_REBEL_model_p4-04.png>)

主实验表格页：
![REBEL: Relation Extraction By End-to-end Language generation 主实验表](<论文图片摘录/08_REBEL_table_p7-07.png>)

## 9. DEGREE: A Data-Efficient Generation-Based Event Extraction Model

- 会议/期刊：NAACL 2022；类型：Top Conference。
- 任务/方法：EE；Generative IE。
- 预训练模型/基础模型：BART-base/BART-large。
- 训练方式：少样本友好的监督微调；用事件类型模板和论元角色模板构造输入输出。
- 数据构造：ACE05-E、ACE05-E+、ERE 等事件抽取数据集。
- 输入方式：输入为文本 + 事件类型/模板 prompt；输出为触发词和事件论元。
- Pipeline 简要说明：事件抽取生成式方法：给定事件模板/trigger prompt，模型生成事件触发词和论元填槽结果，将 EE/EAE 统一为文本生成。
- 主实验表格与指标说明：主实验报告 Trigger-F1、Argument-F1/Role-F1，尤其强调低资源设置下的数据效率。
- 与太空目标实体关系抽取的关系：text-to-structure输出；few-shot训练；事件schema设计

模型图/方法示意页：
![DEGREE: A Data-Efficient Generation-Based Event Extraction Model 模型图](<论文图片摘录/09_DEGREE_model_p2-02.png>)

主实验表格页：
![DEGREE: A Data-Efficient Generation-Based Event Extraction Model 主实验表](<论文图片摘录/09_DEGREE_table_p6-06.png>)

## 10. GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer

- 会议/期刊：NAACL 2024；类型：Top Conference。
- 任务/方法：NER；Traditional IE。
- 预训练模型/基础模型：DeBERTa-v3 / BERT 类双向 Transformer encoder。
- 训练方式：在多源 NER 数据上训练 span-label matching 模型，使模型学习标签语义而非固定分类头。
- 数据构造：训练集整合多种 NER 数据；评估包括 CrossNER、MIT Movie、MIT Restaurant、CoNLL03 等。
- 输入方式：输入为文本 + 实体类型标签或描述；输出为每个类型对应的实体 span。
- Pipeline 简要说明：开放类别 NER：把标签名称/描述和文本一起编码，模型对文本 span 与标签表示进行匹配，实现任意实体类型识别。
- 主实验表格与指标说明：主实验报告 Entity-F1；开放类别/zero-shot NER 上明显强于传统闭集 NER 和部分 LLM prompting。
- 与太空目标实体关系抽取的关系：label description设计；zero-shot抽取；开源baseline

模型图/方法示意页：
![GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer 模型图](<论文图片摘录/10_GLiNER_model_p2-02.png>)

主实验表格页：
![GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer 主实验表](<论文图片摘录/10_GLiNER_table_p4-04.png>)

## 11. GLiREL - Generalist Model for Zero-Shot Relation Extraction

- 会议/期刊：NAACL 2025；类型：Top Conference。
- 任务/方法：RE；Traditional IE。
- 预训练模型/基础模型：DeBERTa-v3 / GLiNER-style encoder，部分实验使用 synthetic pretraining。
- 训练方式：先用合成或多源关系样本训练关系标签语义匹配，再在 zero-shot/unseen relation 设置评估。
- 数据构造：Wiki-ZSL、FewRel、Re-DocRED 等；截图表页显示 Wiki-ZSL/FewRel 和不同 unseen relation 数量 m。
- 输入方式：输入为文本、实体对、关系标签集合/描述；输出为实体对对应关系。
- Pipeline 简要说明：零样本关系抽取：给定文本、候选实体对和关系标签描述，模型编码实体对与关系表示并做匹配/细化，输出关系类别。
- 主实验表格与指标说明：主实验报告 macro precision/recall/F1 或 Accuracy；GLiREL + synthetic pretraining 在 Wiki-ZSL、FewRel 多个 unseen relation 设置下取得最佳或接近最佳结果。
- 与太空目标实体关系抽取的关系：label description设计；zero-shot抽取；开源baseline

模型图/方法示意页：
![GLiREL - Generalist Model for Zero-Shot Relation Extraction 模型图](<论文图片摘录/11_GLiREL_model_p2-02.png>)

主实验表格页：
![GLiREL - Generalist Model for Zero-Shot Relation Extraction 主实验表](<论文图片摘录/11_GLiREL_table_p5-05.png>)

## 12. GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models

- 会议/期刊：NAACL 2024；类型：Top Conference。
- 任务/方法：RE；Generative IE; LLM-based IE。
- 预训练模型/基础模型：评估中使用 text-embedding-ada-002 计算三元组嵌入，也比较多种 LLM/生成式 RE 输出。
- 训练方式：主要是评估协议设计，无模型训练；部分指标使用 LDA/embedding/LLM 判断。
- 数据构造：CDR、DocRED、NYT10m、Wiki20m、TACRED、Wiki80，覆盖文档级、bag-level 和句子级 RE。
- 输入方式：输入为源文本和模型生成的关系三元组；输出为多维评估分数。
- Pipeline 简要说明：不是抽取模型，而是生成式 RE 评估框架：对生成三元组从 topical similarity、uniqueness、granularity、completeness 等维度评分。
- 主实验表格与指标说明：主实验不是传统 F1 单表，而是评估不同生成式 RE 方法在完整性、粒度、事实一致性等方面的差异。
- 与太空目标实体关系抽取的关系：评估指标设计；生成式RE评测协议

模型图/方法示意页：
![GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models 模型图](<论文图片摘录/12_GenRES_model_p3-03.png>)

主实验表格页：
![GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models 主实验表](<论文图片摘录/12_GenRES_table_p5-05.png>)

## 13. GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction

- 会议/期刊：ICLR 2024；类型：Top Conference。
- 任务/方法：General IE；Generative IE; LLM-based IE。
- 预训练模型/基础模型：Code-LLaMA 7B 为主，另训练/分析 13B 和 34B；选择 Code-LLaMA 是因为输入输出用代码表示。
- 训练方式：基于 guideline 的 instruction fine-tuning；训练时使用多任务 IE 数据和对应标注指南，推理时给新任务 guideline 做 zero-shot 抽取。
- 数据构造：训练/评估覆盖 ACE05、BC5CDR、CoNLL03、DIANN、NCBIDisease、OntoNotes5、RAMS、TACRED、WNUT2017、CrossNER、CASIE、MIT Movie/Restaurant、MultiNERD、WikiEvents 等。
- 输入方式：输入为文本 + annotation guideline/code schema；输出为符合 dataclass 的实体、事件、论元或槽位实例。
- Pipeline 简要说明：用 annotation guidelines 作为代码化输入，让模型按 Python dataclass/类型定义输出 IE 结构；重点提升 zero-shot IE。
- 主实验表格与指标说明：主实验覆盖 NER、EE、EAE 等任务，报告 F1；论文明确表 1 给出训练/评估数据集范围，模型图/表页见截图。
- 与太空目标实体关系抽取的关系：标注指南设计；zero-shot抽取；schema设计；开源baseline

模型图/方法示意页：
![GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction 模型图](<论文图片摘录/13_GoLLIE_model_p3-03.png>)

主实验表格页：
![GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction 主实验表](<论文图片摘录/13_GoLLIE_table_p7-07.png>)

## 14. UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition

- 会议/期刊：ICLR 2024；类型：Top Conference。
- 任务/方法：NER；LLM-based IE。
- 预训练模型/基础模型：ChatGPT 作为教师；学生模型基于 LLaMA/Alpaca/Vicuna 系列 instruction-tuned LLM。
- 训练方式：mission-focused instruction tuning/targeted distillation；用 ChatGPT 构造 NER 指令样本训练学生模型。
- 数据构造：评估构建 43 个 NER 数据集、9 个领域的大规模 benchmark，覆盖生物医学、编程、社媒、法律、金融等。
- 输入方式：输入为文本和目标实体类型指令；输出为指定类型的实体列表。
- Pipeline 简要说明：目标蒸馏开放 NER：用 ChatGPT 为大量实体类型和文本生成 NER 指令数据，训练较小 UniversalNER 模型识别任意实体类型。
- 主实验表格与指标说明：摘要报告 UniversalNER 相比 Alpaca/Vicuna 平均提升超过 30 个 F1 点，并平均超过 ChatGPT 7-9 个 F1 点；还超过使用监督 NER 样例的 InstructUIE。
- 与太空目标实体关系抽取的关系：可作为太空 QB 报文低资源/开放标签实体抽取 baseline；标签描述和蒸馏数据构建方式可迁移。

模型图/方法示意页：
![UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition 模型图](<论文图片摘录/14_UniversalNER_model_p3-03.png>)

主实验表格页：
![UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition 主实验表](<论文图片摘录/14_UniversalNER_table_p7-07.png>)

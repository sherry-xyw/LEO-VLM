# 持续适应论文图片摘录

> **用途**：本目录用于归档 `持续适应-核心文献.md` 中可直接支撑综述正文的原始方法图、系统图和关键实验表。当前先完成“图表来源—用途—摘录边界”登记；只有从原始论文、正式出版版本或作者公开版核对后，才保存 PNG/PDF 截图。

## 已确定的优先图表

| 编号 | 来源 | 优先摘录内容 | 用于正文的哪一段 | 摘录边界 |
| --- | --- | --- | --- | --- |
| 01 | CLeaRS | Fig. 1：10 个子集与任务/模态/应用三维组织；Table 2：顺序 LoRA 微调下的 Final、MAA、BWT | 变化建模、任务/模态增量、遗忘评测 | 不截取数据样例中的无关视觉素材；表格只保留连续学习结果和指标说明 |
| 02 | LoRA-Det | LoRA 注入框架、rank 与可训练参数比例/性能比较 | 单星 LoRA 与上行参数增量 | 重点标清检测器而非 VLM；图注要注明“星载应用设定，非在轨训练” |
| 03 | CoSMAE | data mixup + model mixup distillation 框架与性能主表 | 无标签漂移期的表征更新、遗忘控制 | 标注其对象为 MAE 视觉表征，不是多模态生成模型 |
| 04 | RS-RAG | 多模态知识向量库、检索重排序、knowledge-augmented prompt 流程 | RAG 优先更新与局部记忆组织 | 避免把离线知识库流程表述为星上实时索引系统 |
| 05 | FedMosaic | 参数化文档 adapter、document-specific mask、选择性 adapter 聚合流程 | 多星记忆合并、参数化知识共享 | 标明是语言 FedRAG；遥感空间关系与载荷元数据需另行适配 |
| 06 | FEDMEGA | 轨道内 ring all-reduce 与星地全局聚合架构 | 轨内 adapter/LoRA 聚合 | 标明原始通信对象为通用模型，而非低秩 adapter |
| 07 | AsyncFLEO | Fig. 2：异步训练/聚合时序；Fig. 3：ring-of-stars 拓扑 | 异步窗口、卫星分组与陈旧度折扣 | 不引用其绝对性能数值到 VLM；只提取拓扑与更新逻辑 |
| 08 | GLoRA | 共识更新子空间、共享参考坐标和 heterogeneous-rank readout 图 | 反对 LoRA A/B 因子直接 FedAvg | 标明证据来自 NLP 联邦实验，需在视觉/连接器层验证 |
| 09 | CLARE | adapter 的按需扩展、无任务标识路由流程 | adapter 池、任务未知时的路由与遗忘控制 | 标明对象为 VLA；不作为遥感飞行实证 |
| 10 | FedDAA | 动态 client clustering、real/virtual/label drift 区分流程 | 漂移分类、轨道级聚类触发 | 标明为通用联邦学习；遥感漂移信号需另行定义 |

## 建议命名规则

```text
01_CLeaRS_benchmark_protocol.png
01_CLeaRS_forgetting_table.png
02_LoRADet_framework.png
03_CoSMAE_mixup_distillation.png
04_RSRAG_pipeline.png
05_FedMosaic_selective_aggregation.png
06_FEDMEGA_intra_orbit_aggregation.png
07_AsyncFLEO_sequence.png
07_AsyncFLEO_ring_of_stars.png
08_GLoRA_subspace_aggregation.png
09_CLARE_adapter_routing.png
10_FedDAA_drift_clustering.png
```

## 图注模板

```text
图 X. [图的功能性标题]。改绘/摘自 [作者], [论文简称], [年份]，Fig./Table [编号]。

本图在本文中用于说明：[一个明确技术结论]。
证据边界：[地面/仿真/在轨]；[原模型类型]；不可外推到[具体结论]。
```

## 摘录检查清单

- 图表是否来自原始论文、正式出版版本或作者公开版；
- 图号、页码、论文版本和年份是否记录；
- 图中结果是否对应论文的实际测试硬件、数据、任务和通信设定；
- 图注是否同时写明“可支持什么”与“不可外推什么”；
- 版权/许可不明确时，仅保存图号、页码和链接，不在仓库复制整页图片。

## 当前不摘录的材料

- 厂商宣传图、二手综述重绘图、未经原文核验的媒体配图；
- 只展示模型能力、但未给出持续更新/聚合/验证机制的图片；
- 将“在轨推理”包装为“在轨微调”的示意图。

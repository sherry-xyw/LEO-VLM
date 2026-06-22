# 持续适应论文图片摘录

本目录用于保存 `持续适应-核心文献.md` 中核心论文的原始图、系统框图和关键实验表。

## 优先摘录顺序

1. **CLeaRS**：long-horizon、modality-incremental、task-incremental 三类协议图，以及遗忘评测主表。
2. **LoRA-Det**：LoRA 注入位置、rank 选择与可训练参数/精度对比。
3. **RS-RAG**：多模态知识向量库、检索重排序与生成流程。
4. **FedMosaic**：参数化知识 adapter、document mask 和选择性聚合流程。
5. **FEDMEGA**：轨道内 ring all-reduce 与星地全局聚合架构。
6. **AsyncFLEO**：ring-of-stars 拓扑、卫星分组和陈旧度折扣。
7. **GLoRA**：gauge-aware 服务器表示、共识子空间和 heterogeneous-rank readout。

## 命名规则

```text
01_CLeaRS_protocol.png
01_CLeaRS_results.png
04_LoRADet_framework.png
07_RSRAG_pipeline.png
09_FedMosaic_aggregation.png
10_FEDMEGA_architecture.png
11_AsyncFLEO_topology.png
16_GLoRA_server_representation.png
```

仅保存从原始论文、正式出版版本或作者公开版提取的图。图注应注明原论文编号和页码；不使用二手综述、媒体转绘或未经核验的宣传图。

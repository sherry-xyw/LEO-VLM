# 表2：已发射星上 AI 计算平台、上星计算单元与算力

| 国家/地区      | 任务平台                                            |                                                             发射日期 | 平台形态                     | 计算单元                                                                               | 算力                                                                      | 内存、存储与功耗                                                                               | 在轨任务                                                                  | 参考文献                 |
| ---------- | ----------------------------------------------- | ---------------------------------------------------------------: | ------------------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------- |
| 欧洲         | Φsat-1 / FSSCat                                 |                                                       2020-09-03 | 6U CubeSat 遥感技术验证        | HyperScout-2 载荷内集成 Intel Movidius Myriad 2 VPU                                     | 约 1 INT8 TOPS                                                           | 器件/板级资料常见 512 MB LPDDR3                                                                | 星上 CNN 云检测、图像筛选，减少无效影像下传                                              | [R1–R3]              |
| 欧洲         | Φsat-2                                          |                                                       2024-08-16 | 6U CubeSat 遥感技术验证        | Ubotica CogniSAT-XE1；Intel Movidius Myriad 2 VPU（12 个向量核）                          | 未公开                                                                     | CogniSAT-XE1 板级：典型 2 W、最大 5 W（应用相关）；整星内存与存储未公开                                         | 在轨运行可安装式 AI 应用：云检测、船舶检测与分类、图像压缩、图像到地图转换等                              | [R3, R4]             |
| 澳大利亚 / 意大利 | SpIRIT / Loris                                  |                                                       2023-12-01 | 6U CubeSat 多相机 AI 载荷     | NVIDIA Jetson Nano 单板计算机；128-core Maxwell GPU + 四核 Cortex-A57 CPU                  | 模组标称 472 GFLOPS（FP16；非任务实测）                                             | 4 GB 64-bit LPDDR4（25.6 GB/s）；16 GB eMMC；计算功耗估计约 5 W                                   | 可见光/长波红外相机处理、AI 模型在轨微调、JPEG-XL 在轨压缩                                   | [R5, R6, R24]        |
| 德国         | SONATE-2                                        |                                                       2024-03-04 | 6U CubeSat 星上 AI 实验平台    | NVIDIA Jetson Xavier NX；6-core Carmel CPU、384-core Volta GPU、48 Tensor Cores       | 模组标称最高约 21 INT8 TOPS                                                    | 8 GB LPDDR4x；16 GB eMMC；10/15 W 模式（均为模组规格）                                             | 星上图像分类、异常处理，以及在轨训练/再训练相关试验                                            | [R7, R8]             |
| 波兰         | Intuition-1                                     |                                                       2023-11-11 | 高光谱遥感卫星                  | KP Labs Leopard DPU；AMD Zynq UltraScale+ MPSoC（四核 Cortex-A53、双核锁步 Cortex-R5）与 FPGA | 约 3 TOPS                                                                | 产品规格：16 GiB DDR4 ECC；4 GiB SLC NAND（EDAC）；2×240 GiB pSLC 冗余存储；5–20 W（负载相关）             | 192 波段高光谱数据处理、云筛选、目标/场景分析                                             | [R9]                 |
| 美国 / 国际空间站 | HyTI                                            |                                      2024-03-21（随 CRS-30 前往 ISS） | 热红外高光谱遥感载荷               | Unibap iX5-106：AMD Steppe Eagle CPU + Radeon GPU + Intel Myriad X VPU              | 4 TOPS（iX5-106 标称的 Myriad X AI 加速；非任务实测）                                | 2 GB DDR3 ECC；2×128 GB SATA SSD；10–30 W（系统负载相关）                                        | 热红外高光谱处理与农业用水监测；公开资料对独立部署和完整任务级计算结果披露有限                               | [R10]                |
| 美国         | LizzieSat-3 / FeatherEdge Gen-2                 |                                                       2025-03-14 | 商业遥感/边缘 AI 卫星            | FeatherEdge Gen-2，采用 NVIDIA Jetson Orin NX                                         | 约 100 TOPS                                                              | 16 GB 128-bit LPDDR5；680 GB pSLC NVMe SSD（ECC）；9.3 W 空闲、20 W 典型、30 W 峰值（均为 Gen-2 模组规格） | 多传感器数据融合、事件检测和边缘 AI；运营方宣布 FeatherEdge Gen-2 已在轨运行                     | [R11, R12]           |
| 美国 / 国际空间站 | HPE Spaceborne Computer 系列                      |                                                       2017 年起多批次 | ISS 轨道设施级高性能计算           | HPE Apollo、Edgeline EL4000、ProLiant 等服务器节点；GPU 配置随批次变化                             | 约 1 TFLOPS                                                              | 服务器级内存与存储，具体批次配置不同；依托 ISS 供电、散热与机架环境                                                   | HPC、科学数据分析、医疗计算及 AI 工作流                                               | [R13]                |
| 美国         | Starcloud-1                                     |                                                       2025-11-02 | 约 60 kg 自由飞行轨道数据中心技术验证卫星 | NVIDIA H100 GPU                                                                    | 飞行构型的 H100 SKU、精度、峰值算力与持续吞吐未公开；不以地面 H100 数据表替代在轨能力                      | 整星内存、运行功率与热稳态未公开；公司公开架构采用太阳能供电与被动辐射散热，但未单独披露 Starcloud-1 的具体配置                         | Starcloud 报告运行 Gemma 类模型并进行 nanoGPT 在轨训练                              | [R14, R15, R25, R26] |
| 美国         | Planet Pelican 星座（含 Pelican-4）                  | Pelican-1：2023-11-11；Pelican-2：2025-01-14；Pelican-3/4：2025-08-26 | 商业高分辨率遥感星座               | NVIDIA Jetson 平台；具体 Jetson SKU 未公开                                                 | 未公开                                                                     | 未公开                                                                                    | Pelican-4 完成在轨飞机检测；Pelican 系列使用 Jetson 平台开展星上边缘计算                     | [R16, R17]           |
| 中国         | 宝云卫星（Baoyun）                                    |                                                       2021-12-07 | 天算星座云原生协同推理实验卫星          | Raspberry Pi 计算载荷；Ubuntu Server 20.04 ARM                                          | 未公开                                                                     | 整星测得总功耗约 51.07 W；Raspberry Pi 子系统约 8.78 W；论文将星上计算归因于约 17% 的整星能耗                        | 与地面协同执行 YOLOv3-tiny 预检测和图像分块；论文报告协同推理相较单纯星上推理平均精度提升约 50%，下传数据量减少约 90% | [R19]                |
| 中国         | 创新雷神卫星（Chuangxingleishen / Innovation Raytheon） |                                                       2022-02-27 | 天算星座云原生协同推理实验卫星          | Raspberry Pi；Debian Buster                                                         | 未公开                                                                     | 未公开                                                                                    | 与宝云共同参与星地协同推理验证；轻量模型星上预处理、复杂模型地面推理                                    | [R19]                |
| 中国         | 天算星座北邮一号（BUPT-1）                                |                                                       2023-01-15 | 云原生星上计算主星/开放试验平台         | 多个星上计算组件与容器化服务平台                                                                   | 未公开                                                                     | 未公开                                                                                    | 已实现容器化载荷部署；开展脑机控星、实时视频传输、星上图像/视频处理与星地协同等试验                            | [R20]                |
| 中国         | 珞珈三号 01（烟台一号）                                   |                                                       2023-01-15 | 遥感视频卫星                   | 未公开                                                                                | 未公开                                                                     | 未公开                                                                                    | 已发射并获得遥感视频数据。用户提供图件称其支持遥感数据智能处理、预装目标监测/变化检测/图像压缩算法及手机端分发              | [R21, U1]            |
| 中国         | 四维高景三号 02                                       |                                                       2025-03-15 | 高分辨率遥感卫星                 | 未公开                                                                                | 星上算力达到 128 TOPS                                                         | 未公开                                                                                    | 用户提供图件标注支持海量图像的实时目标检测、识别、定位与切片                                        | [U1]                 |
| 中国         | 太空计算星座首批 12 星 / 三体计算星座首批                        |                                                       2025-05-14 | 星座级在轨计算节点                | 星载智能计算节点；芯片、板卡、内存和功耗清单未公开                                                          | 政府确认首批 12 星入轨与远期 1,000 POPS 目标；单星约 744 TOPS、每星搭载 8B 参数 AI 模型、首批约 5 POPS | 未公开                                                                                    | 实时在轨数据处理、模型推理、星间激光互联与星座协同；媒体转述还称 100 Gbps 星间光链路和 30 TB 共享存储           | [R22, R23]           |


## 参考文献

### 国际任务与硬件平台

- **[R1]** Giuffrida, G., et al. *The Φ-sat-1 Mission: The First On-Board Deep Neural Network Demonstrator for Satellite Earth Observation*. IEEE Transactions on Geoscience and Remote Sensing, 2021.  
  https://doi.org/10.1109/TGRS.2021.3070011
- **[R2]** European Space Agency. *Newly Space Qualified Myriad 2 Video Processor to Fly on CubeSat Mission*.  
  https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/Newly_Space_Qualified_Myriad_2_Video_Processor_to_Fly_on_CubeSat_Mission
- **[R3]** European Space Agency. *Φsat-2 Facts and Figures*.  
  https://www.esa.int/Applications/Observing_the_Earth/Phsat-2/Facts_and_figures
- **[R4]** Réaltra Space Systems Engineering / Ubotica. *CogniSAT-XE1 AI Accelerator & Vision Processing Board for CubeSats*, Issue 1.9, 2022.  
  https://ubotica.com/wp-content/uploads/2022/11/CogniSat-XE1-Board-Realtra-Datasheet-V1-9-November-2022.pdf
- **[R5]** Ortiz del Castillo, M., et al. *Mitigating Challenges of the Space Environment for Onboard Artificial Intelligence: Design Overview of the Imaging Payload on SpIRIT*. arXiv:2404.08399, 2024.  
  https://arxiv.org/abs/2404.08399
- **[R6]** Trenti, M., et al. *SpIRIT Mission: In-Orbit Results and Technology Demonstrations*. arXiv:2407.14034, 2024.  
  https://arxiv.org/abs/2407.14034
- **[R7]** University of Würzburg. *SONATE-2*.  
  https://www.informatik.uni-wuerzburg.de/en/space-technology/projects/active/sonate-2/
- **[R8]** Hübner, M., et al. *Testing the NVIDIA Jetson Xavier NX Module for the SONATE-2 Nano Satellite Mission*. IAC 2022, IAC-22-C3.2.7.  
  https://dl.iafastro.directory/event/IAC-2022/paper/71619/
- **[R9]** KP Labs. *Leopard Data Processing Unit and Intuition-1*.  
  https://kplabs.space/solutions/hardware/leopard/
- **[R10]** Unibap. *SpaceCloud iX5-106 Product Overview*, ver. 2.8.  
  https://unibap.com/wp-content/uploads/2023/10/spacecloud-ix5-product-overview-v28.pdf
- **[R11]** Sidus Space. *FeatherEdge Gen-2 Specifications*.  
  https://sidusspace.com/wp-content/uploads/2024/04/FeatherEdge-04.15.24.pdf
- **[R12]** Sidus Space. *Successful On-Orbit Operation of FeatherEdge Gen-2 Aboard LizzieSat-3*. 2025.  
  https://sidusspace.com/2025/05/16/sidus-space-announces-successful-on-orbit-operation-of-featheredge-gen-2-aboard-lizziesat-3/
- **[R13]** Hewlett Packard Enterprise. *HPE Spaceborne Computer*.  
  https://www.hpe.com/us/en/compute/hpc/supercomputing/spaceborne.html
- **[R14]** Reuters. *Starcloud Reaches $1.1 Billion Valuation as AI Space Race Heats Up*. 2026-03-30.  
  https://www.reuters.com/business/retail-consumer/starcloud-reaches-11-billion-valuation-ai-space-race-heats-up-2026-03-30/
- **[R15]** Starcloud. *Starcloud-1*（运营方任务资料）。  
  https://www.starcloud.com/starcloud-1
- **[R16]** Planet Labs. *Pelican Constellation / On-Orbit Edge Computing*（任务系列资料；具体 Jetson SKU 未公开）。  
  https://www.planet.com/
- **[R17]** Courier Mail. *AI Space Breakthrough Made Right Above Alice Springs: Tech Company*. 2026.  
  https://www.couriermail.com.au/news/planet-labs-says-its-made-an-ai-space-breakthrough-right-above-alice-springs/news-story/cc4acf455fe45c4efa249cd68172cd65

### 中国任务与在轨计算平台

- **[R18]** Li, L., Hu, J., Wu, F., and Zhao, J. *A Research and Strategy of Remote Sensing Image Denoising Algorithms*. ICNC-FSKD 2019.  
  https://arxiv.org/abs/1905.10236
- **[R19]** Wang, S., Zhang, Q., Xing, R., Qi, F., and Xu, M. *The First Verification Test of Space-Ground Collaborative Intelligence via Cloud-Native Satellites*. China Communications, 2023.  
  https://arxiv.org/abs/2311.06078
- **[R20]** Wang, C., Zhang, Y., Li, Q., Zhou, A., and Wang, S. *Satellite Computing: A Case Study of Cloud-Native Satellites*. arXiv:2307.08530, 2023.  
  https://arxiv.org/abs/2307.08530
- **[R21]** 中国运载火箭发射记录与任务公开资料：2023-01-15 长征二号丁任务包含北邮一号、珞珈三号01等卫星。建议后续以卫星运营方原始任务页面替换该发射记录类来源。
- **[R22]** 国务院新闻办公室/中国政府网. *China Launches Space Computing Satellite Constellation*. 2025-05-15.  
  https://english.www.gov.cn/news/202505/15/content_WS6825452ec6d0868f4e8f28e6.html
- **[R23]** Davis, W. *China Begins Assembling Its Supercomputer in Space*. The Verge, 2025-05-18.  
  https://www.theverge.com/news/669157/china-begins-assembling-its-supercomputer-in-space
- **[R24]** NVIDIA. *Jetson Nano Technical Specifications*.  
  https://www.nvidia.com/jetson-nano/
- **[R25]** SpaceX. *Bandwagon-4 Mission*.  
  https://www.spacex.com/launches/bandwagon-4
- **[R26]** Gunter's Space Page. *Starcloud 1 (Lumen 1)*（卫星数据库；用于质量与发射记录交叉核验）。  
  https://space.skyrocket.de/doc_sdat/starcloud-1.htm

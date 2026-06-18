# 单星部署 VLM 文献精读包

主文档：[single_satellite_vlm_deployment_reading.md](single_satellite_vlm_deployment_reading.md)

内容包括：

- 16 篇核心论文总览与逐篇 pipeline 精读；
- 每篇两张模型图、系统图或实验表整页截图；
- 量化、剪枝、蒸馏、视觉 token、KV cache、稀疏推理、early exit 和硬件感知横向矩阵；
- HPSC、SpaceCube、Leopard、Jetson/TensorRT、Myriad 2 官方资料附录；
- 本地 PDF 统一存放在 `../参考文献/References/`。

`render_selected_pages.py` 可从本地 PDF 重新生成 32 张截图。脚本只读取 PDF，不移动或修改原文件。

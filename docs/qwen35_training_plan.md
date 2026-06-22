# Qwen3.5-4B 遥感 Code-as-Action 训练计划

> 模型起点：`/home/wxy/data/models/Qwen3.5-4B`  
> 算力：2 × 48 GB GPU  
> 目标：在不调用预训练专家视觉模型、专家答案 API 或网络服务的前提下，使模型能够在遥感任务中自主决定是否使用代码，读取图像与科学数据文件，利用真实 execution observation 完成定量推理，并在失败后恢复或在证据充分时正确停止。

---

## 1. 总体训练路线

训练分为两个阶段：

```text
Qwen3.5-4B
  -> 阶段一：SFT
       SFT-1 领域知识与无代码决策
       SFT-2 视觉 grounding 与空间证据
       SFT-3 数据发现与单步可执行代码
       SFT-4 多轮执行、错误恢复与正确停止
  -> 阶段二：RL
       RL-1 可验证单步定量任务
       RL-2 多文件、多轮与时序任务
       RL-3 视觉—代码联合任务
```

训练目标不是提高代码调用率，而是提高 **code-induced gain**：模型应在需要代码时调用正确、可执行且有证据贡献的代码；在不需要代码时直接回答；在执行失败后根据 observation 修正；在证据已经充分时停止继续调用。

所有阶段复用 benchmark evaluator 的交互协议：

```text
system：说明代码沙箱能力与输出格式
user：图像、问题、DATA_FILES、INPUT_DIRS
assistant：<think>...</think> + 可选 <code>...</code>
tool：<observation>stdout / stderr / image artifact</observation>
assistant：继续推理、修正代码或 <answer>...</answer>
```

SFT 中只对 assistant 的 `<think>`、`<code>` 和 `<answer>` 计算 loss；user 内容和 tool observation 仅作为条件上下文。RL 中代码必须在与最终评测一致的 sandbox 内在线执行。

---

# 2. 阶段一：SFT

## 2.1 SFT 总体原则

SFT 不把“会写 Python”当成独立目标，而把 Code-as-Action 拆为五种行为：

1. 判断当前问题是否真的需要代码；
2. 发现并读取正确的输入文件、图像区域和元数据；
3. 编写最小必要、可执行、可验证的代码；
4. 根据 stdout、stderr 和图像产物更新推理；
5. 在证据充分时停止调用代码，输出最终答案。

SFT 采用课程式训练，但不是四组完全独立的数据。每一阶段从上一阶段 checkpoint 继续训练，并混入前序高质量样本，避免领域知识、视觉 grounding 和格式能力遗忘。

## 2.2 SFT-1：领域知识与无代码决策

### 目标

建立遥感基础知识、稳定输出协议，以及“简单题不应滥用代码”的先验。

### 数据内容

- 光谱、传感器、SAR/InSAR、热红外、遥感指数、预处理、GIS、投影和坐标系统；
- 遥感应用知识：农业、生态、水文、灾害、城市、气象；
- 简单场景识别、属性判断、概念题和 MCQ；
- 明确可以由题干或图像直接回答的空间关系题；
- 输出格式纠正样本：`<think>`、`<answer>`、MCQ 单字母答案。

### 样本要求

- 样本标记 `trajectory_type=no_code`、`code_necessity=must_not_use_code`；
- assistant 不输出 `<code>`；
- 知识题可使用 answer-only 或简短理由，不强制生成不可核查的长 CoT；
- MCQ 的 `<answer>` 仅保留选项字母。

## 2.3 SFT-2：视觉 grounding 与空间证据

### 目标

让模型把领域知识落到具体图像区域、对象关系、尺度和时相证据上，为后续视觉—代码联合任务提供可靠视觉基础。

### 数据内容

- 场景、目标、属性、计数和局部区域识别；
- 方向、邻近、包含、交叉、拓扑和整体布局；
- bbox、center point、mask、polygon、像素坐标；
- GSD、比例尺、像素距离与实际距离；
- 双时相图像中的变化区域、变化对象和变化方向；
- 高分辨率图像中的局部证据定位。

### 样本要求

过程监督样本至少具备一种可验证视觉证据：bbox、mask、polygon、中心点、像素距离、GSD 或明确比例尺。没有可靠视觉 GT 的开放描述样本可只保留 final-answer 监督，不进入强过程监督集合。

## 2.4 SFT-3：数据发现与单步可执行代码

### 目标

让模型学会使用 `DATA_FILES` 和 `INPUT_DIRS`，发现正确输入，编写最小可执行代码，并输出可被后续推理使用的关键中间结果。

### 轨迹类型

| trajectory_type | 训练内容 |
|---|---|
| `data_discovery` | 枚举目录、筛选文件、识别日期、变量、波段和输入类型 |
| `single_code` | 读取单个 raster/npy/csv，计算均值、极值、比例或像元数 |
| `index_code` | NDVI、NDWI、NBR、LST 等指数或公式计算 |
| `geometry_code` | 距离、面积、长度、坐标转换、单位换算 |
| `nodata_code` | NoData、NaN、dtype、scale factor、有效像元处理 |
| `image_processing` | 阈值、裁剪、连通域、形态学和基础图像统计 |

### 样本要求

每条代码样本必须满足：

- 在固定 sandbox 镜像中真实执行；
- 读取的路径、目录或图像来自用户可见输入；
- stdout 包含关键中间值，而不是只在代码内部完成计算；
- 最终答案能由 stdout、生成图像产物或确定性 verifier 验证；
- 不调用网络、专家视觉模型、专家答案 API 或未登记系统命令；
- 代码遵循最小必要原则，不为展示而写无关操作。

## 2.5 SFT-4：多轮执行、错误恢复与正确停止

### 目标

让模型完成完整的交互闭环：

```text
发现输入 -> 写代码 -> 执行 -> 读取 observation -> 修正或回答
```

### 数据内容

| trajectory_type | 典型能力 |
|---|---|
| `multi_step_code` | 多文件配对、多时相排序、统计聚合、趋势拟合 |
| `vision_code` | 视觉定位后利用 GSD、mask 或栅格完成量测 |
| `error_recovery` | 根据 stderr/stdout 修正路径、band、shape、NoData、CRS 或单位错误 |
| `correct_stop` | observation 已足够时停止调用代码并直接作答 |
| `tool_harm` | 识别错误文件、错误裁剪、错误单位或错误 observation 的危害 |

### 错误恢复原则

错误必须真实、可复现且与遥感任务有关，例如：

- `FileNotFoundError`：路径或文件 pattern 错误；
- `IndexError`：band 选择错误；
- `ValueError`：多时相数据 shape 不一致；
- `NaN`：未处理 NoData；
- CRS mismatch：经纬度与投影坐标混用；
- 单位错误：比例/百分比、Kelvin/Celsius、像元/平方米混淆。

错误代码与对应 stderr 只作为上下文；只监督模型对 observation 的解释、修正后的 reasoning、修正后的 code 和最终答案。无意义错误轨迹不得作为正样本。

## 2.6 SFT 数据规模与配比

先构建 4K 左右的 pilot，再按相同结构扩展到 30K–50K 正式数据。

| 数据类型 | Pilot 建议数量 | 作用 |
|---|---:|---|
| 领域知识与无代码决策 | 800 | 保留领域能力并抑制工具滥用 |
| 视觉 grounding 与空间关系 | 700 | 建立视觉证据和空间基础 |
| 数据发现与目录枚举 | 600 | 学会使用 `INPUT_DIRS` / `DATA_FILES` |
| 单步栅格/数组/几何代码 | 700 | 建立最小可执行代码能力 |
| 多步代码与时序计算 | 450 | 学习文件配对、聚合和趋势 |
| 视觉—代码联合 | 300 | 连接视觉 grounding 与数值计算 |
| 错误恢复 | 350 | 学习利用 stderr/stdout 修正 |
| 正确停止与工具负例 | 100 | 防止过度调用代码 |

pilot 的代码样本应覆盖所有 trajectory type；不能只扩大 `single_code` 数据而忽略 `no_code`、`error_recovery` 和 `correct_stop`。

## 2.7 SFT 起始配置

| 参数 | 建议起始值 |
|---|---|
| 训练方式 | LoRA |
| 精度 | bf16 |
| LoRA rank / alpha | 16 / 32；欠拟合时再试 32 / 64 |
| 训练模块 | LLM all-linear + multimodal projector |
| 视觉编码器 | 第一轮冻结 ViT；后续单独做最后若干 block 的低学习率 LoRA 消融 |
| 最大长度 | 4096；长多轮轨迹单独做 8192 消融 |
| 每卡 batch | 1 |
| 梯度累积 | 16–32 |
| LoRA 学习率 | 1e-4 起步 |
| epoch | pilot 1–2；正式训练 2–3，以验证集早停 |
| 优化器 | AdamW + 3%–5% warmup + cosine |
| 内存策略 | gradient checkpointing；packing 仅在图像边界与多轮角色边界正确时开启 |

---

# 3. 阶段二：RL

## 3.1 RL 启动条件

仅当 SFT-D checkpoint 满足以下条件后启动 RL：

- 训练格式能够稳定被 chat template 解析；
- 多轮 `<code>` / `<observation>` 不会在训练或推理中丢失；
- 代码块在 sandbox 中具有可接受的执行成功率；
- 模型至少能够完成一次基于 stderr/stdout 的有效修正；
- Q-Track sandbox run 不再显著弱于 disabled run；
- `must_not_use_code` 样本上的 unnecessary code rate 未明显上升。

RL 的任务是优化已有的工具行为，不是从零探索 Python 语法或 sandbox 接口。

## 3.2 RL 数据范围

第一轮 RL 仅使用具有确定性 verifier 的任务：

| 子类 | 主要 verifier |
|---|---|
| 指数与公式计算 | 数值容差、范围和单位 |
| 栅格统计 | 均值、极值、比例、像元数、分区统计 |
| 面积、距离、长度 | 单位、投影、数值容差 |
| 多时相趋势 | 斜率、趋势方向、变化类别 |
| 有 bbox/mask/坐标 GT 的量测 | IoU、center distance、面积或距离误差 |
| 数据发现 | 正确文件集、日期排序、变量匹配 |
| 错误恢复 | 修正后成功执行且最终答案正确 |

以下任务暂不进入 RL：开放文本解释、只能由 LLM judge 判分的长答案、无空间 GT 的细粒度量测、开放词汇小目标计数，以及存在多个同样合理答案的样本。

## 3.3 RL 课程

### RL-1：可验证单步任务

训练模型在需要时读取正确数据，完成一次计算，并让最终答案与 stdout 对齐。

### RL-2：多文件、多轮与时序任务

训练目录发现、变量配对、日期排序、跨文件计算、observation 驱动的继续或停止。

### RL-3：视觉—代码联合任务

训练模型先从图像得到可靠区域或对象证据，再通过代码完成量测、统计或变化计算。仅使用同时具有视觉 GT 与数值 GT 的任务。

## 3.4 RL 环境

RL rollout 使用与最终 evaluator 完全一致的环境：

- 相同 Docker 镜像和库版本；
- 相同 `PROJECT_ROOT`、`DATA_FILES`、`INPUT_DIRS` 注入方式；
- 相同文件挂载、timeout、内存限制和禁止项；
- 代码在线执行，不能以 SFT 缓存 observation 替代环境反馈。

初始设置建议每个 prompt 使用 `G=4` 个 rollout。优先保留部分成功、部分失败的样本；对于所有 rollout 都正确或都失败的样本，降低采样权重，以提高策略比较信号密度。

## 3.5 奖励设计

奖励约束“是否获得必要证据”，而不是约束固定函数名、固定代码文本或固定工具序列。

```text
R = R_final
  + I(final_correct) × (
      R_format
    + R_execution
    + R_necessity
    + R_evidence
    + R_observation
    + R_recovery
  )
  - R_excess
  - R_tool_harm
  - R_violation
```

| 项 | 含义 |
|---|---|
| `R_final` | 最终答案通过确定性 verifier |
| `R_format` | 输出协议完整，最终 `<answer>` 唯一且可解析 |
| `R_execution` | 代码成功执行，无 timeout 和非法操作 |
| `R_necessity` | `must_use_code` 时获得必要证据；`must_not_use_code` 时正确不调用 |
| `R_evidence` | 文件选择、band、NoData、维度、CRS、单位和中间数值正确 |
| `R_observation` | 最终答案与 stdout/图像产物一致，且未忽略 observation |
| `R_recovery` | 真实失败后完成有效修正 |
| `R_excess` | 重复读取、重复计算、无意义多轮和过长轨迹惩罚 |
| `R_tool_harm` | 错文件、错裁剪、错单位、错误 observation 导致结论变差 |
| `R_violation` | 调用网络、专家模型、越权文件或未授权系统命令 |

过程正向奖励应受 final correctness 门控，避免“代码执行成功但答案错误”的轨迹获得过多奖励。

## 3.6 必要证据 metadata

每个 RL prompt 保存 `required_evidence`，不保存强制函数名。例如：

```json
{
  "required_evidence": {
    "input_discovery": true,
    "required_file_pattern": "*NDVI*.tif",
    "nodata_checked": true,
    "expected_intermediate": {
      "valid_pixel_ratio_min": 0.8,
      "mean_range": [0.2, 0.8]
    },
    "final_unit": "%"
  }
}
```

模型可以通过不同代码实现获得答案，但必须完成可验证的输入发现、数据处理和证据获取。

---

# 4. 训练数据集构建流程

## 4.1 建立任务矩阵

按五个主任务及其细分类建立训练任务矩阵：

```text
visual_perception
domain_knowledge
spatial_reasoning
quantitative_estimation
temporal_change
```

每个任务模板必须定义：

- 输入类型：图像、栅格、矢量、表格、时间序列或混合输入；
- 目标答案与 answer format；
- `trajectory_type`；
- `code_necessity`：`must_use_code` / `optional_code` / `must_not_use_code`；
- 可用库与禁止操作；
- `required_evidence`；
- final/evidence/execution verifier；
- 可能的错误恢复类型。

## 4.2 数据源隔离与 scene-level split

在生成训练样本前固定数据源、训练/验证 split 和 OurBench 隔离规则。

以下内容不得进入训练数据：

- OurBench 原题、原图、原数据文件和近似改写；
- Earth-Bench、ThinkGeo、GeoMMBench 中与 OurBench 同图、同目录、同影像产品或同一事件的派生样本；
- 由 benchmark 答案、rubric 或 diagnostic chain 反推生成的数据；
- 同一产品的相邻裁剪、同一经纬度区域的高度重叠场景。

split 以 scene、image product、geographic region 和 event 为分组单位，而不是仅按题目文本随机切分。

## 4.3 构建原始任务包

每个任务包必须包含：

- 图像或科学数据文件；
- 输入目录；
- 问题模板；
- 原始标注、矢量、栅格或公式；
- 传感器、时间、地点、分辨率、投影等元数据；
- 许可证和来源信息。

每个任务包必须可以在固定 sandbox 中独立复现。

## 4.4 生成 GT 与 verifier

GT 优先来自：原始标注、栅格数组计算、矢量几何计算、物理公式、明确 bbox/mask/polygon、已知坐标和尺度。

每条样本至少具备三层 verifier：

| verifier | 检查内容 |
|---|---|
| final verifier | 最终答案是否正确 |
| evidence verifier | 文件选择、Band、NoData、单位、CRS、关键中间数值是否正确 |
| execution verifier | 代码是否可执行、是否超时、是否越权、是否读取合法输入 |

## 4.5 标注代码必要性

每条样本标注为：

```text
must_use_code
optional_code
must_not_use_code
```

`must_use_code` 至少满足以下之一：

- 关键数值仅存在于 tif/npy/nc/csv 等文件中；
- 更换输入文件后正确答案会改变；
- 不执行代码无法获得数组统计、时序趋势或几何量；
- direct-answer 与 no-op 对照无法稳定通过 verifier。

`must_not_use_code` 样本用于训练正确的不调用行为，覆盖概念题、简单视觉题和可以直接回答的 MCQ。

## 4.6 生成成功轨迹

对每个任务生成 2–4 条候选轨迹。教师模型只能看到用户可见信息：问题、图像、`DATA_FILES`、`INPUT_DIRS`、可用库和输出协议；不得读取答案、参考程序或 verifier 内部信息。

每条候选轨迹在真实 sandbox 执行。仅保留：

- 成功执行；
- 输入发现和文件读取正确；
- 关键中间证据通过 verifier；
- 最终答案正确；
- 未调用禁止工具；
- 轨迹长度合理。

在多个成功轨迹中，优先保留“最短但证据充分”的轨迹，防止模型学到冗余调用习惯。

## 4.7 生成错误恢复轨迹

从已验证成功轨迹做受控变异，而不是随机伪造错误：

- 错误路径或文件 pattern；
- 错误 band；
- 缺失 import；
- 未处理 NoData；
- shape mismatch；
- 日期排序错误；
- 比例/百分比、温度或面积单位错误；
- CRS 错误。

执行变异代码，确认产生真实 stderr 或异常 observation；再生成修正代码并二次验证。最终数据格式为：

```text
错误尝试（上下文，不计 loss）
-> 真实 stderr/stdout
-> 修正 reasoning + code（监督）
-> 成功 observation
-> final answer（监督）
```

## 4.8 生成正确停止与工具伤害样本

在 observation 已充分的任务上构造：

- `correct_stop`：模型应直接回答；
- `excess`：继续调用无意义代码；
- `tool_harm`：错误文件、错误裁剪、错误单位或错误阈值导致答案变差；
- `observation_ignored`：工具已经给出证据但最终答案忽略或违背该证据。

这些样本用于监督和 RL 中约束“工具调用越多不等于推理越好”。

## 4.9 质量控制与数据标签

每条样本至少通过：schema 校验、路径校验、sandbox 可复现执行、final verifier、evidence verifier、禁止工具检查、OurBench overlap 检查和人工抽检。

| 标签 | 标准 |
|---|---|
| `gold` | 原始标注或公式产生，执行和人工审核均通过 |
| `silver` | 自动生成，执行与双 verifier 通过，抽检通过 |
| `recovery` | 包含真实失败、真实 observation 和成功修正 |
| `negative` | 不应调用代码、工具伤害或正确停止样本 |
| `discard` | 未执行、证据不足、不可验证或存在泄漏风险 |

---

# 5. Pilot、评测与推进条件

## 5.1 Mini-pilot：500 条轨迹

正式 4K pilot 前，先构建 500 条 mini-pilot：

| 类型 | 数量 |
|---|---:|
| `no_code` | 75 |
| `single_code` | 100 |
| `data_discovery` | 100 |
| `multi_step_code` | 75 |
| `vision_code` | 50 |
| `error_recovery` | 75 |
| `correct_stop` | 25 |

通过条件：

1. 100% 样本可解析为训练 messages；
2. 至少 95% 代码轨迹可复现执行；
3. 各类型至少 90% 通过 final/evidence verifier；
4. 小规模 SFT 后格式合规率不下降；
5. Q-Track mini-val 上 sandbox 相对 disabled 有净增益；
6. `must_not_use_code` 样本上的 unnecessary code rate 不上升。

## 5.2 正式评测与消融

主评测使用 OurBench 五类 macro：

```text
visual_perception
domain_knowledge
spatial_reasoning
quantitative_estimation
temporal_change
```

按以下顺序进行实验：

| 实验 | 目的 |
|---|---|
| 原始 Qwen3.5-4B | 训练前基线 |
| SFT-1 | 领域知识与无代码决策贡献 |
| SFT-1+2 | 视觉 grounding 贡献 |
| SFT-1+2+3 | 数据发现与单步代码贡献 |
| SFT-1+2+3+4 | 多轮恢复与正确停止贡献 |
| SFT 全量 + RL | 可验证 RL 总增益 |
| 去掉 `no_code` | 检查工具滥用 |
| 去掉 `data_discovery` | 检查目录与文件选择能力 |
| 去掉 `error_recovery` | 检查失败后的修正能力 |
| 去掉 `correct_stop` | 检查重复调用与停止能力 |
| 只监督最终答案 | 验证轨迹监督是否必要 |
| 只保留成功轨迹 | 验证恢复轨迹的价值 |
| 未执行代码轨迹对照 | 验证真实执行筛选的必要性 |
| 禁用工具 / 空操作 / 随机操作 | 估计 tool gain、tool harm 和净收益 |
| 反事实 observation | 检查最终答案是否真正依赖执行结果 |

除准确率外，必须报告：

- code use rate；
- code execution success rate；
- data file read rate；
- observation use rate；
- unnecessary code rate；
- recovery success rate；
- correct-stop rate；
- mean turns；
- tool-induced gain / harm / net gain；
- 各主任务、子类和 operation tag 的准确率；
- 图像消融和反事实 observation 下的性能变化。

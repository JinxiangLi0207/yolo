# 面向 DUO 水下小目标检测的改进 YOLO11n 论文方法论与实验指导

> 适用项目：基于 YOLO11n 的 DUO 水下小目标检测  
> 研究目标：面向浑浊、低照度和小目标密集场景，形成一篇以算法改进为主的 SCI 二区/三区小论文  
> 当前硬件：两台 RTX 3090 云服务器，分别记为“训练1”和“训练2”  
> 当前状态：N4 在 `power=3.0` 达到 0.678852 mAP50-95，所有相同 power 下均高于 C1A；C1A 终止多种子训练，当前使用既有 N4 seed 1/2 权重验证 QP3 稳定性  
> 文档目标：将 11 篇真实论文的方法论转化为本项目当前可执行、可复现、可写入论文的实验路线  
> 项目适配更新时间：2026-07-15

---

## 1. 核心结论

近年的有效改进 YOLO 论文通常不是“找到一个热门模块并塞进网络”，而是遵循以下链条：

```text
数据和错误分析
→ 明确一个主要失败模式
→ 选择与失败模式匹配的结构改动
→ 单模块受控消融
→ 组合实验与交互验证
→ 第二公共数据集验证泛化
→ 精度、参数量、GFLOPs、延迟联合评价
→ 多随机种子与可视化支持结论
```

对当前 DUO 项目，已经确认的关键实验现象是：

```text
早期 A2-Lite：在 batch=16、SGD、100 epochs 下 Recall +0.016，
               但 mAP50-95 -0.005，该增益没有在后续统一协议中稳定复现。

N4-Full：在 batch=96、100 epochs、seeds 0/1/2 下，
         Precision 平均 +0.025，mAP50-95 平均 +0.005，
         Recall 平均 -0.023，mAP50 平均 -0.001，
         scallop mAP50-95 平均 -0.010。
```

因此当前问题已经从“寻找任意涨点模块”收敛为两个明确问题：

1. **如何保留 N4 的高 IoU 定位收益，同时恢复 Recall 和 mAP50**；
2. **如何避免类别无关质量学习对稀有类别 scallop 的稳定伤害**。

当前最近的一步不是重新堆叠注意力，而是执行受控组合实验：

```text
C1A：严格 P3-only MSFF + QualityDetect
C1B：原 A2 数据流 MSFF + QualityDetect
```

两者只改变 MSFF 是否继续流入 P4/P5，用于判断 MSFF 的额外候选能否被 N4 的质量排序转化为有效 AP。

---

## 2. 近年真实论文与可复用经验

以下论文均在公共数据集上进行了实验。表中提升值应理解为各论文自身训练协议下相对其 baseline 的结果，不能直接跨论文横向比较。

| 论文 | 年份 / 来源 | Baseline 与公共数据集 | 主要改动 | 报告的主要收益 | 可复用的方法论 |
|---|---|---|---|---|---|
| **AGS-YOLO: An Efficient Underwater Small-Object Detection Network for Low-Resource Environments** | 2025, *Journal of Marine Science and Engineering* | YOLO11n；DUO、RUOD | AMSA 多尺度注意力、跨尺度 Neck、GSConv | DUO 上 mAP50 +1.3 个百分点，mAP50:95 +2.6 个百分点，并在 RUOD 验证泛化 | 主模块解决小目标表征，Neck 解决跨尺度融合，轻量模块最后用于控制代价；完整消融而非只报最终组合 |
| **Lightweight underwater object detection method based on multi-scale edge information selection**（MAW-YOLOv11） | 2025, *Scientific Reports* | YOLO11；URPC | 图像去雾、MSEIS/C3kMSEIS、ADown、WIoUv3 | mAP 达 81.4%，比 YOLO11 提高 2.1 个百分点；参数量降至 2.11M | 把水下退化、边缘弱化和部署约束分别映射到预处理、Backbone 与轻量化设计 |
| **Efficient underwater object detection based on feature enhancement and attention detection head** | 2025, *Scientific Reports* | YOLOv5n/v6n/v8n；UTDAC2020、RUOD | PSEM Neck 特征增强、SDWH 注意力检测头 | YOLOv8n 在 UTDAC2020 +2.8 mAP，在 RUOD +2.7 mAP；模块在多个 YOLO 版本上均验证 | 好模块应尽量跨 baseline 验证；Neck 与 Head 的作用要分别消融，并展示特征图或注意力可视化 |
| **An Improved YOLO-Based Algorithm for Aquaculture Object Detection**（AOD-YOLO） | 2025, *Applied Sciences* | YOLO11s；URPC2020、RUOD | 用自研 RGL 模块替换 C3k2，并进行结构效率优化 | URPC2020 mAP50 +2.6，RUOD +2.4；参数量减少 0.68M | 论文主创新最好落到一个自研模块；同设置双数据集、逐项消融、PR 曲线和效率对比形成完整证据链 |
| **A Lightweight underwater detector enhanced by Attention mechanism, GSConv and WIoU on YOLOv8**（AGW-YOLOv8） | 2024, *Scientific Reports* | YOLOv8；URPC2020 | LCAHE-WT、CBAM、GSConv、SE、WIoU | mAP 提高 2.5 个百分点，参数量略降 | 数据增强/图像处理、注意力、轻量化和 Loss 各自解决不同问题，但模块较多时必须逐项消融，避免无法归因 |
| **Efficient Small-Object Detection in Underwater Images Using the Enhanced YOLOv8 Network** | 2024, *Applied Sciences* | YOLOv8s；UTDAC2020、Pascal VOC、UODD | FasterNet-T0、额外小目标 Head、DCNv2、Coordinate Attention | 小目标 Head 单独带来约 1.2 个百分点收益；在 UTDAC2020 达到 52.12 AP，1280 输入时 53.18 AP；VOC 验证通用性 | 小目标问题可直接通过高分辨率 Head 处理；输入分辨率必须作为独立变量，不得与结构收益混在一起 |
| **SRE-YOLOv8: An Improved UAV Object Detection Model Utilizing Swin Transformer and RE-FPN** | 2024, *Sensors* | YOLOv8；VisDrone2021 | Swin Transformer、轻量残差 FPN、动态检测头 | 相对原 YOLOv8 报告 9.2% 提升 | 对高密度小目标，扩大上下文和重构 FPN 往往比单纯注意力更有效；但应核对复杂度与速度代价 |
| **CF-YOLO for small target detection in drone imagery based on YOLOv11 algorithm** | 2025, *Scientific Reports* | YOLO11n；VisDrone2019、TinyPerson、HIT-UAV | CS-FPN、FRM、Sandwich 特征增强、RFAConv、轻量解耦 Head | 三个数据集 mAP50 分别提高 12.7、10.1、3.5 个百分点 | 强论文会用多个公共数据集证明不是只对一个数据集有效；主干、融合、Head 均围绕同一“小目标细节与背景干扰”问题设计 |
| **MFA-YOLO: a multi-feature aggregation approach for small-object detection in drone imagery** | 2025, *Scientific Reports* | YOLOv8n；VisDrone、UAVDT | 多特征聚合与轻量结构 | VisDrone AP50 +3.6、AP +2.4，参数量减少 17%；UAVDT 验证泛化 | 同时报告 AP50 和更严格 AP，防止只在低 IoU 阈值下受益；精度与参数下降可形成更强贡献 |
| **RPS-YOLO: A Recursive Pyramid Structure-Based YOLO Network for Small Object Detection in UAV Scenarios** | 2025, *Applied Sciences* | YOLOv8s；VisDrone-DET2021、UAVDT | 递归金字塔特征提取与细节复用 | mAP50 分别提高 8.2 和 3.4 个百分点；GFLOPs 从 28.5 增至 37.7 | 精度提升不等于部署价值；论文需诚实报告计算代价，并明确是“高精度模型”还是“轻量模型” |
| **SRM-YOLO for Small Object Detection in Remote Sensing Images** | 2025, *Remote Sensing* | YOLOv8n；VisDrone2019、SSDD、NWPU VHR-10 | Reuse Fusion、SPD-Conv、小目标 Head、MPDIoU | VisDrone2019 mAP50 +5.2 个百分点，并进行跨数据集测试 | 同一模块在别的数据集有效，不代表在 DUO 一定有效；你的 MPDIoU 失败说明 Loss 与数据、框尺度、实现细节存在强耦合，必须先单独验证 |

### 论文链接

1. AGS-YOLO: https://doi.org/10.3390/jmse13081465  
2. MAW-YOLOv11: https://doi.org/10.1038/s41598-025-13566-3  
3. PSEM + SDWH: https://doi.org/10.1038/s41598-025-89421-2  
4. AOD-YOLO: https://doi.org/10.3390/app152111724  
5. AGW-YOLOv8: https://doi.org/10.1038/s41598-024-75809-z  
6. Enhanced YOLOv8 for underwater small objects: https://doi.org/10.3390/app14031095  
7. SRE-YOLOv8: https://doi.org/10.3390/s24123918  
8. CF-YOLO: https://doi.org/10.1038/s41598-025-99634-0  
9. MFA-YOLO: https://doi.org/10.1038/s41598-025-32247-9  
10. RPS-YOLO: https://doi.org/10.3390/app15042039  
11. SRM-YOLO: https://doi.org/10.3390/rs17122099

---

## 3. 从论文中提炼出的通用研究方法论

### 3.1 先定义失败模式，再选模块

每个候选模块都必须对应一个可观测问题。

| 可观测问题 | 证据 | 合理的干预方向 | 不优先的方向 |
|---|---|---|---|
| 小目标漏检严重 | 小目标 Recall/AP 明显低、浅层特征弱 | P2/P3 高分辨率分支、浅层细节复用、小目标 Head | 直接换 Loss |
| Recall 提升但 mAP 下降 | 检出更多候选，但 PR 曲线、Precision 或高 IoU AP 下降 | 门控融合、背景抑制、Head 重标定、残差缩放 | 再叠加一个纯召回模块 |
| mAP50 尚可但 mAP50:95 低 | 高 IoU 阈值定位质量不足 | 可变形卷积、定位分支增强、温和 Loss 消融 | 一次性完全替换 Loss 且不检查尺度 |
| 参数量/GFLOPs 过高 | 准确率已有优势，但部署代价大 | GSConv、PConv、ADown、轻量 Head | 在没有精度优势时先轻量化 |
| 某类别样本少且 AP 低 | 类别统计、混淆矩阵、PR 曲线 | 类别感知采样、Copy-Paste、分类分支重加权 | 仅靠 Backbone 堆模块 |
| 复杂背景误检 | Precision 下降、热力图关注背景 | 通道/空间门控、前景增强、解耦 Head | 无选择地增强所有尺度 |

### 3.2 一篇改进 YOLO 论文最好只有一个主创新

推荐结构：

```text
主创新：解决核心失败模式，必须带来主要 mAP 收益
辅助创新：控制参数量、补充特征或改善训练稳定性
```

不推荐：

```text
注意力 A + 卷积 B + Neck C + Loss D + 数据增强 E
```

如果完整模型有四个以上改动，审稿人容易质疑：

- 每个模块是否必要；
- 性能到底来自结构还是训练设置；
- 是否只是已有模块拼接；
- 是否存在选择性汇报。

### 3.3 模块必须经过三层验证

```text
第一层：单模块是否优于 baseline
第二层：与已有候选模块组合后是否互补
第三层：第二数据集是否仍有效
```

只有第一层有效而第二层失效，说明模块存在交互冲突；只有单数据集有效，说明可能过拟合数据特征。

### 3.4 精度提升应同时观察四种信号

不能只看 mAP50：

1. mAP50；
2. mAP50:95；
3. Precision / Recall 变化；
4. 各类别 AP、尤其弱类别 AP。

常见解释：

```text
Recall ↑，Precision ↓，mAP 不升
→ 模型变得更激进，候选框增加，但背景误检或置信度排序变差

mAP50 ↑，mAP50:95 不升
→ 粗定位改善，但高质量定位没有改善

弱类别 Recall ↑，弱类别 AP ↓
→ 找到更多疑似目标，但定位、分类置信度或排序质量较差
```

### 3.5 轻量化必须排在准确率主线之后

GSConv、深度可分离卷积、PConv 等模块通常用于降低开销。它们可能保持或小幅改善精度，但不应被假设为“mAP 提升模块”。

当前项目推荐顺序：

```text
先得到稳定高于 A0 的准确率模型
→ 再用 GSConv 等降低参数/GFLOPs
→ 最后验证精度损失是否可接受
```

---

## 4. 重要的数据划分修正

### 4.1 不要继续把官方 test 当作频繁调参的 val

如果每次模块筛选都在 DUO 官方 test 上验证，并根据该结果选择模块，最终会产生测试集选择偏差。SCI 论文需要把开发验证和最终测试分开。

当前项目的历史实验，包括 S0、N4 和正在运行的 C1，仍使用 `images/test` 作为 Ultralytics `val`，并据此选择 `best.pt`。因此这些结果可用于工程筛选和方向判断，但在严格论文协议中存在测试集参与模型选择的风险。算法结构确定后，应重建下面的开发划分，并至少重跑 baseline、N4 和 final model。

推荐重建开发划分：

```text
DUO 官方 train
├── dev_train：90%
└── dev_val：10%，按类别与含目标图像进行分层划分

DUO 官方 test
└── official_test：只用于最终模型和主要 baseline 的最终评价
```

操作要求：

1. 固定划分文件并提交到项目仓库；
2. 所有模块筛选只使用 `dev_val`；
3. 模型确定后，用完整官方 train 重新训练；
4. 最终在 official test 上评价；
5. 不根据 official test 结果继续改模型。

建议生成：

```text
datasets/DUO-YOLO/splits/dev_train.txt
datasets/DUO-YOLO/splits/dev_val.txt
datasets/DUO-YOLO/splits/official_test.txt
datasets/DUO-YOLO/duo_dev.yaml
datasets/DUO-YOLO/duo_official.yaml
```

---

## 5. 当前项目的实验主线

### 5.1 当前假设

N4 已经证明定位质量预测能够改善高 IoU 候选框排序，但完整乘法融合使预测变得保守。A2-Lite 的早期结果则表明 P3 多尺度增强可能增加候选目标，但也容易增加背景响应。

当前 C1 的核心假设是：

> MSFF 提高 P3 小目标响应，QualityDetect 对新增候选进行定位质量监督和置信度排序，两者可能形成“提高检出倾向 + 抑制低质量框”的互补关系。

该假设必须通过二因素消融验证，不能只训练最终组合：

| MSFF | QualityDetect | 模型 | 当前状态 |
|---|---|---|---|
| 否 | 否 | YOLO11n | 已完成 50/100 轮基准 |
| 是 | 否 | A2/B1 MSFF-P3 | 已完成，结果不稳定 |
| 否 | 是 | N4-Full | 已完成三种子正式验证 |
| 是 | 是 | C1A/C1B | 当前实验 |

### 5.2 推荐实验优先级

```text
P0：完成 C1A 严格 P3-only 与 C1B 原 A2 数据流的 50 轮对照
P1：若 C1 有效，完成 100 轮三种子验证
P2：若 C1 无效，回到 N4，设计残差/尺度自适应质量融合
P3：针对 scallop 设计只作用于质量分支的类别均衡监督
P4：完成置信度-IoU 相关性、尺度级 AP 和退化鲁棒性分析
P5：最终再做第二公共数据集和同环境效率比较
```

---

## 6. 统一训练协议

当前两台服务器均为 RTX 3090 24GB。协议必须与已经完成的 S0、N4 和模块筛选实验保持一致。

### 6.1 开发筛选设置

```yaml
epochs: 50
imgsz: 640
batch: 96
optimizer: auto  # 当前日志实际选择 AdamW(lr=0.00125, momentum=0.9)
amp: true
workers: 8
cache: ram
seed: 0
deterministic: true
quality_loss_gain: 0.5  # 仅 QualityDetect 系列生效
rcqfl: false
sqr: false
plots: true
val: true
```

说明：

- P2 四检测头在 `batch=96` 下曾触发 TaskAlignedAssigner OOM/CPU fallback，因此该结构后来使用 `batch=48`，不能与常规 `batch=96` 结果混为公平对比；
- 常规 YOLO11n、MSFF 和 QualityDetect 模型使用 `batch=96`；
- `cache=True` 会出现 RAM cache 非完全确定性警告。开发阶段沿用该设置保持历史协议一致，正式论文复跑建议改为 `cache=disk` 或 `cache=False`；
- 早期 A2 的 `batch=16 + SGD` 结果只作为研究线索，不作为正式主表；
- 两台服务器必须使用同一 Git commit、相同数据配置和相同预训练权重文件。

### 6.2 正式实验设置

```yaml
epochs: 100
imgsz: 640
batch: 96
optimizer: auto
amp: true
workers: 8
cache: disk  # 磁盘允许时使用
seeds: [0, 1, 2]
deterministic: true
quality_loss_gain: 0.5
rcqfl: false
sqr: false
```

最终报告使用：

```text
mean ± std，至少三个随机种子
```

当前 100 轮正式结果：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n | 0.834 ± 0.003 | **0.773 ± 0.009** | **0.844 ± 0.005** | 0.658 ± 0.001 |
| N4-Full | **0.860 ± 0.011** | 0.750 ± 0.007 | 0.843 ± 0.001 | **0.663 ± 0.001** |
| 均值变化 | **+0.025** | -0.023 | -0.001 | **+0.005** |

---

## 7. 代码 Agent 的通用工作规则

代码 Agent 必须遵守以下规则。

### 7.1 修改前检查

1. 输出当前 Git commit；
2. 输出 Ultralytics 版本；
3. 检查实际源码位置；
4. 检查已有自定义模块是否存在；
5. 不覆盖 YOLO11n、A2-Lite、N4 和已经完成实验的配置；
6. 新建独立分支或 commit。

建议分支：

```text
codex/c1-msff-quality
codex/n4-residual-quality-fusion
codex/n4-class-balanced-quality
```

### 7.2 每个实验必须产生独立文件

```text
ultralytics/cfg/models/11/yolo11n-<experiment>.yaml
work/<experiment>_Experiment_Guide.md
work/<experiment>_Result_Report.md
runs/detect/<experiment>/args.yaml
runs/detect/<experiment>/results.csv
runs/detect/<experiment>/weights/best.pt
```

当前关键文件：

```text
ultralytics/cfg/models/11/yolo11n-quality-n4.yaml
ultralytics/cfg/models/11/yolo11n-msff-quality-c1.yaml
ultralytics/cfg/models/11/yolo11n-msff-quality-c1-legacy.yaml
ultralytics/nn/modules/head.py
ultralytics/nn/modules/conv.py
ultralytics/utils/loss.py
work/N4_QualityDetect_Paper_Innovation_Record.md
```

### 7.3 不允许的行为

- 不得同时修改网络、Loss、数据增强和训练超参数；
- 不得把模块插入 P3、P4、P5 多处后再声称是 P3-only；
- 不得修改类别映射或数据划分；
- 不得在同一组对照中混用 `optimizer=auto` 与显式 SGD；当前主协议统一使用 `optimizer=auto`；
- 不得在 OOM fallback 后继续把该 run 当作正式结果；
- 不得只保存最好结果而丢弃失败 seed；
- 不得在没有备份时修改 `loss.py` 或 `metrics.py`；
- 不得直接复制论文代码而不检查许可证与引用要求。
- 不得把 `Precision/Recall` 单个置信度阈值下的变化等同于 AP 提升；
- 不得把早期 batch=16 的 A2 结果写成已在当前正式协议下稳定复现。

---

## 8. C1：MSFF 与质量检测头的结构消融

### 8.1 二因素实验矩阵

| 编号 | MSFF | QualityDetect | 目的 |
|---|---|---|---|
| S0 | 否 | 否 | 统一协议 YOLO11n 基准 |
| B1 | 是 | 否 | 测试 MSFF 独立贡献 |
| N4 | 否 | 是 | 测试质量检测头独立贡献 |
| C1 | 是 | 是 | 测试两者是否互补 |

C1 只有在同时超过 N4 和 B1 时，才能证明组合具有互补性。仅超过 B1 不能说明 MSFF 对 N4 有贡献。

### 8.2 C1A：严格 P3-only

配置：`ultralytics/cfg/models/11/yolo11n-msff-quality-c1.yaml`

```yaml
- [16, 1, MSFF, [256]]
- [16, 1, Conv, [256, 3, 2]]
```

解释：MSFF 输出只进入 P3 检测输入，P4/P5 仍由原始 P3 构建。因此该版本能够隔离“只增强小目标检测尺度”的作用。

### 8.3 C1B：原 A2 数据流

配置：`ultralytics/cfg/models/11/yolo11n-msff-quality-c1-legacy.yaml`

```yaml
- [16, 1, MSFF, [256]]
- [-1, 1, Conv, [256, 3, 2]]
```

解释：MSFF 输出继续生成 P4/P5，实际会间接影响三个检测尺度。该版本用于复现早期 A2 的真实数据流，不应再称为严格 P3-only。

### 8.4 公平性检查

C1A 与 C1B 必须满足：

```text
同一 Git commit
同一 DUO 数据配置
相同层数、参数量和 GFLOPs
相同 pretrained transfer 数量
epochs=50, batch=96, imgsz=640
seed=0, deterministic=True
quality_loss_gain=0.5, quality_power=1.0
rcqfl=False, sqr=False
```

启动日志中唯一关键结构差异应为：

```text
C1A layer 18: from 16
C1B layer 18: from -1
```

当前两个组合模型插入 MSFF 后发生层号移动，预训练迁移日志为 `288/553`。该值低于 N4 的迁移数量，因此 C1 若失败，需要区分“模块交互失败”和“预训练迁移不足”两种可能，不能直接得出 MSFF 与质量头绝对不兼容。

### 8.5 当前训练命令

训练1，C1A严格版：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-msff-quality-c1.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 epochs=50 imgsz=640 workers=8 device=0 \
  pretrained=weights/yolo11n.pt cache=True \
  seed=0 deterministic=True \
  quality_loss_gain=0.5 rcqfl=False sqr=False \
  name=C1A_strict_p3_msff_quality_e50_b96_seed0
```

训练2，C1B原A2数据流：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-msff-quality-c1-legacy.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 epochs=50 imgsz=640 workers=8 device=0 \
  pretrained=weights/yolo11n.pt cache=True \
  seed=0 deterministic=True \
  quality_loss_gain=0.5 rcqfl=False sqr=False \
  name=C1B_legacy_msff_quality_e50_b96_seed0
```

两台服务器的数据集绝对路径可能不同，本地保存的 `DUO.yaml` 可能通过 `git stash` 管理。训练前必须检查 `path:` 和实际图像目录，不能因 `git pull` 后遗漏 `git stash pop` 而使用错误路径。

### 8.6 C1 的 50 epochs 结果

统一条件：`batch=96`、`imgsz=640`、`seed=0`、`quality_loss_gain=0.5`、`quality_power=1.0`。

| 模型 | Precision | Recall | mAP50 | mAP50-95 | scallop mAP50 | scallop mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| S0 YOLO11n | **0.875** | 0.738 | **0.842** | **0.650** | **0.672** | **0.507** |
| C1A strict P3-only | 0.837 | **0.750** | 0.832 | 0.640 | 0.627 | 0.464 |
| C1B original A2 dataflow | 0.854 | 0.740 | 0.835 | 0.642 | 0.653 | 0.483 |

相对 S0，C1A 的变化为 `P -0.038`、`R +0.012`、`mAP50 -0.010`、`mAP50-95 -0.010`；C1B 的变化为 `P -0.021`、`R +0.002`、`mAP50 -0.007`、`mAP50-95 -0.008`。两种组合都没有通过 50 轮晋级线。

C1B 相对 C1A 为 `P +0.017`、`R -0.010`、`mAP50 +0.003`、`mAP50-95 +0.002`。这说明原 A2 的级联数据流在 AP 上略优于严格 P3-only，但优势很小，尚不能证明级联多尺度增强有效。两者的主要共同问题是 scallop 明显退化，其中 C1A 的 scallop mAP50-95 比 S0 低 0.043，C1B 低 0.024。

当前解释只能写成：MSFF 与 QualityDetect 的直接串联在训练前 50 轮没有表现出互补性，新增特征增强可能扰乱了质量分支的排序学习；同时，组合模型仅迁移 `288/553` 项预训练权重，收敛速度慢于 S0/N4，因此允许进行一次 100 轮最终诊断。

### 8.7 100 epochs 最终诊断规则

100 轮实验必须从 `weights/yolo11n.pt` 重新开始，不能对 50 轮任务执行 `resume`。原因是 50 轮任务的学习率进度、Mosaic 关闭时点和优化器状态均按 50 轮总长度生成，继续训练不能视为严格的 100 轮同协议实验。

seed 0 的正式参照为：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n，100 epochs | 0.831 | **0.778** | 0.844 | 0.657 |
| N4，100 epochs，统一质量融合评估 | **0.859** | 0.760 | **0.848** | **0.664** |

C1 的首要判据不是“超过 YOLO11n”，而是“超过同 seed 的 N4”。建议采用以下晋级条件：

```text
必要条件：mAP50-95 > 0.664
必要条件：mAP50 >= 0.844，且相对 N4 不出现明显退化
稀有类条件：scallop mAP50-95 >= 0.510，至少恢复到基准附近
解释条件：Recall 或 scallop AP 至少有一项明确优于 N4
```

如果 C1A/C1B 在 100 轮仍未超过 N4，则停止 MSFF 组合路线，不运行 seeds 1/2；曲线末端仍上升只能说明尚未完全平台化，不能替代 `best.pt` 指标和同协议对照。若只有一个版本超过 N4，再只对该版本运行 seeds 1/2。

### 8.8 C1 的 100 epochs 结果与决策

所有结果均为 `seed=0`，C1 和 N4 均按 `quality_power=1.0` 比较。

| 模型 | Precision | Recall | mAP50 | mAP50-95 | scallop P | scallop R | scallop mAP50 | scallop mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 YOLO11n | 0.831 | **0.778** | 0.844 | 0.657 | 0.730 | **0.590** | 0.671 | 0.510 |
| N4 QualityDetect | 0.859 | 0.760 | 0.848 | **0.664** | 0.791 | 0.557 | 0.670 | 0.501 |
| C1A strict P3-only | 0.862 | 0.762 | **0.854** | 0.662 | 0.812 | **0.590** | **0.704** | **0.521** |
| C1B original A2 dataflow | **0.877** | 0.755 | 0.842 | 0.654 | **0.824** | 0.560 | 0.644 | 0.481 |

C1A 相对 S0：

```text
Precision   +0.031
Recall      -0.016
mAP50       +0.010
mAP50-95    +0.005
scallop mAP50       +0.033
scallop mAP50-95    +0.011
```

C1A 相对 N4：

```text
Precision   +0.003
Recall      +0.002
mAP50       +0.006
mAP50-95    -0.002
scallop mAP50       +0.034
scallop mAP50-95    +0.020
```

这说明严格 P3-only MSFF 确实恢复了 N4 丢失的低 IoU 检出能力和 scallop 性能，但尚未保住 N4 的高 IoU 定位收益。C1A 不是已经击败 N4 的最终模型，而是具有明确互补现象的保留候选。

C1A 相对 C1B 的 `mAP50 +0.012`、`mAP50-95 +0.008`，scallop mAP50-95 高 `0.040`。seed 0 结果支持这样的阶段性判断：在当前结构和训练协议下，MSFF 应限制在 P3 检测分支；让增强特征继续生成 P4/P5 可能扰动更高层语义和质量排序。该机制解释仍需后续多种子或特征可视化支持，但 C1B 已无继续投入的价值，不再运行其他随机种子或更长训练。

在再次训练之前，先对 C1A 的同一 `best.pt` 扫描：

```text
quality_power = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
```

该实验不更新权重，只判断质量分数融合强度是否造成当前 `mAP50-95 -0.002`。若某个固定指数使 C1A 的 mAP50-95 超过 0.664，且 mAP50/scallop 优势仍保留，则锁定该指数并运行 C1A seeds 1/2；否则不对原 C1A 直接做多种子训练，转向 P3/P4/P5 尺度自适应质量融合。

训练1执行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/C1A_strict_p3_msff_quality_e100_b96_seed0/weights/best.pt"
for power in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
    model = YOLO(weight)
    model.model.model[-1].quality_power = power
    model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        device=0,
        plots=False,
        name=f"C1A_e100_seed0_power{power:g}",
    )
PY
```

必须同时保存总体指标与 scallop 指标，不能只按单个最高 mAP50-95 选择指数。该扫描仍属于开发集上的后处理校准；正式论文协议中应在 dev_val 上选定指数，再冻结后报告 test 结果。

### 8.9 D1 质量融合指数扫描结果

| power | Precision | Recall | mAP50 | mAP50-95 | scallop mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.843 | **0.780** | 0.851 | 0.652 | 0.515 |
| 0.25 | 0.845 | 0.777 | 0.852 | 0.655 | 0.517 |
| 0.50 | 0.846 | 0.773 | 0.853 | 0.658 | 0.519 |
| 0.75 | 0.846 | 0.773 | **0.854** | 0.660 | 0.522 |
| 1.00 | **0.860** | 0.760 | **0.854** | 0.663 | 0.523 |
| 1.25 | 0.852 | 0.764 | **0.854** | **0.665** | **0.525** |

随着 power 增大，mAP50-95 和 scallop mAP50-95 持续上升，而 mAP50 保持稳定，说明质量分支主要改善高 IoU 候选排序。`power=1.25` 已满足当前晋级条件，并相对 N4 获得约 `mAP50-95 +0.001`、`mAP50 +0.006`、`Recall +0.004` 和 `scallop mAP50-95 +0.024`。

由于最优值仍位于搜索边界，且对 N4 的优势只有三位小数下的 0.001，暂不启动 seeds 1/2。追加 `power=[1.0,1.25,1.5,1.75,2.0,2.5]` 的 D1b，并打印六位小数；若 2.5 仍最优，只追加 3.0。相邻候选差值不超过 0.001 时选择较小 power。

### 8.10 D1b 高区间结果与公平性修正

| power | Precision | Recall | mAP50 | mAP50-95 | fitness | scallop mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | **0.860056** | 0.760335 | 0.853885 | 0.662783 | 0.681893 | 0.523 |
| 1.25 | 0.851814 | 0.763738 | **0.854150** | 0.665257 | 0.684147 | 0.525 |
| 1.50 | 0.845842 | **0.766495** | 0.853883 | 0.666864 | 0.685566 | 0.526 |
| 1.75 | 0.849449 | 0.764584 | 0.853760 | 0.668753 | 0.687254 | 0.527 |
| 2.00 | 0.853832 | 0.762232 | 0.853773 | 0.670497 | 0.688824 | 0.529 |
| 2.50 | 0.859187 | 0.754412 | 0.853292 | **0.673893** | **0.691832** | **0.534** |

`power=2.5` 相对 1.0 的 mAP50-95 提升为 0.011110，且 scallop mAP50-95 提升 0.011，但 Recall 低于预设 0.755 下限 0.000588。当前 `power=2.0` 是满足全部预设条件的平衡点，2.5 是最大 AP 候选。

此前将 C1A 高 power 结果与 N4 power=1.0 比较并不充分公平。质量融合指数属于两者共享的推理超参数，N4 也必须扫描 `power=[1.0,1.25,1.5,1.75,2.0,2.5,3.0]`。只有 C1A 超过“各自校准后”的 N4，才能把额外收益归因于严格 P3-only MSFF。当前暂停 seeds 1/2，同时由训练1补测 C1A power=3.0、训练2执行 N4 公平扫描。

### 8.11 C1A power=3.0 边界结果

| Precision | Recall | mAP50 | mAP50-95 | fitness | scallop mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 0.869024 | 0.743898 | 0.853475 | **0.677000** | **0.694647** | **0.539** |

相对 YOLO11n seed 0，C1A power=3.0 的 mAP50-95 提升 0.020000、mAP50 提升 0.009475、Precision 提升 0.038024，scallop mAP50-95 提升 0.029；Recall 下降 0.034102。因此保留 `power=3.0` 作为最大 AP 候选，保留 `power=2.0` 作为均衡候选。

不再扫描高于 3.0 的指数。停止原因不是 mAP 已出现峰值，而是 Recall 已下降至 0.743898，且继续使用同一评价划分搜索会加剧后处理参数过拟合。最终 power 必须等待 N4 同区间扫描后决定。

### 8.12 N4 与 C1A 的公平校准结论

| power | N4 mAP50-95 | C1A mAP50-95 | C1A - N4 |
|---:|---:|---:|---:|
| 1.00 | 0.664253 | 0.662783 | -0.001470 |
| 1.25 | 0.666306 | 0.665257 | -0.001049 |
| 1.50 | 0.668597 | 0.666864 | -0.001733 |
| 1.75 | 0.670932 | 0.668753 | -0.002179 |
| 2.00 | 0.672657 | 0.670497 | -0.002160 |
| 2.50 | 0.675928 | 0.673893 | -0.002035 |
| 3.00 | **0.678852** | 0.677000 | -0.001852 |

结论：C1A 的 mAP50-95 增长主要来自 QualityDetect 共享的 power 校准，而不是 MSFF 带来的额外总体 AP。C1A 在所有相同 power 下均未超过 N4，因此停止 C1A seeds 1/2；但 C1A 在 `power=3.0` 的 mAP50 比 N4 高 0.008952，scallop mAP50-95 高 0.013，可作为“MSFF 改善稀有小目标但轻微损害总体高 IoU AP”的结构消融。

N4-QP3 seed 0 的最终校准结果为：`P=0.868320`、`R=0.745971`、`mAP50=0.844523`、`mAP50-95=0.678852`。相对 YOLO11n seed 0，mAP50-95 提升约 0.022。下一步无需重新训练，直接对既有 N4 seed 1/2 权重固定 `quality_power=3.0` 复验。

---

## 9. N4 质量感知检测头：当前主创新基础

### 9.1 结构

N4 在 YOLO11n 的 P3/P4/P5 检测头上增加类别无关的定位质量分支：

```text
P3/P4/P5 feature
├── box regression
├── classification
└── localization quality prediction
```

质量分支采用预测框与分配真实框之间的 IoU 作为连续监督，推理时使用：

```text
final_score = class_score × quality_score
```

最终固定：

```text
quality_loss_gain = 0.5
quality_power = 1.0
```

### 9.2 已验证贡献

N4 相对 YOLO11n 只增加约 11,747 个参数和约 0.1 GFLOPs。100 轮三种子的 mAP50-95 配对增益分别为：

```text
seed 0: +0.006
seed 1: +0.005
seed 2: +0.004
mean:   +0.005
```

### 9.3 当前边界

N4 不是最终完成的方法，原因包括：

1. mAP50 平均下降 0.001；
2. Recall 平均下降 0.023；
3. scallop mAP50-95 平均下降 0.010；
4. 独立质量预测和 IoU-aware scoring 与 GFL、VFNet、TOOD 等已有方向相关，不能声称该思想首次提出。

因此论文中的原创重点必须落在后续的水下小目标适配机制，例如尺度自适应残差质量融合或类别均衡质量学习，而不是只描述“增加一个质量分支”。

### 9.4 后续性能约束

最终算法候选至少应满足：

```text
参数增量 ≤ 5%
GFLOPs 增量 ≤ 5%
mAP50-95 三种子平均提升 ≥ 1.0 个百分点（目标值）
mAP50 不低于 baseline
Recall 下降不超过 0.5 个百分点
scallop AP 不出现稳定退化
```

---

## 10. Agent 的自动验证流水线

每个结构变体必须按以下顺序执行。

### Step 1：静态导入测试

```bash
python -c "from ultralytics.nn.modules import MSFF, QualityDetect; print('import ok')"
```

### Step 2：张量形状测试

要求 Agent 自动生成测试：

```python
x = torch.randn(2, C, 80, 80, device="cuda")
y = msff(x)
assert y.shape == x.shape
assert torch.isfinite(y).all()
```

并测试反向传播：

```python
loss = y.mean()
loss.backward()
```

### Step 3：模型构建测试

```bash
yolo detect train model=<yaml> data=<duo_dev.yaml> epochs=1 imgsz=640 batch=8 device=0 workers=2 \
  quality_loss_gain=0.5 rcqfl=False sqr=False
```

检查：

- YAML 索引是否正确；
- Detect 输入层是否正确；
- 参数量是否合理；
- 最后一层是否为 `QualityDetect`；
- `quality_power` 是否为 1.0；
- C1A/C1B 的 layer 18 来源是否正确；
- 是否存在未使用参数；
- 是否出现 NaN/Inf。

### Step 4：小规模过拟合测试

抽取 32–64 张训练图像，训练 20–30 epoch。

预期：

```text
loss 明显下降，训练集 mAP 显著上升
```

若无法过拟合小数据，优先检查实现，不进入完整训练。

### Step 5：性能 Profiling

记录：

```text
Params
GFLOPs
峰值 GPU 显存
单张 latency（batch=1）
吞吐量（统一 batch）
```

### Step 6：50 epoch 筛选训练

只在前五步通过后执行。

### Step 7：统一验证与报告生成

Agent 必须从结果目录自动提取：

```text
best epoch
P / R
mAP50
mAP50:95
每类 AP50 / AP50:95 / Recall
Params / GFLOPs / latency / GPU memory
Quality loss 曲线
```

进入正式阶段后，再执行 100 epochs、seeds 0/1/2。不得直接对所有候选运行三种子，避免浪费算力和扩大 test 选择偏差。

---

## 11. 实验保留与淘汰标准

### 11.1 开发阶段保留标准

普通单模块相对同协议 S0 baseline，满足以下任一条件：

```text
条件 A：mAP50:95 提升 ≥ 0.3 个百分点，且 mAP50 不下降超过 0.2 个百分点
条件 B：mAP50 提升 ≥ 0.5 个百分点，且 mAP50:95 不下降
条件 C：总体 mAP 基本持平，但 scallop AP50 提升 ≥ 1.5 个百分点且 Recall 提升 ≥ 2 个百分点
```

C1 是组合实验，标准必须高于单模块：

```text
mAP50-95 ≥ 0.662
mAP50 ≥ 0.842
Recall ≥ 0.755
scallop mAP50-95 ≥ 0.510
并且主要指标超过同 seed 的 N4，而不只是超过 B1/MSFF
```

同时要求：

```text
参数增量与 GFLOPs 增量符合设计目标
没有类别持续明显退化
没有 OOM / CPU fallback
```

### 11.2 淘汰标准

```text
mAP50:95 下降 > 0.5 个百分点
或 mAP50 下降 > 0.8 个百分点
或弱类别 AP 明显下降
或正式三种子结果方向不一致且方差过大
或计算量增加明显但精度无收益
```

### 11.3 SCI 论文最终目标

开发阶段小幅提升只能说明方向可行。正式投稿建议争取：

```text
DUO：mAP50:95 三种子平均提升约 1.0 个百分点或以上（目标，不是录用硬门槛）
同时 mAP50、Precision、Recall 至少不出现明显牺牲
第二数据集同方向提升
三个 seed 的提升超过随机波动
```

不应把固定阈值当作录用保证；论文质量还取决于创新性、实验完整性和期刊定位。

---

## 12. C1 之后的分支决策

### 12.1 如果 C1 成功

成功指 C1 同时超过同 seed 的 N4，并恢复 Recall 或 scallop AP。下一步只做：

```text
1. 在 C1A/C1B 中选择更优数据流；
2. 运行 100 epochs、seeds 0/1/2；
3. 建立 S0、B1、N4、C1 的完整二因素消融表；
4. 分析 P3/P4/P5 各尺度质量分数与真实 IoU 的相关性；
5. 若三种子平均 mAP50-95 提升仍不足 1.0 个百分点，再做一次有针对性的质量融合改进。
```

### 12.2 如果 C1 失败

停止 MSFF/GMSFF 路线，不再尝试 UFE、EMA、CBAM、CA、SimAM、SGF 或 P2 Head 的随机组合。已有实验已表明这些方向在当前协议下不能稳定提高 mAP。

转向 N4 内部改进：

#### N6-A：残差质量融合

当前 N4：

```text
score = class_score × quality_score
```

候选公式：

```text
score = class_score × [(1 - λ) + λ × quality_score]
```

先使用现有 N4 权重做 `λ ∈ {0.25, 0.50, 0.75, 1.00}` 的离线验证，不重新训练。目标是降低质量分支对真实小目标的过度压制。

#### N6-B：尺度自适应质量融合

```text
λP3 < λP4 < λP5
```

P3 小目标定位质量波动较大，使用较弱质量抑制；P4/P5 使用更强质量排序。优先采用固定少量系数或三个可学习标量，避免复杂注意力。

#### N6-C：类别均衡质量监督

只对正样本质量损失增加温和类别权重，不修改 YOLO11 原始分类损失：

```text
Lquality = wc × BCE(q, IoU)
```

目标是修复 scallop 退化。该实验必须独立于残差融合消融，不能一次加入两项后只报告最终结果。

### 12.3 继续暂停的方向

- RCQFL：三种子/单种子实验未提高 mAP；
- SQR：与 N4 组合后没有超过 N4；
- MPDIoU：已有明显退化；
- P2 Head：计算量显著增加且公平 batch 对照下未获益；
- 现成注意力模块：快速筛选未超过基准。

---

## 13. 轻量化实验何时开始

本小论文暂不把 Jetson Nano 部署作为必要贡献，但算法论文仍必须报告 Params、GFLOPs 和统一 GPU 延迟。不要把“暂缓部署”等同于“不做效率实验”。

只有在最终候选满足：

```text
mAP50 与 mAP50:95 均高于公平 baseline
```

才考虑额外 GSConv/PConv 轻量化。N4 和 C1 本身开销已经很小，若轻量化不能形成明确的精度-效率优势，可以不再增加第三个结构改动。

建议：

```text
L0：最终高精度模型
L1：只替换 Neck 中一组 Conv 为 GSConv
L2：扩大到全部 Neck
```

每一步都要比较：

```text
精度损失 / 参数下降 / GFLOPs 下降 / 实测延迟
```

不要只用 GFLOPs 推断速度；深度可分离卷积在部分 GPU 上不一定产生等比例加速。

Jetson Nano、TensorRT 和功耗测试可留到研究生毕业大论文，但投稿稿件至少保留同一 RTX 3090 下的 batch=1 latency 和吞吐量。

---

## 14. 第二公共数据集计划

当前阶段可以暂缓第二数据集，先完成 C1/N6 的算法筛选；但在正式投稿前，SCI 二区工作强烈建议至少加入 RUOD，SCI 三区也建议加入一个额外水下数据集。第二数据集属于算法泛化证据，不等同于部署实验。

### 最低方案

```text
DUO：完整消融、主对比、类别分析
RUOD：YOLO11n、N4、最终模型三组对比
```

### 更完整方案

```text
DUO：完整实验与三种子
RUOD：主要消融与对比，至少一个正式 seed
可选 UTDAC2020 / URPC2020：最终模型泛化
```

跨数据集要求：

- 使用相同方法结构；
- 不为每个数据集重新设计一套模块；
- 可调整类别数和数据路径，但避免大幅改变训练策略；
- 报告模型在不同目标尺度和背景中的行为差异。
- 不要求在第二数据集重新进行大规模超参数搜索。

---

## 15. 对比实验清单

### 必做 baseline

```text
YOLOv8n
YOLO10n（代码环境可稳定复现时）
YOLO11n
RT-DETR-R18 或同级实时检测器
N4-Full
最终模型（C1 或 N6）
```

### 水下专项对比

```text
AGS-YOLO
AOD-YOLO
AGW-YOLOv8
PSEM + SDWH 改进模型
FEFM-YOLO11
其他使用 DUO 且公开协议明确的方法
```

### 公平性规则

1. 自己复现的模型放在一个表区；
2. 文献引用结果单独标注 `reported`；
3. 不把不同输入分辨率、不同划分、不同训练轮数的结果混为公平对比；
4. 如果引用论文指标，脚注明确其 baseline、输入尺寸和数据划分。
5. 文献结果不能替代自己在同一代码、数据划分和输入尺寸下复现的 YOLO11n baseline；
6. N4/C1 的所有比较必须固定 `quality_power=1.0` 和 `quality_loss_gain=0.5`。

---

## 16. 可视化与分析清单

至少生成：

```text
1. YOLO11n vs N4 vs Final 的 PR 曲线
2. 每类别 AP 与 Recall 柱状图
3. 混淆矩阵
4. 小/中/大框 AP（可计算时）
5. 目标框面积分布
6. 复杂背景、遮挡、低对比度和密集场景检测对比
7. scallop 漏检与误检案例
8. 特征热力图或 Grad-CAM
9. 失败案例与限制分析
10. 分类置信度、质量分数与真实 IoU 的相关性散点图
11. P3/P4/P5 三尺度质量分数分布
12. 质量融合前后候选框排序变化案例
13. 低照度、模糊和浑浊模拟退化下的鲁棒性曲线
```

可视化不能只挑成功案例；至少包含一组模型仍然失败的场景。

对于质量检测头，相关性分析比普通 Grad-CAM 更重要。论文需要证明质量分数确实与定位 IoU 正相关，而不是仅依靠最终 mAP 推断机制有效。

---

## 17. Agent 实验注册表模板

创建 `experiments/registry.csv`：

```csv
experiment_id,parent_id,model_yaml,module,position,loss,imgsz,batch,epochs,seed,status,run_dir,notes
S0,,yolo11n.yaml,none,none,default,640,96,50,0,passed,/root/yolo/runs/detect/S0_yolo11n_e50_b96_seed0,baseline
N4,S0,yolo11n-quality-n4.yaml,QualityDetect,P3-P5,default+quality,640,96,50,0,passed,,quality_power=1.0
C1A,N4,yolo11n-msff-quality-c1.yaml,MSFF+QualityDetect,strict-P3,default+quality,640,96,100,0,conditional_pass,/root/yolo/runs/detect/C1A_strict_p3_msff_quality_e100_b96_seed0,mAP50-95=0.662; retain for power sweep
C1B,N4,yolo11n-msff-quality-c1-legacy.yaml,MSFF+QualityDetect,cascade-P3-P5,default+quality,640,96,100,0,stopped,/root/yolo/runs/detect/C1B_legacy_msff_quality_e100_b96_seed0,mAP50-95=0.654; do not continue
N6A,N4,yolo11n-quality-residual.yaml,ResidualQualityFusion,P3-P5,default+quality,640,96,50,0,blocked,,run only if C1 fails
N6B,N6A,yolo11n-quality-scale-aware.yaml,ScaleAwareQualityFusion,P3-P5,default+quality,640,96,50,0,blocked,,conditional
N6C,N6B,yolo11n-quality-balanced.yaml,BalancedQualityLoss,P3-P5,balanced-quality,640,96,50,0,blocked,,conditional
```

状态值：

```text
pending / running / passed / failed / blocked / invalid
```

发生 OOM fallback 的 run 标记为：

```text
invalid
```

---

## 18. 自动报告模板

每次训练结束生成：

```markdown
# 实验报告：<experiment_id>

## 训练协议
- Git commit:
- Ultralytics version:
- GPU:
- Dataset split hash:
- Model YAML:
- Epochs / batch / imgsz / optimizer / lr / seed:
- quality_power / quality_loss_gain / rcqfl / sqr:
- Pretrained transferred items:

## 复杂度
| Params | GFLOPs | Peak GPU Memory | Latency bs=1 | Throughput |
|---:|---:|---:|---:|---:|

## 总体指标
| Model | P | R | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|

## 类别指标
| Class | AP50 | AP50:95 | Recall |
|---|---:|---:|---:|

## 相对 baseline
| Metric | Baseline | Current | Delta |
|---|---:|---:|---:|

## 训练健康检查
- NaN/Inf:
- OOM/fallback:
- Best epoch:
- Last 10 epochs trend:
- Quality loss trend:
- Quality score mean/std by P3/P4/P5:
- Score-IoU correlation:

## 决策
- passed / failed / invalid
- 原因：
- 下一步：
```

N4/C1 特别需要记录质量分支统计量。若质量分数与真实 IoU 相关性很弱，说明最终涨点可能只是额外参数或训练扰动，不能充分支持质量感知机制的论文解释。

---

## 19. 推荐的完整实验队列

### 阶段 I：已完成基础工作

```text
S0  YOLO11n，50 epochs，seeds 0/1/2
F0  YOLO11n，100 epochs，seeds 0/1/2
N4  QualityDetect，50 epochs 融合指数消融
F1  N4-Full，100 epochs，seeds 0/1/2
```

### 阶段 II：当前 C1 组合筛选

```text
C1A  strict P3-only MSFF + QualityDetect，训练1
C1B  original A2 dataflow MSFF + QualityDetect，训练2
50 epochs 结果：0.640 / 0.642 mAP50-95，均未通过筛选
100 epochs 结果：C1A=0.662，C1B=0.654，C1B 终止
当前结果：N4 power=3.0 达到 0.678852，所有相同 power 下均高于 C1A
当前动作：停止 C1A seeds 1/2；使用既有 N4 seed 1/2 权重复验 power=3.0
晋级条件：N4-QP3 三种子平均 mAP50-95 相对 YOLO11n 提升至少 0.015
```

### 阶段 III-A：C1 成功时

```text
选择 C1A/C1B 最优版本
100 epochs，seeds 0/1/2
完成 S0/B1/N4/C1 二因素消融
分析每尺度质量分数与真实 IoU
```

### 阶段 III-B：C1 失败时

```text
N6A  残差质量融合，先复用现有权重扫描 λ
N6B  尺度自适应质量融合
N6C  类别均衡质量监督
每次只增加一个变量
```

### 阶段 IV：最终算法证据

```text
DUO 100 epochs 三种子 mean ± std
逐模块消融
YOLOv8n/YOLO10n/YOLO11n/RT-DETR 对比
类别级与尺度级 AP
置信度-IoU 相关性
低照度/模糊/浑浊退化测试
同 RTX 3090 Params/GFLOPs/latency
```

### 阶段 V：投稿前补齐

```text
RUOD：YOLO11n、N4、Final
整理代码、配置、随机种子和结果表
绘制网络结构图、PR曲线和机制图
Jetson Nano 部署留到毕业大论文
```

---

## 20. 代码 Agent 的直接任务说明

可把下面内容作为 Agent 的顶层任务：

```text
目标：在 Ultralytics 8.3.185 的 YOLO11n 上完成面向 DUO 水下小目标的质量感知检测改进，优先解决 N4 提高 mAP50-95 但降低 Recall、mAP50 和 scallop AP 的问题。

执行原则：
1. 使用 DUO dev_train/dev_val 做模块筛选，官方 test 保留到最终评价。
2. 在两台 RTX 3090 上统一 batch=96、imgsz=640、optimizer=auto、epochs=50、seed=0 进行开发筛选。
3. 正式实验使用 epochs=100、seeds=0/1/2，并报告 mean ± std。
4. C1A/C1B 已完成 100 轮诊断；C1A 保留，C1B 终止，禁止继续训练 C1B。
5. C1A/C1B 只能有 layer18 的 from 不同，其他结构、训练参数和预训练迁移数量必须一致。
6. C1 只有超过同 seed N4 才能晋级；仅恢复 Recall 或仅超过 MSFF 不算成功。
7. C1A 先做免训练 `quality_power` 扫描，只有超过同 seed N4 才运行 seeds 1/2；若失败，转向尺度自适应质量融合，不再扩大 MSFF 结构。
8. 每个实验先完成导入、模型构建、前向/反向、1 epoch smoke test和 profiling，再进行50 epoch训练。
9. 自动记录 results.csv、args.yaml、模型摘要、预训练迁移数量和实验报告。
10. 发现 OOM、TaskAlignedAssigner CPU fallback、NaN 或数据划分变化时，将实验标记 invalid，不参与比较。
11. 不修改 Loss、数据增强或其他超参数，除非实验编号明确要求。
12. 以 mAP50:95 为主指标，同时记录 mAP50、Precision、Recall、每类 AP 和每尺度质量统计，重点关注 scallop。
13. 不使用失败模块继续盲目组合，不根据单次 test 小数点波动包装创新。
```

---

## 21. 论文写作层面的最低证据链

一篇较完整的改进 YOLO SCI 论文应至少包含：

1. **问题证据**：框尺寸、类别分布、弱类别和典型失败场景；
2. **主创新模块**：结构图、公式、设计动机；
3. **单模块消融**：证明主创新独立有效；
4. **组合消融**：证明辅助模块互补而非堆叠；
5. **双数据集验证**：DUO + RUOD；
6. **多 baseline 对比**：同协议复现与文献结果分开；
7. **复杂度与速度**：Params、GFLOPs、GPU latency、峰值显存；
8. **多随机种子**：mean ± std；
9. **类别级结果**：突出 scallop 等难类，但不隐藏其他类别退化；
10. **机制证据**：质量分数与真实 IoU 的相关性、融合前后排序变化和尺度级统计；
11. **鲁棒性证据**：低照度、模糊、浑浊条件下 baseline 与最终模型的相对变化；
12. **可视化与失败分析**：成功和失败都展示；
13. **可复现材料**：配置、划分、随机种子和代码 commit。

本文以算法改进为主，不要求加入 Jetson Nano 实机部署章节；但 Params、GFLOPs、统一 RTX 3090 延迟和吞吐量仍属于算法论文的必要证据。

---

## 22. 最终决策建议

当前项目不再以 GMSFF 或注意力模块筛选为主线。最短且逻辑完整的路线是：

```text
C1A/C1B 回答 MSFF 与 N4 是否互补
→ 成功：三种子确认并形成“特征增强 + 质量排序”完整方法
→ 失败：停止 MSFF，直接改进 N4 的融合和类别均衡机制
→ 最终模型达到稳定 mAP50-95 增益且不明显损害 Recall/scallop
→ 补齐机制分析、对比模型、第二数据集和同GPU效率
→ 投稿算法小论文；部署留到毕业大论文
```

对当前项目，优先级建议为：

```text
C1A strict P3-only MSFF + N4
≈ C1B original A2 dataflow MSFF + N4
> N6-A residual quality fusion
> N6-B scale-aware quality fusion
> N6-C class-balanced quality supervision
> 第二数据集与机制分析
> 额外轻量化（条件执行）
```

当前 N4 的 `mAP50-95 +0.005` 是稳定但偏小的阶段成果，不足以支持“全面提升”表述。最终论文应争取三种子平均 mAP50-95 提升约 0.010 或以上，同时让 mAP50、Recall 和 scallop AP 至少接近 baseline。若无法达到，应降低目标期刊档次或重新定义贡献为“高 IoU 定位质量与低开销权衡”，不能选择性忽略负指标。

最重要的不是训练更多模型，而是确保每个实验回答一个清楚问题，并且所有结论都能由公平、可复现的数据支持。

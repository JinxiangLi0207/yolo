# N4 轻量定位质量感知检测头论文创新点记录

## 1. 文档用途

本文档记录 N4 实验的研究动机、方法设计、实现位置、实验结果和论文写作边界，供后续撰写方法章节、消融实验章节和结果分析章节使用。

当前核心结论基于 DUO 数据集、YOLO11n、50 epochs、batch=96 的三随机种子实验。100 epochs、seed 0 的 Power 1 正式验证已完成，mAP50-95 相对同配置基准提高 0.007；仍需补充 100 epochs 的 seeds 1/2，才能形成论文最终主表。

---

## 2. 暂定名称

中文名称：轻量定位质量感知检测头

英文名称：Lightweight Localization-Quality-Aware Detection Head

建议缩写：LQH（Lightweight Quality Head）

当前实验编号：N4 / N4-Full

最终版本采用：

```text
quality_power = 1.0
final_score = class_score * quality_score
```

---

## 3. 研究背景与问题

DUO 水下目标检测存在以下特点：

1. 目标尺寸偏小，弱目标、模糊目标和密集目标较多。
2. 水体浑浊、光照不足和复杂海底纹理容易产生低质量候选框。
3. YOLO11n 的最终排序主要依赖分类置信度，但分类置信度不一定能准确反映边界框定位质量。
4. 此前 MSFF、SQR、RCQFL 等实验多次提高 Recall，却没有稳定提高 mAP，说明模型能够产生更多候选框，但低质量框的置信度排序和定位质量没有得到有效控制。

因此，当前问题不只是“能否检测到目标”，还包括：

```text
如何让定位准确的候选框获得更高置信度，
并降低定位不准或背景误检框在预测列表中的排序。
```

---

## 4. 核心思想

N4 在 YOLO11n 原有分类分支和回归分支之外，为 P3、P4、P5 每个检测尺度增加一个轻量定位质量分支。

```text
输入特征 F_l
   |-- 回归分支 ------> 边界框 b_l
   |-- 分类分支 ------> 分类概率 p_l
   `-- 质量分支 ------> 定位质量 q_l

最终置信度：s_l = p_l * q_l
```

其中，质量分支学习预测当前候选框与匹配真实框之间的 IoU。推理时，分类概率和定位质量相乘，使定位质量参与最终置信度排序。

该设计不改变 YOLO11n 的 backbone 和 neck，仅对 Detect head 进行轻量扩展。

---

## 5. 轻量质量分支

对于第 `l` 个检测尺度的输入特征 `F_l`，质量分支可以表示为：

```text
z_l = Conv1x1(Conv1x1(DWConv3x3(F_l)))
q_l = sigmoid(z_l)
```

设计特点：

- 使用深度可分离卷积降低参数量和计算量。
- 每个空间位置只输出一个类别无关的质量分数。
- 质量分数与所有类别的分类分数共享，避免为每个类别重复预测质量。
- P3、P4、P5 分别预测质量，适应不同尺度目标。

质量分支的初始 bias 使用 0.01 稀疏目标先验：

```text
bias_q = log(0.01 / 0.99)
```

这样可以防止训练初期大量背景位置产生过大的质量损失。

---

## 6. IoU 软质量监督

TaskAlignedAssigner 完成正负样本分配后，对前景位置计算预测框与匹配真实框的 IoU：

```text
q_i* = IoU(stop_gradient(b_i), b_i_gt),  i in foreground
q_i* = 0,                              i in background
```

其中：

- `b_i` 是当前预测框。
- `b_i_gt` 是分配给该位置的真实框。
- IoU 目标停止梯度，不通过质量目标反向影响边界框解码。
- 质量目标是 `[0, 1]` 连续软标签，而不是简单的 0/1 标签。

连续 IoU 标签使质量分支学习“框有多准”，而不仅是“该位置是否存在目标”。

---

## 7. 质量损失

正样本使用 TaskAlignedAssigner 的匹配质量作为权重；背景位置使用类似 Varifocal Loss 的难负样本权重：

```text
w_i = target_score_i,                 i in foreground
w_i = 0.75 * sigmoid(z_i)^2,          i in background
```

质量损失为：

```text
L_quality = sum(w_i * BCEWithLogits(z_i, q_i*)) / sum(target_scores)
```

总损失为：

```text
L = 7.5 * L_box
  + 0.5 * L_cls
  + 1.5 * L_dfl
  + lambda_q * L_quality
```

当前设置：

```text
lambda_q = quality_loss_gain = 0.5
```

为了兼容现有 Ultralytics 训练日志，`L_quality` 被合并到 `cls_loss` 项中显示，但其计算逻辑独立于分类 BCE。

---

## 8. 质量感知置信度融合

通用融合形式为：

```text
s_i,c = p_i,c * q_i^eta
```

其中：

- `p_i,c` 是位置 `i` 对类别 `c` 的分类概率。
- `q_i` 是类别无关的定位质量概率。
- `eta` 是质量融合指数，即代码中的 `quality_power`。

消融实验测试了：

```text
eta = 0.0：不使用质量分数，仅保留质量辅助训练
eta = 0.5：class_score * sqrt(quality_score)
eta = 1.0：class_score * quality_score
```

最终采用 `eta=1.0`，因为它在三个随机种子上都获得最高的 mAP50-95。

---

## 9. 实现位置

主要代码文件：

| 文件 | 作用 |
|---|---|
| `ultralytics/nn/modules/head.py` | 实现 `QualityDetect` 和质量分支 |
| `ultralytics/nn/modules/__init__.py` | 导出 `QualityDetect` |
| `ultralytics/nn/tasks.py` | 注册模块、解析 YAML、初始化 stride |
| `ultralytics/utils/loss.py` | 生成 IoU 质量标签并计算质量损失 |
| `ultralytics/cfg/default.yaml` | 注册 `quality_loss_gain` |
| `ultralytics/cfg/models/11/yolo11n-quality-n4.yaml` | N4 模型结构 |

最终正式 YAML 应设置：

```yaml
- [[16, 19, 22], 1, QualityDetect, [nc, 1.0]]
```

---

## 10. 三随机种子基准结果

统一设置：

```text
epochs=50
batch=96
imgsz=640
pretrained=yolo11n.pt
cache=True
deterministic=True
seeds=[0, 1, 2]
```

YOLO11n 基准结果：

| Seed | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0 | 0.875 | 0.738 | 0.842 | 0.650 |
| 1 | 0.834 | 0.765 | 0.836 | 0.648 |
| 2 | 0.851 | 0.753 | 0.837 | 0.651 |
| **Mean** | **0.853** | **0.752** | **0.838** | **0.650** |
| **Std** | **0.021** | **0.014** | **0.0032** | **0.0015** |

该结果说明 Precision 和 Recall 波动较大，论文应以三种子平均 mAP 为主要判断依据。

---

## 11. 融合指数消融实验

所有融合消融均使用相同的 N4 权重和相同独立验证配置，仅在推理前修改 `quality_power`。

### 11.1 整体结果

| Seed | Power 0 mAP50 / mAP50-95 | Power 0.5 mAP50 / mAP50-95 | Power 1 mAP50 / mAP50-95 |
|---:|---:|---:|---:|
| 0 | 0.844 / 0.650 | 0.845 / 0.655 | 0.844 / **0.659** |
| 1 | 0.841 / 0.649 | 0.843 / 0.654 | 0.843 / **0.659** |
| 2 | 0.846 / 0.654 | 0.848 / 0.659 | 0.848 / **0.663** |
| **Mean** | **0.844 / 0.651** | **0.845 / 0.656** | **0.845 / 0.660** |

### 11.2 相对基准提升

最终 Power 1 与对应随机种子基准的配对比较：

| Seed | Baseline mAP50-95 | N4-Full mAP50-95 | 提升 |
|---:|---:|---:|---:|
| 0 | 0.650 | 0.659 | **+0.009** |
| 1 | 0.648 | 0.659 | **+0.011** |
| 2 | 0.651 | 0.663 | **+0.012** |
| **Mean** | **0.650** | **0.660** | **+0.0107** |

N4-Full 的三种子平均结果：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n | 0.853 | 0.752 | 0.838 | 0.650 |
| N4-Full | 0.855 | 0.756 | **0.845** | **0.660** |
| 提升 | +0.002 | +0.004 | **+0.0067** | **+0.0107** |

结论：

1. Power 0 相比基准略有提升，说明辅助质量监督对共享特征有一定帮助。
2. Power 0.5 明显提升 mAP50-95，说明质量分数融合能够改善预测排序。
3. Power 1 在三个随机种子上均取得最高 mAP50-95，说明完整乘法融合最有效。

---

## 12. 各类别结果

三种子平均 AP 对比：

| 类别 | Baseline mAP50 | N4-Full mAP50 | 提升 | Baseline mAP50-95 | N4-Full mAP50-95 | 提升 |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.855 | 0.855 | 0.000 | 0.623 | 0.629 | **+0.006** |
| echinus | 0.919 | 0.922 | **+0.003** | 0.741 | 0.755 | **+0.014** |
| scallop | 0.657 | 0.680 | **+0.023** | 0.498 | 0.513 | **+0.015** |
| starfish | 0.922 | 0.923 | +0.001 | 0.737 | 0.744 | **+0.008** |

scallop 和 echinus 的 mAP50-95 提升最明显。尤其 scallop 的 mAP50 和 mAP50-95 同时提高，说明 N4 不只是改变单一置信度阈值下的 Precision/Recall，而是改善了整条 PR 曲线和高质量框排序。

---

## 13. 100 epochs 阶段性正式结果

### 13.1 实验设置

```text
epochs=100
batch=96
imgsz=640
seed=0
deterministic=True
quality_loss_gain=0.5
quality_power=0.5
```

注意：该实验训练时仍使用原 YAML 中的 `quality_power=0.5`。融合指数消融已经确定最终 N4-Full 应采用 `quality_power=1.0`，因此本节结果不是最终 N4-Full 结果。

### 13.2 整体结果

同配置 100 epochs 基准为 A8-0 YOLO11n：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n-100 | 0.831 | **0.778** | 0.844 | **0.657** |
| N4-100，Power 0.5 | **0.863** | 0.759 | 0.847 | 0.657 |
| N4-Full-100，Power 1 | 0.859 | 0.760 | **0.848** | **0.664** |
| N4-Full 相对基准 | **+0.028** | -0.018 | **+0.004** | **+0.007** |

Power 0.5 只提高 mAP50，mAP50-95 与基准持平；Power 1 将 mAP50-95 提高到 0.664，验证了完整质量乘法融合在 100 epochs 下仍然有效。

### 13.3 各类别结果

| 类别 | Baseline mAP50 | N4-Full mAP50 | 变化 | Baseline mAP50-95 | N4-Full mAP50-95 | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.858 | 0.873 | **+0.015** | 0.630 | 0.647 | **+0.017** |
| echinus | 0.921 | 0.925 | **+0.004** | 0.746 | 0.760 | **+0.014** |
| scallop | 0.671 | 0.670 | -0.001 | 0.510 | 0.501 | **-0.009** |
| starfish | 0.926 | 0.925 | -0.001 | 0.742 | 0.748 | **+0.006** |

holothurian、echinus 和 starfish 的 mAP50-95 均得到改善，其中 holothurian 和 echinus 提升最明显。scallop 相对 Power 0.5 已有所恢复，但 mAP50-95 仍比基准低 0.009。该现象与 50 epochs 多种子实验中 scallop 的稳定提升不一致，论文不能声称所有类别全面提升。

Power 1 相对 Power 0.5 的变化：

| 类别 | mAP50 变化 | mAP50-95 变化 |
|---|---:|---:|
| all | +0.001 | **+0.007** |
| holothurian | +0.001 | +0.005 |
| echinus | +0.001 | **+0.009** |
| scallop | +0.003 | +0.005 |
| starfish | +0.001 | **+0.008** |

### 13.4 当前判断

1. Power 1 在 100 epochs、seed 0 下将 mAP50-95 从 0.657 提高到 0.664，达到正式晋级条件。
2. Power 1 相对 Power 0.5 的 mAP50-95 提高 0.007，进一步证明质量置信度融合是主要增益来源。
3. 当前结果仍然只有一个 100 epochs 随机种子，不能直接作为最终统计结论。
4. scallop 在 100 epochs 下仍低于基准，需要通过 seeds 1/2 判断这是随机波动还是稳定退化。
5. 下一步应补充 N4-Full-100 的 seeds 1/2，而不是继续添加新模块。

本次 Power 1 验证命令：

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("/root/yolo/runs/detect/N4_yolo11n_quality_e100_b96_seed0/weights/best.pt")
model.model.model[-1].quality_power = 1.0
model.val(
    data="ultralytics/cfg/datasets/DUO.yaml",
    batch=192,
    imgsz=640,
    device=0,
    plots=True,
    name="N4_e100_seed0_power1",
)
PY
```

这里使用 `batch=192`，是为了与训练结束时 `6/6` 的验证 batch 数保持一致。

结果目录：

```text
/root/yolo/runs/detect/N4_e100_seed0_power1
```

---

## 14. 模型复杂度

| 模型 | Params | GFLOPs | 参数变化 | 计算量变化 |
|---|---:|---:|---:|---:|
| YOLO11n | 2,582,932 | 6.3 | - | - |
| N4 | 2,594,679 | 6.4 | +11,747（约 +0.45%） | +0.1（约 +1.6%） |

当前 PyTorch 验证日志中，N4 推理时间约比 YOLO11n 增加 0.1 ms，但日志精度较低且不同服务器波动明显，不能作为论文最终速度结论。

论文最终需要在以下统一环境中重新测试：

- 相同 GPU、相同 batch、相同 warmup 和重复次数。
- TensorRT FP16。
- Jetson Nano 实际端到端延迟、FPS、显存和功耗。

---

## 15. 失败组合与方法边界

### 15.1 N5：QualityDetect + SQR

Seed 0 结果：

| 模型 | mAP50 | mAP50-95 | scallop mAP50 | scallop mAP50-95 |
|---|---:|---:|---:|---:|
| N4 | 0.845 | 0.655 | 0.685 | 0.515 |
| N5 | 0.845 | 0.655 | 0.679 | 0.509 |

SQR 没有进一步提高整体 mAP，并使 scallop AP 下降，因此 N5 淘汰。论文最终模型不包含 SQR。

### 15.2 不建议继续组合的模块

- MSFF：当前公平协议下 mAP50-95 明显下降。
- SGF：显著伤害 Recall 和 scallop AP。
- RCQFL：只移动 Precision/Recall，没有提高 mAP。
- P2 四检测头：计算量增加明显且 AP 下降。

这些结果表明，N4 的优势来自质量感知排序，而不是简单堆叠更多增强模块。

---

## 16. 可用于论文的贡献表述

中文草稿：

> 针对水下复杂背景中分类置信度与边界框定位质量不一致的问题，本文设计了一种轻量定位质量感知检测头。该检测头在 YOLO11n 多尺度预测分支上增加类别无关的质量估计支路，并采用预测框与匹配真实框之间的 IoU 作为连续监督信号。在推理阶段，分类概率与定位质量分数联合构成最终检测置信度，从而降低定位不准确候选框的排序，并提高弱小目标的高 IoU 检测性能。该方法仅引入约 0.45% 参数量和 1.6% 计算量，在 DUO 数据集三随机种子实验中将 mAP50-95 平均提高约 1.07 个百分点。

英文草稿：

> To alleviate the mismatch between classification confidence and localization accuracy in complex underwater scenes, we introduce a Lightweight Localization-Quality-Aware Detection Head. A class-agnostic quality estimation branch is attached to each prediction scale of YOLO11n and is supervised by the IoU between the predicted box and its assigned ground-truth box. During inference, the classification probability is multiplied by the predicted localization quality to form the final confidence score, suppressing poorly localized candidates in the ranking. The proposed head adds only approximately 0.45% parameters and 1.6% FLOPs, while improving the three-seed mean mAP50-95 by about 1.07 percentage points on DUO.

---

## 17. 学术表述边界

定位质量预测、IoU-aware confidence、Generalized Focal Loss 和 Varifocal Loss 等方向已有相关研究。因此，在完成系统文献检索前，不应使用以下表述：

```text
首次提出定位质量预测
首次将 IoU 用于置信度校准
完全原创的质量感知检测思想
```

更稳妥的创新表述是：

```text
面向水下弱小目标的轻量化质量感知 YOLO11n 检测头
适配多尺度水下目标的类别无关质量分支
低参数开销的质量监督与置信度联合排序方案
针对 DUO 弱类和高 IoU 指标的系统消融与部署验证
```

最终是否具有足够新颖性，需要与 IoU-Net、GFL、VFNet、TOOD 及近期水下质量感知检测工作进行对比。

---

## 18. 后续必须完成的实验

### 18.1 正式训练

- YOLO11n：100 epochs，seed 0 已完成，仍需补充 seeds 1/2。
- N4 Power 0.5：100 epochs、seed 0 已完成，mAP50-95 与基准持平。
- N4-Full Power 1：100 epochs、seed 0 已完成，mAP50-95 从 0.657 提高到 0.664。
- N4-Full：需要补充 100 epochs 的 seeds 1/2，并报告平均值和标准差。
- 报告平均值和标准差。

### 18.2 方法消融

- Baseline：无质量分支。
- Power 0：质量辅助训练，不进行质量融合。
- Power 0.5：平方根融合。
- Power 1：完整乘法融合。
- 可选：`quality_loss_gain` 的少量消融，但不要在测试集上反复搜索最优值。

### 18.3 可视化

- Baseline 与 N4 的 PR 曲线。
- 混淆矩阵。
- scallop、低照度、浑浊和密集场景的检测对比图。
- 质量分数热力图或候选框质量排序案例。
- 典型误检、漏检和定位不准案例。

### 18.4 泛化实验

- 至少增加一个水下数据集，如 URPC2020、RUOD 或 UTDAC2020。
- 与 YOLOv5n、YOLOv8n、YOLO10n、YOLO11n 等轻量模型比较。

### 18.5 部署实验

- ONNX 导出正确性。
- TensorRT FP16 精度和速度。
- Jetson Nano FPS、单帧延迟、显存和功耗。
- 确认质量分支及分数融合能够正确导出。

---

## 19. 数据划分注意事项

当前训练过程使用 DUO test 作为 Ultralytics 的验证集，并根据该集合选择 `best.pt`。这是许多现有 DUO 实验采用的工程流程，但严格论文评审可能认为存在测试集参与模型选择的问题。

更严谨的方案是：

1. 从 DUO train 中划分训练集和内部验证集。
2. 使用内部验证集选择 epoch 和超参数。
3. 最终只在官方 test 上评估一次。

如果继续采用现有 DUO 协议，应确保所有对比方法使用完全相同的数据划分和模型选择方式，并在论文中明确说明。

---

## 20. 当前结论

N4 是目前唯一通过三随机种子验证、能够稳定提高整体 mAP50 和 mAP50-95 的改进：

```text
三种子平均 mAP50：     0.838 -> 0.845，提升约 0.67 个百分点
三种子平均 mAP50-95：  0.650 -> 0.660，提升约 1.07 个百分点
参数量增加：           约 0.45%
GFLOPs 增加：          约 1.6%
```

当前建议将 N4-Full 作为论文主模型的第一个核心创新点，正式配置固定为 `quality_power=1.0`。50 epochs 三种子结果已经证明该方向稳定有效；100 epochs、seed 0 的 Power 1 结果进一步将 mAP50-95 从 0.657 提高到 0.664。仍需完成 100 epochs 的 seeds 1/2、多数据集和 Jetson Nano 部署实验，并谨慎处理 scallop 在 100 epochs 下的退化问题，才能给出最终投稿结论。

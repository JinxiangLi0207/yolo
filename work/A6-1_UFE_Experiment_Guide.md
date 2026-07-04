# A6-1 实验指导文档：YOLO11n + UFE(P3)

## 0. 实验背景

当前第一阶段已经完成，A0 YOLO11n 被确定为基准模型：

| 实验 | 模型 | mAP@50 | mAP@50:95 | Recall | GFLOPs | 结论 |
|---|---|---:|---:|---:|---:|---|
| A0 | YOLO11n | 0.849 | 0.656 | 0.762 | 6.3 | 基准模型 |
| A2-Lite | YOLO11n + MSFF(P3) | 0.848 | 0.651 | 0.778 | 6.4 | 当前最有价值的中间模型 |
| A5-Lite | YOLO11n + SPDConv(P2) + MSFF(P3) | 0.833 | 0.638 | 0.743 | 7.1 | 组合失败 |

已有实验说明：

- `MSFF(P3)` 对小目标分支有价值，主要提升 Recall。
- `SPDConv` 有小目标 Recall 信号，但会带来 mAP 损失，且与 MSFF 组合后失败。
- `MPDIoU` 对 DUO 小目标不适用，必须放弃。

A6-1 的目标是验证一个新的水下特征增强模块 `UFE` 是否能在 YOLO11n 上单独带来稳定收益。

---

## 1. 实验编号与模型定义

| 项目 | 内容 |
|---|---|
| 实验编号 | A6-1 |
| 模型名称 | YOLO11n + UFE(P3) |
| 数据集 | DUO |
| 基准模型 | A0 YOLO11n |
| 修改位置 | Head 中 P3/8 小目标检测分支后 |
| 新增模块 | UFE, Underwater Feature Enhancement |
| Loss | 原始 YOLO11 Loss |
| Epochs | 100 |
| Batch | 16，若显存不足改为 8 |
| imgsz | 640 |
| Optimizer | SGD/auto，优先保持与 A0 一致 |
| Pretrained | yolo11n.pt |

模型配置文件：

```text
ultralytics/cfg/models/11/yolo11n-ufe-a6-1.yaml
```

模块实现文件：

```text
ultralytics/nn/Attmodules/UFE.py
```

---

## 2. UFE 模块设计

UFE 的论文定位不是“堆注意力模块”，而是：

```text
面向水下机器人视觉退化场景的轻量残差特征增强模块。
```

水下图像常见问题包括低照度、浑浊散射、低对比度背景、小目标纹理弱。UFE 围绕这些问题设计三个轻量分支：

| 分支 | 作用 | 实现 |
|---|---|---|
| 局部细节增强 | 提取小目标边缘、纹理和弱局部响应 | `DWConv 3x3 + DWConv kxk` |
| 通道重标定 | 强调有效目标通道，抑制水体背景通道 | GAP/GMP + 1x1 Conv |
| 空间弱目标门控 | 突出疑似小目标区域 | avg/max map + spatial conv |

残差形式：

```text
Y = X + gamma * UFE_enhance(X)
```

其中 `gamma` 是可学习参数，初始化为 `1e-3`，用于避免训练初期破坏 YOLO11n 原始特征分布。

---

## 3. A6-1 插入位置

A6-1 只在 P3 小目标分支插入 UFE：

```yaml
- [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)
- [-1, 1, UFE, []]            # 17 A6-1: underwater feature enhancement on P3
```

最终检测头使用：

```yaml
- [[17, 20, 23], 1, Detect, [nc]]
```

选择 P3 的原因：

- DUO 中小目标占比较高。
- A2-Lite 已经证明 P3 分支增强比 P4/P5 全分支增强更有效。
- P3 插入对 Jetson Nano 部署成本相对可控。

---

## 4. 云服务器训练前检查

进入仓库目录：

```bash
cd /path/to/yolov11-attention
```

拉取最新代码：

```bash
git pull
```

安装本地魔改版 Ultralytics：

```bash
pip install -e .
```

确认 Python 使用的是当前仓库源码：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

期望输出路径指向当前 Git 仓库下的 `ultralytics` 目录。

检查 UFE 是否能被导入：

```bash
python -c "from ultralytics.nn.Attmodules import UFE; print(UFE)"
```

---

## 5. 训练命令

### 5.1 使用仓库内 DUO.yaml

如果云服务器上的 DUO 数据集路径与 `ultralytics/cfg/datasets/DUO.yaml` 一致，可以执行：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ufe-a6-1.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  pretrained=yolo11n.pt \
  name=A6_1_yolo11n_ufe_p3
```

### 5.2 使用云服务器自定义 DUO.yaml

如果数据集放在独立路径，建议使用云端自己的数据配置文件：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ufe-a6-1.yaml \
  data=/path/to/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  pretrained=yolo11n.pt \
  name=A6_1_yolo11n_ufe_p3
```

### 5.3 8GB 显存训练命令

如果继续在 5060 8GB 显卡上跑，建议：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ufe-a6-1.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=0 \
  pretrained=yolo11n.pt \
  name=A6_1_yolo11n_ufe_p3
```

Windows 环境建议 `workers=0`。

---

## 6. 验证命令

训练完成后验证最佳权重：

```bash
yolo detect val \
  model=runs/detect/A6_1_yolo11n_ufe_p3/weights/best.pt \
  data=ultralytics/cfg/datasets/DUO.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A6_1_yolo11n_ufe_p3_val
```

如果云端使用自定义数据配置，替换 `data=` 路径。

---

## 7. 成功标准

A6-1 不要求所有指标同时提升，但必须至少满足以下条件之一：

| 条件 | 成功标准 | 说明 |
|---|---|---|
| 主指标 1 | `mAP@50 >= 0.852` | 相比 A0 提升至少 0.003 |
| 主指标 2 | `mAP@50:95 >= 0.660` | 相比 A0 提升至少 0.004 |
| 召回路线 | `Recall >= 0.780` 且 `mAP@50 >= 0.849` | 召回提升但不能牺牲 mAP |
| 弱类路线 | `scallop Recall >= 0.590` 且 `scallop mAP@50 >= 0.684` | 提升弱类但不能只增加误检 |

复杂度约束：

| 指标 | 要求 |
|---|---|
| Params | 不超过 2.8M |
| GFLOPs | 不超过 6.8G |
| FPS | 不应明显低于 A0 |

如果 A6-1 只提升 Recall，但 `mAP@50` 或 `mAP@50:95` 明显下降，则只能作为边界结果，不适合作为最终主模型。

---

## 8. 实验记录表

训练完成后填写：

| 指标 | A0 YOLO11n | A6-1 YOLO11n+UFE | 差值 | 结论 |
|---|---:|---:|---:|---|
| Precision | 0.848 |  |  |  |
| Recall | 0.762 |  |  |  |
| mAP@50 | 0.849 |  |  |  |
| mAP@50:95 | 0.656 |  |  |  |
| Params | 2.58M |  |  |  |
| GFLOPs | 6.3 |  |  |  |
| FPS | 455 |  |  |  |

各类别指标：

| 类别 | A0 mAP@50 | A6-1 mAP@50 | 差值 | A0 Recall | A6-1 Recall | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.860 |  |  | 0.786 |  |  |
| echinus | 0.925 |  |  | 0.845 |  |  |
| scallop | 0.684 |  |  | 0.552 |  |  |
| starfish | 0.927 |  |  | 0.865 |  |  |

---

## 9. 后续决策

| A6-1 结果 | 下一步 |
|---|---|
| mAP 和 Recall 同时提升 | 保留 UFE，进入 A7：`YOLO11n + MSFF(P3) + UFE(P3)` |
| Recall 提升但 mAP 下降很小 | 保留为边界有效，尝试降低 UFE 强度或做 A7 |
| mAP 提升但 Recall 持平 | 保留 UFE，重点分析误检减少和定位质量 |
| 全面下降 | 放弃当前 UFE 结构，优先调整 UFE 插入位置或简化空间门控 |

如果 A6-1 成功，下一组建议：

```text
A7-1 = YOLO11n + MSFF(P3) + UFE(P3)
A7-2 = YOLO11n + UFE(backbone shallow)
A7-3 = YOLO11n + UFE(P3) + underwater degradation augmentation
```

---

## 10. 论文表述建议

不要把 UFE 写成“加了一个注意力模块”。建议表述为：

```text
We propose a lightweight Underwater Feature Enhancement (UFE) module for real-time
small-object perception on underwater robots. UFE integrates local detail enhancement,
channel recalibration, and spatial weak-target gating in a residual form, improving
feature robustness under low-light and turbid underwater conditions with limited
computational overhead.
```

中文表述：

```text
本文提出一种轻量级水下特征增强模块 UFE。该模块通过局部细节增强、通道退化感知重标定和空间弱目标门控，在保持实时性的同时增强低照度、浑浊和小目标密集场景下的目标特征表达。
```

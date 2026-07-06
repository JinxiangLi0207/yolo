# A3-GMSFF 实验指导文档：YOLO11n + Gated-MSFF(P3)

## 0. 实验背景

当前实验结论：

| 实验 | 模型 | Precision | Recall | mAP@50 | mAP@50:95 | 结论 |
|---|---|---:|---:|---:|---:|---|
| A0 | YOLO11n | 0.867/0.848 | 0.753/0.762 | 0.848/0.849 | 0.658/0.656 | 稳定 baseline |
| A2-Lite | YOLO11n + MSFF(P3) | 0.842 | 0.778 | 0.848 | 0.651 | Recall 提升，mAP 未提升 |
| A6-1/A6-2 | YOLO11n + UFE 系列 | 最高 Recall 0.779 | mAP@50:95 均下降 | - | - | 不推荐继续 |

A2-Lite 已经证明：`MSFF(P3)` 能增强小目标分支并提升 Recall。  
但 A2-Lite 的问题也很清楚：Precision 和 mAP@50:95 没有提升，说明 MSFF 可能同时增强了水下复杂背景响应。

A3-GMSFF 的目标是把 MSFF 改造成自己的受控版本：

```text
原始 P3 特征
↓
MSFF 多尺度增强特征
↓
门控权重 gate 自适应控制增强强度
↓
输出 = 原始特征 + alpha * gate * MSFF特征
```

核心论文叙事：

```text
原始 MSFF 直接增强多尺度特征，容易同时增强水下复杂背景。
本文提出 Gated-MSFF，通过门控残差增强机制自适应选择目标相关信息，
在保留小目标 Recall 优势的同时抑制背景过增强，提高 Precision 和 mAP。
```

---

## 1. 实验编号与模型定义

| 项目 | 内容 |
|---|---|
| 实验编号 | A3-GMSFF |
| 模型名称 | YOLO11n + GatedMSFF(P3) |
| 数据集 | DUO |
| 基准模型 | A0 YOLO11n |
| 直接对照 | A2-Lite YOLO11n + MSFF(P3) |
| 修改位置 | Head 中 P3/8 小目标检测分支后 |
| 新增模块 | GatedMSFF |
| 模型配置 | `ultralytics/cfg/models/11/yolo11n-gmsff-a3.yaml` |
| Epochs | 100 |
| imgsz | 640 |
| Pretrained | yolo11n.pt |

---

## 2. Gated-MSFF 模块设计

### 2.1 原始 MSFF 的问题

原始 MSFF 形式近似为：

```text
MSFF(x) = x + x * multi_scale_gate(x)
```

它对 P3 小目标分支有效，但属于直接增强：

- 有助于提高 Recall。
- 也可能增强水下背景纹理、悬浮颗粒、低对比度伪目标。
- 因此 Precision 和 mAP@50:95 不一定提升。

### 2.2 Gated-MSFF 的改进

Gated-MSFF 使用可学习门控：

```text
ms_feat = x * multi_scale_gate(x)
gate = sigmoid(GateNet(concat(x, ms_feat)))
y = x + alpha * gate * ms_feat
```

其中：

| 组件 | 作用 |
|---|---|
| `multi_scale_gate` | 继承 MSFF 的 3/5/7 多尺度深度卷积信息 |
| `GateNet` | 根据原始特征和增强特征判断哪些响应值得增强 |
| `alpha` | 可学习残差强度，初始化为 0.1 |
| `x + ...` | 保留原始 P3 主路径，降低过增强风险 |

### 2.3 预期优势

相对 A2-Lite：

- 保留 Recall 提升。
- 减少背景过增强。
- Precision 应该回升。
- mAP@50 和 mAP@50:95 应该不低于 A2-Lite，理想情况下超过 A0。

---

## 3. 代码文件

模块实现：

```text
ultralytics/nn/modules/conv.py
```

新增类：

```python
class GatedMSFF(nn.Module):
    ...
```

模型 YAML：

```text
ultralytics/cfg/models/11/yolo11n-gmsff-a3.yaml
```

P3 插入位置：

```yaml
- [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)
- [16, 1, GatedMSFF, [256]]    # 17 A3-GMSFF
```

检测头：

```yaml
- [[17, 20, 23], 1, Detect, [nc]]
```

---

## 4. 云服务器训练前检查

进入仓库目录：

```bash
cd /root/yolo
```

拉取最新代码：

```bash
git pull
```

安装本地源码：

```bash
pip install -e .
```

检查模块是否能导入：

```bash
python -c "from ultralytics.nn.modules import GatedMSFF; print(GatedMSFF)"
```

检查 YAML 是否存在：

```bash
ls ultralytics/cfg/models/11/yolo11n-gmsff-a3.yaml
```

---

## 5. 训练命令

### 5.1 云服务器 RTX 3090

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-gmsff-a3.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  pretrained=yolo11n.pt \
  name=A3_yolo11n_gmsff_p3
```

如果使用云端自定义数据配置：

```bash
data=/root/datasets/DUO.yaml
```

### 5.2 8GB 显存机器

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-gmsff-a3.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=0 \
  pretrained=yolo11n.pt \
  name=A3_yolo11n_gmsff_p3
```

Windows 环境建议 `workers=0`。

---

## 6. 验证命令

```bash
yolo detect val \
  model=runs/detect/A3_yolo11n_gmsff_p3/weights/best.pt \
  data=ultralytics/cfg/datasets/DUO.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A3_yolo11n_gmsff_p3_val
```

---

## 7. 成功标准

A3-GMSFF 的目标不是单纯提高 Recall，而是解决 A2-Lite 的 Precision/mAP 不提升问题。

### 7.1 相对 A2-Lite 的成功标准

至少满足以下任一条件：

| 条件 | 标准 |
|---|---|
| Precision 回升 | `Precision > 0.842` 且 `Recall >= 0.770` |
| mAP 回升 | `mAP@50 >= 0.849` 且 `mAP@50:95 >= 0.653` |
| 综合提升 | `Recall >= 0.775` 且 `mAP@50 >= 0.848` 且 `Precision >= 0.848` |

### 7.2 相对 A0 的理想标准

理想结果：

| 指标 | 标准 |
|---|---|
| Precision | `>= 0.867`，或至少不低于 0.855 |
| Recall | `> 0.753` |
| mAP@50 | `>= 0.850` |
| mAP@50:95 | `>= 0.658` |

### 7.3 弱类 scallop 标准

| 指标 | 标准 |
|---|---|
| scallop Recall | `>= 0.570` |
| scallop mAP@50 | `>= 0.683` |
| scallop mAP@50:95 | `>= 0.504` |

如果 scallop Recall 提升但 mAP 下降，不能算真正成功。

---

## 8. 实验记录表

### 8.1 整体指标

| 实验 | 模型 | Precision | Recall | mAP@50 | mAP@50:95 | Params | GFLOPs | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A0 | YOLO11n | 0.867 | 0.753 | 0.848 | 0.658 | 2.58M | 6.3 | baseline |
| A2-Lite | YOLO11n + MSFF(P3) | 0.842 | 0.778 | 0.848 | 0.651 | 2.59M | 6.4 | Recall 高 |
| A3-GMSFF | YOLO11n + GatedMSFF(P3) |  |  |  |  |  |  |  |

### 8.2 各类别指标

| 类别 | A0 mAP@50 | A2-Lite mAP@50 | A3-GMSFF mAP@50 | A0 Recall | A2-Lite Recall | A3-GMSFF Recall |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.860 | 0.871 |  | 0.773/0.786 | 0.812 |  |
| echinus | 0.921 | 0.923 |  | 0.825/0.845 | 0.854 |  |
| scallop | 0.683 | 0.670 |  | 0.558/0.552 | 0.576 |  |
| starfish | 0.927 | 0.927 |  | 0.854/0.865 | 0.870 |  |

---

## 9. 结果解释规则

| 结果现象 | 说明 | 下一步 |
|---|---|---|
| Precision 和 Recall 同时优于 A2-Lite | GatedMSFF 有效 | 保留为主线模块 |
| Recall 接近 A2-Lite，mAP@50:95 回升 | 门控抑制背景有效 | 保留，继续调 alpha |
| Precision 回升但 Recall 掉回 A0 | 门控过强 | 降低 gate 抑制或提高 alpha |
| Recall 高但 mAP 仍低 | 仍存在低质量增强 | 尝试更小 alpha 或加入定位训练策略 |
| 全面不如 A2-Lite | GatedMSFF 不成立 | 回到 A2-Lite + 训练策略优化 |

---

## 10. 后续实验建议

如果 A3-GMSFF 有效：

```text
A3-v2 = GatedMSFF alpha=0.05
A3-v3 = GatedMSFF alpha=0.2
A3-v4 = GatedMSFF + close_mosaic=20 + cos_lr=True
```

如果 A3-GMSFF 无效：

```text
A8 = A2-Lite + close_mosaic=20 + cos_lr=True
A8-2 = A2-Lite + imgsz=800
A8-3 = A2-Lite + box/dfl 权重微调
```

---

## 11. 论文表述建议

英文：

```text
To mitigate the over-enhancement of complex underwater backgrounds caused by
direct multi-scale feature fusion, we propose a gated residual MSFF module.
The module preserves the original P3 small-object feature as the main path and
adaptively injects target-relevant multi-scale responses through a learnable
gate and residual strength.
```

中文：

```text
针对原始 MSFF 直接增强多尺度特征时可能同时放大水下复杂背景的问题，
本文提出门控残差多尺度特征融合模块 Gated-MSFF。该模块保留原始 P3
小目标特征作为主路径，并通过可学习门控和残差强度自适应注入目标相关
多尺度响应，从而在提升小目标召回的同时抑制背景过增强。
```

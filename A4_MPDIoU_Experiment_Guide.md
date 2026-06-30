# A4 实验指导文档：YOLO11n + MPDIoU Loss

## 1. 实验目标

本实验编号为 **A4**。

实验名称：

```text
A4 = YOLO11n + MPDIoU Loss
```

本实验只修改 **边界框回归损失函数**，不修改网络结构。

也就是说：

```text
不加 SPDConv
不加 MSFF
不加 GSConv
不改 Neck
不改 Head
只把 YOLO11n 默认 bbox loss 中的 IoU 计算从 CIoU 风格替换为 MPDIoU 风格
```

实验目的：

```text
验证改进 IoU Loss 是否能够提升 DUO 数据集上的定位质量，尤其是 mAP@50:95 和 scallop 类别 AP。
```

当前已有实验共同现象：

```text
A1 SPDConv、A1-OnlyP2、A2-Lite MSFF 都能不同程度提升 Recall，
但整体 mAP@50 或 mAP@50:95 没有超过 A0。
```

这说明模型可能已经能检出更多目标，但框定位质量、置信度排序或误检控制还不足。因此，A4 先验证 Loss 改进是否能弥补这个短板。

---

## 2. A4 对比基准

A4 必须与 A0 保持相同训练设置。

A0 baseline 结果：

| 模型 | mAP@50 | mAP@50:95 | Precision | Recall | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0.849 | 0.656 | 0.848 | 0.762 | 2.58M | 6.3 |

A4 的目标：

| 指标 | 目标 |
|---|---|
| mAP@50 | 尽量 ≥ 0.849，最好 ≥ 0.852 |
| mAP@50:95 | 必须重点观察，最好 ≥ 0.659 |
| scallop mAP@50 | 尽量超过 0.684，最好 ≥ 0.700 |
| scallop Recall | 不要明显低于 0.552 |
| Params / GFLOPs | 理论上与 A0 完全一致 |

A4 成功标准：

```text
满足以下任意一条，即可认为 MPDIoU 值得进入后续组合实验：

1. mAP@50:95 ≥ 0.659
2. mAP@50 ≥ 0.852
3. scallop mAP@50 ≥ 0.700
4. mAP@50:95 基本持平，但 scallop mAP 或 Recall 明显提升
```

---

## 3. 实验原则

本实验必须遵守：

```text
1. 不修改模型 YAML。
2. 不加入 SPDConv。
3. 不加入 MSFF。
4. 不加入 GSConv。
5. 不调整数据增强策略。
6. 不改变输入尺寸。
7. 不改变训练轮数。
8. 不改变 batch、optimizer、lr0 等核心训练参数。
```

A4 只回答一个问题：

```text
把 YOLO11n 的 bbox regression loss 改为 MPDIoU 后，是否能提升定位质量和 mAP@50:95？
```

---

## 4. 修改文件位置

你已经使用可编辑安装的 Ultralytics，因此直接修改源码即可。

需要修改两个文件：

```text
ultralytics/utils/metrics.py
ultralytics/utils/loss.py
```

建议修改前备份：

```bash
cp ultralytics/utils/metrics.py ultralytics/utils/metrics.py.bak_a4_mpdiou
cp ultralytics/utils/loss.py ultralytics/utils/loss.py.bak_a4_mpdiou
```

Windows PowerShell：

```powershell
Copy-Item ultralytics/utils/metrics.py ultralytics/utils/metrics.py.bak_a4_mpdiou
Copy-Item ultralytics/utils/loss.py ultralytics/utils/loss.py.bak_a4_mpdiou
```

---

## 5. 在 metrics.py 中添加 MPDIoU 函数

打开：

```text
ultralytics/utils/metrics.py
```

搜索已有函数：

```python
def bbox_iou(
```

在 `bbox_iou` 函数后面添加以下函数。

```python

def bbox_mpdiou(box1, box2, xywh=True, eps=1e-7):
    """
    MPDIoU-style bounding box IoU.

    This function computes IoU with an additional point-distance penalty
    between the top-left and bottom-right corners of two boxes.

    Args:
        box1 (torch.Tensor): shape (..., 4)
        box2 (torch.Tensor): shape (..., 4)
        xywh (bool): if True, boxes are in xywh format; otherwise xyxy.
        eps (float): numerical stability term.

    Returns:
        torch.Tensor: MPDIoU value with shape (..., 1)

    Notes:
        In Ultralytics YOLO detection loss, boxes are usually passed in xyxy format.
        Therefore, this function will normally be called with xywh=False.
    """
    if xywh:
        # Convert xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        # Boxes are already xyxy
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # Box width and height
    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    # Intersection
    inter = (
        (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0)
        * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(min=0)
    )

    # Union and IoU
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Distance between top-left points and bottom-right points
    d1 = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    d2 = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)

    # Adaptive normalization by enclosing diagonal.
    # This avoids hard-coding image size and is stable inside YOLO loss,
    # where decoded boxes may be represented in feature-map units.
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    mpdiou = iou - (d1 + d2) / c2
    return mpdiou
```

### 说明

这里采用的是 **MPDIoU-style point-distance penalty**：

```text
MPDIoU = IoU - 顶左角距离惩罚 - 右下角距离惩罚
```

归一化项使用预测框和真实框的最小外接矩形对角线，原因是 Ultralytics 的 bbox loss 内部坐标不一定是原始 640 像素坐标，使用自适应外接框对角线更稳定。

---

## 6. 在 loss.py 中使用 bbox_mpdiou

打开：

```text
ultralytics/utils/loss.py
```

### 6.1 修改 import

搜索：

```python
from ultralytics.utils.metrics import bbox_iou
```

如果存在，改成：

```python
from ultralytics.utils.metrics import bbox_iou, bbox_mpdiou
```

如果是多行导入，就把 `bbox_mpdiou` 加进去。

---

### 6.2 修改 BboxLoss.forward 中的 IoU 计算

在 `loss.py` 中搜索：

```python
class BboxLoss
```

然后在它的 `forward()` 方法里搜索类似代码：

```python
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
```

或：

```python
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).squeeze(-1)
```

将其替换为：

```python
iou = bbox_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False).squeeze(-1)
```

如果原代码没有 `.squeeze(-1)`，则使用：

```python
iou = bbox_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
```

但建议保持原代码的张量形状处理方式。

---

## 7. 检查修改是否生效

运行：

```bash
python - <<'PY'
from ultralytics.utils.metrics import bbox_mpdiou
import torch

box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
box2 = torch.tensor([[1.0, 1.0, 11.0, 11.0]])
print('MPDIoU:', bbox_mpdiou(box1, box2, xywh=False))
PY
```

预期：

```text
能正常输出一个 tensor，不报错。
```

再运行：

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
print('YOLO11n load success')
PY
```

预期：

```text
YOLO11n load success
```

---

## 8. Smoke Test：先跑 1 个 epoch

正式训练前先跑 1 个 epoch，确认 loss 正常、显存正常。

```bash
yolo detect train \
  model=yolo11n.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=1 \
  batch=16 \
  device=0 \
  optimizer=SGD \
  lr0=0.01 \
  cache=False \
  amp=True \
  workers=4 \
  name=A4_mpdiou_smoke_test
```

如果 8GB 显存不足，改成：

```text
batch=8
```

如果仍然不足：

```text
batch=4
workers=2
```

Smoke Test 通过标准：

```text
1. 模型能正常构建；
2. 数据能正常读取；
3. loss 没有 NaN；
4. 训练能完成 1 个 epoch；
5. 验证阶段不报错。
```

---

## 9. 正式 A4 训练命令

为了与 A0 公平对比，使用 100 epochs。

```bash
yolo detect train \
  model=yolo11n.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0 \
  optimizer=SGD \
  lr0=0.01 \
  cache=False \
  amp=True \
  workers=4 \
  name=A4_yolo11n_mpdiou
```

如果显存不足：

```bash
yolo detect train \
  model=yolo11n.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=100 \
  batch=8 \
  device=0 \
  optimizer=SGD \
  lr0=0.01 \
  cache=False \
  amp=True \
  workers=4 \
  name=A4_yolo11n_mpdiou_b8
```

注意：

```text
如果 batch 从 16 改成 8，需要在实验报告中明确记录。
最好保持 A0 的 batch=16。
```

---

## 10. 训练后验证命令

训练完成后运行：

```bash
yolo detect val \
  model=runs/detect/A4_yolo11n_mpdiou/weights/best.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A4_yolo11n_mpdiou_val
```

如果正式训练时使用了其他 name，例如 `A4_yolo11n_mpdiou_b8`，需要把路径改成对应目录。

---

## 11. 结果记录模板

训练完成后，新建：

```text
experiments/A4_MPDIoU_Report.md
```

内容模板：

```markdown
# A4 YOLO11n + MPDIoU 实验报告

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | A4 |
| 模型名称 | YOLO11n + MPDIoU |
| 数据集 | DUO |
| 修改内容 | bbox loss 从 CIoU 风格替换为 MPDIoU 风格 |
| 网络结构 | YOLO11n 原结构 |
| Epochs | 100 |
| Batch | 16 |
| imgsz | 640 |
| Optimizer | SGD |
| lr0 | 0.01 |
| 预训练权重 | yolo11n.pt |

## 模型复杂度

| 指标 | A0 YOLO11n | A4 YOLO11n+MPDIoU | 差值 |
|---|---:|---:|---:|
| Params | 2.58M |  |  |
| GFLOPs | 6.3 |  |  |
| Model size | 5.5MB |  |  |

## 整体结果

| 指标 | A0 YOLO11n | A4 YOLO11n+MPDIoU | 差值 |
|---|---:|---:|---:|
| Precision | 0.848 |  |  |
| Recall | 0.762 |  |  |
| mAP@50 | 0.849 |  |  |
| mAP@50:95 | 0.656 |  |  |

## 各类别结果

| 类别 | A0 mAP@50 | A4 mAP@50 | 差值 | A0 mAP@50:95 | A4 mAP@50:95 | 差值 | A0 Recall | A4 Recall | 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| holothurian | 0.860 |  |  | 0.632 |  |  | 0.786 |  |  |
| echinus | 0.925 |  |  | 0.747 |  |  | 0.845 |  |  |
| scallop | 0.684 |  |  | 0.501 |  |  | 0.552 |  |  |
| starfish | 0.927 |  |  | 0.745 |  |  | 0.865 |  |  |

## 判断结论

- 如果 mAP@50:95 ≥ 0.659：保留 MPDIoU，进入 A6。
- 如果 mAP@50 ≥ 0.852：保留 MPDIoU，进入 A6。
- 如果 scallop mAP@50 ≥ 0.700：保留 MPDIoU，进入 A6。
- 如果整体 mAP 不提升，但 scallop mAP 和 Recall 明显提升：暂时保留，后续与 MSFF-Lite 组合。
- 如果整体 mAP 和 scallop 都下降：放弃 MPDIoU，尝试 Wise-IoU 或回到结构改进路线。
```

---

## 12. A4 后续决策

### 情况 1：A4 有效

如果 A4 提升 mAP@50:95 或 scallop mAP，则下一步做：

```text
A6 = YOLO11n + MSFF-Lite + MPDIoU
```

逻辑：

```text
MSFF-Lite：提升 Recall，减少漏检
MPDIoU：提升定位质量，改善 mAP@50:95
```

---

### 情况 2：A4 对整体有效，但 scallop 无提升

仍然可以进入 A6，但后续要重点观察：

```text
MSFF-Lite 是否能补充 scallop Recall
MPDIoU 是否能改善定位质量
```

---

### 情况 3：A4 无效

如果 A4 结果低于 A0，例如：

```text
mAP@50 < 0.849
mAP@50:95 < 0.656
scallop mAP 也下降
```

则不要继续使用 MPDIoU。

下一步可以改做：

```text
A5 = YOLO11n + SPDConv-OnlyP2 + MSFF-Lite
```

或者尝试：

```text
A4-WiseIoU = YOLO11n + Wise-IoU
```

---

## 13. 常见问题处理

### 13.1 ImportError: cannot import name 'bbox_mpdiou'

原因：

```text
loss.py 已经导入 bbox_mpdiou，但 metrics.py 中没有成功添加该函数。
```

处理：

```text
检查 ultralytics/utils/metrics.py 中是否存在 def bbox_mpdiou(...)
```

---

### 13.2 loss 出现 NaN

可能原因：

```text
框宽高为 0、归一化项过小、AMP 半精度不稳定。
```

处理顺序：

```text
1. 确认 bbox_mpdiou 中所有分母都加 eps。
2. 临时关闭 amp：amp=False。
3. 降低 lr0，例如 lr0=0.005。
```

但如果修改 lr0，需要单独记录，不可直接和 A0 绝对公平对比。

---

### 13.3 训练正常但精度明显下降

可能原因：

```text
MPDIoU 的角点距离惩罚过强，使模型过度关注框角点距离，影响分类置信度排序。
```

处理：

```text
放弃 MPDIoU，或尝试 Wise-IoU。
```

---

## 14. 恢复原始代码

如果 A4 结束后需要恢复原始 Ultralytics：

```bash
cp ultralytics/utils/metrics.py.bak_a4_mpdiou ultralytics/utils/metrics.py
cp ultralytics/utils/loss.py.bak_a4_mpdiou ultralytics/utils/loss.py
```

Windows PowerShell：

```powershell
Copy-Item ultralytics/utils/metrics.py.bak_a4_mpdiou ultralytics/utils/metrics.py -Force
Copy-Item ultralytics/utils/loss.py.bak_a4_mpdiou ultralytics/utils/loss.py -Force
```

---

## 15. 本阶段不要做的事

不要同时加入 MSFF。

不要同时加入 SPDConv。

不要同时加入 GSConv。

不要改 batch 后直接和 A0 做结论。

不要用 A4 的结果替代 A0 baseline。

A4 的唯一目标是：

```text
验证 MPDIoU 是否能提升 YOLO11n 在 DUO 上的边界框定位质量。
```

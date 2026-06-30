# A1-OnlyP2 实验指导文档：YOLO11n + SPDConv，仅替换 P2/4

## 1. 实验目的

本实验编号建议命名为：

```text
A1-OnlyP2
```

实验模型为：

```text
YOLO11n + SPDConv，仅替换 Backbone 中 P2/4 下采样 Conv
```

本实验是在 A1 实验基础上的修正版本。

A1 原实验替换了 Backbone 中 **P2/4 和 P3/8 两处下采样卷积**，结果显示：

```text
A0 YOLO11n:
  mAP@50    = 0.849
  mAP@50:95 = 0.656
  Recall    = 0.762
  GFLOPs    = 6.3

A1 YOLO11n + SPDConv(P2/4 + P3/8):
  mAP@50    = 0.841
  mAP@50:95 = 0.654
  Recall    = 0.774
  GFLOPs    = 8.4
```

A1 的积极信号是：

```text
scallop Recall: 0.552 → 0.592，提升 +0.040
整体 Recall:    0.762 → 0.774，提升 +0.012
```

A1 的问题是：

```text
整体 mAP@50 下降 0.008
整体 mAP@50:95 下降 0.002
GFLOPs 从 6.3 增加到 8.4，增加约 33.3%
```

因此，A1-OnlyP2 的目标是验证：

```text
只替换 P2/4 一处 SPDConv，能否保留 Recall 提升，同时降低计算量和 mAP 损失。
```

---

## 2. 实验假设

A1 中 SPDConv 使 scallop Recall 明显提升，说明它确实有助于减少小目标漏检。

但同时替换 P2/4 和 P3/8 两处下采样可能过度改变了 Backbone 的特征分布，导致整体 mAP 下降，并显著增加计算量。

因此，本实验只保留最浅层的 P2/4 替换：

```text
Conv(P2/4) → SPDConv(P2/4)
Conv(P3/8) 保持不变
```

预期效果：

```text
1. GFLOPs 低于 A1 的 8.4；
2. Recall 高于 A0 的 0.762；
3. scallop Recall 尽量高于 A0 的 0.552；
4. mAP@50 和 mAP@50:95 尽量接近或超过 A0。
```

---

## 3. 当前对比基准

后续结果需要与以下两个实验对比。

### 3.1 A0 baseline

```text
模型：YOLO11n
Params: 2,582,932
GFLOPs: 6.3
Precision: 0.848
Recall: 0.762
mAP@50: 0.849
mAP@50:95: 0.656
scallop Recall: 0.552
scallop mAP@50: 0.684
```

### 3.2 A1 双位置 SPDConv

```text
模型：YOLO11n + SPDConv(P2/4 + P3/8)
Params: 2,707,348
GFLOPs: 8.4
Precision: 0.842
Recall: 0.774
mAP@50: 0.841
mAP@50:95: 0.654
scallop Recall: 0.592
scallop mAP@50: 0.662
```

---

## 4. 修改位置

本实验只修改 YOLO11n Backbone 中的 **P2/4 下采样卷积**。

原始 YOLO11n Backbone 通常类似：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
```

A1-OnlyP2 修改后：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, SPDConv, [128, 3, 2]]    # 1-P2/4，OnlyP2 实验唯一替换位置
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8，保持原始 Conv
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16，保持原始 Conv
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32，保持原始 Conv
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
```

注意：

```text
只改模块类型，不改变层数，不改变输出特征图尺寸，所以 Head 索引一般不需要修改。
```

---

## 5. 创建 A1-OnlyP2 YAML

在 Ultralytics 源码目录中，复制 A1 的 YAML 或原始 YOLO11 YAML。

建议文件名：

```text
ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml
```

Linux / macOS：

```bash
cp ultralytics/cfg/models/11/yolo11.yaml \
   ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml
```

Windows PowerShell：

```powershell
Copy-Item ultralytics/cfg/models/11/yolo11.yaml `
  ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml
```

然后打开：

```text
ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml
```

只将 P2/4 位置的 `Conv` 修改为 `SPDConv`。

不要修改 P3/8、P4/16、P5/32。

---

## 6. 模型结构检查

先确认模型能够正常构建。

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml")
model.info()
print("A1-OnlyP2 model build success.")
PY
```

如果出现：

```text
KeyError: 'SPDConv'
```

说明 `SPDConv` 没有被正确注册到 `tasks.py` 或 `modules/__init__.py`。

如果出现通道不匹配，说明 `parse_model()` 没有把 `SPDConv` 放入和 `Conv` 相同的解析分支。

---

## 7. 前向传播测试

运行：

```bash
python - <<'PY'
import torch
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml").model
model.eval()

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    y = model(x)

print("Forward success.")
print(type(y))
PY
```

如果能输出：

```text
Forward success.
```

说明模型结构基本可用。

---

## 8. Smoke Test：1 epoch 快速测试

正式训练前，先跑 1 epoch。

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=1 \
  batch=16 \
  device=0 \
  optimizer=SGD \
  lr0=0.01 \
  pretrained=yolo11n.pt \
  cache=False \
  amp=True \
  workers=4 \
  name=A1_OnlyP2_smoke_test
```

观察：

```text
1. 是否正常读取 DUO 数据集；
2. 是否正常加载 yolo11n.pt 预训练权重；
3. 是否无显存爆炸；
4. loss 是否正常；
5. 日志中 Params 和 GFLOPs 是否低于 A1 的 2.71M / 8.4G。
```

---

## 9. 正式训练命令

为了与 A0、A1、A2 保持公平对比，A1-OnlyP2 使用相同训练设置：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0 \
  optimizer=SGD \
  lr0=0.01 \
  pretrained=yolo11n.pt \
  cache=False \
  amp=True \
  workers=4 \
  name=A1_OnlyP2_yolo11n_spd
```

如果 8GB 显存不足：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-spd-a1-onlyp2.yaml \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=100 \
  batch=8 \
  device=0 \
  optimizer=SGD \
  lr0=0.01 \
  pretrained=yolo11n.pt \
  cache=False \
  amp=True \
  workers=4 \
  name=A1_OnlyP2_yolo11n_spd_b8
```

如果仍然显存不足：

```text
batch=4
workers=2
```

但如果 batch 改了，报告中必须明确说明，和 A0 对比时要谨慎。

---

## 10. 验证命令

训练完成后，运行：

```bash
yolo detect val \
  model=runs/detect/A1_OnlyP2_yolo11n_spd/weights/best.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A1_OnlyP2_yolo11n_spd_val
```

如果训练时使用了不同的 `name`，对应修改权重路径。

---

## 11. 结果记录模板

训练完成后，新建：

```text
experiments/A1_OnlyP2_SPDConv_Report.md
```

推荐记录以下内容：

```markdown
# A1-OnlyP2 YOLO11n + SPDConv 实验报告

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | A1-OnlyP2 |
| 模型 | YOLO11n + SPDConv |
| 修改位置 | Backbone P2/4 下采样 Conv 替换为 SPDConv |
| 数据集 | DUO |
| Epochs | 100 |
| Batch | 16 |
| imgsz | 640 |
| optimizer | SGD |
| lr0 | 0.01 |
| pretrained | yolo11n.pt |

## 模型复杂度对比

| 指标 | A0 YOLO11n | A1 P2+P3 SPDConv | A1-OnlyP2 |
|---|---:|---:|---:|
| Params | 2,582,932 | 2,707,348 |  |
| GFLOPs | 6.3 | 8.4 |  |
| Model size | 5.5MB | 5.7MB |  |

## 整体结果对比

| 指标 | A0 YOLO11n | A1 P2+P3 SPDConv | A1-OnlyP2 | A1-OnlyP2 - A0 |
|---|---:|---:|---:|---:|
| Precision | 0.848 | 0.842 |  |  |
| Recall | 0.762 | 0.774 |  |  |
| mAP@50 | 0.849 | 0.841 |  |  |
| mAP@50:95 | 0.656 | 0.654 |  |  |

## 各类别结果对比

| 类别 | A0 mAP@50 | A1 mAP@50 | A1-OnlyP2 mAP@50 | A0 Recall | A1 Recall | A1-OnlyP2 Recall |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.860 | 0.852 |  | 0.786 | 0.787 |  |
| echinus | 0.925 | 0.925 |  | 0.845 | 0.851 |  |
| scallop | 0.684 | 0.662 |  | 0.552 | 0.592 |  |
| starfish | 0.927 | 0.927 |  | 0.865 | 0.865 |  |

## 判断结论

- 如果 A1-OnlyP2 的 mAP@50:95 >= 0.656，且 Recall > 0.762，保留 SPDConv-OnlyP2。
- 如果 A1-OnlyP2 的 mAP@50 接近 A0，且 scallop Recall 明显高于 0.552，暂时保留到 A5-Lite 组合实验。
- 如果 A1-OnlyP2 的 mAP 和 Recall 都没有优势，淘汰 SPDConv。
- 如果 A1-OnlyP2 明显优于 A1 P2+P3，则后续组合实验使用 OnlyP2，不再使用 P2+P3。
```

---

## 12. 判断标准

A1-OnlyP2 的保留标准分为三档。

### 12.1 强保留

满足以下任意一组：

```text
mAP@50 >= 0.849 且 mAP@50:95 >= 0.656
```

或：

```text
mAP@50:95 > 0.656
```

结论：

```text
SPDConv-OnlyP2 可以作为后续最终模型候选模块。
```

### 12.2 弱保留

满足：

```text
mAP@50 不低于 0.845
mAP@50:95 不低于 0.653
Recall 高于 0.762
scallop Recall 高于 0.570
GFLOPs 明显低于 8.4
```

结论：

```text
可以进入 A5-Lite 组合实验，但不能单独作为有效改进点。
```

### 12.3 淘汰

出现以下任意情况：

```text
mAP@50 < 0.841
mAP@50:95 < 0.651
scallop Recall <= 0.552
GFLOPs 接近 A1 的 8.4，但精度没有提升
```

结论：

```text
SPDConv 不进入最终模型。
```

---

## 13. 后续实验决策

根据 A1-OnlyP2 结果决定下一步。

### 情况 A：A1-OnlyP2 优于 A1 P2+P3

如果：

```text
A1-OnlyP2 mAP > A1
且 GFLOPs < A1
且 scallop Recall 仍高于 A0
```

下一步做：

```text
A5-Lite = YOLO11n + SPDConv-OnlyP2 + MSFF-Lite(P3 only)
```

### 情况 B：A1-OnlyP2 不如 A1，但仍提升 Recall

如果：

```text
scallop Recall 高于 A0
但整体 mAP 仍下降
```

下一步建议先做：

```text
A4 = YOLO11n + MPDIoU
```

再考虑：

```text
A6 = YOLO11n + MSFF-Lite + MPDIoU
```

### 情况 C：A1-OnlyP2 没有价值

如果：

```text
mAP 下降
Recall 没明显提升
scallop Recall 没提升
```

则：

```text
淘汰 SPDConv，后续不再组合 SPDConv。
```

后续重点转向：

```text
MSFF-Lite + 改进 IoU Loss
```

---

## 14. 本实验不要做的事

```text
不要加入 MSFF。
不要加入 GSConv。
不要修改 Loss。
不要改数据增强策略。
不要调整 imgsz。
不要把 P3/8、P4/16、P5/32 一起替换。
不要用 A1-OnlyP2 和 batch 不同的 A0 直接比较。
```

A1-OnlyP2 只回答一个问题：

```text
SPDConv 只替换 P2/4 时，是否比 P2/4 + P3/8 更合理？
```

---

## 15. 最终预期

理想结果：

```text
mAP@50    >= 0.849
mAP@50:95 >= 0.656
Recall    > 0.762
scallop Recall > 0.570
GFLOPs < 8.4
```

可接受结果：

```text
mAP@50    >= 0.845
mAP@50:95 >= 0.653
Recall    > 0.762
scallop Recall > 0.570
GFLOPs 明显低于 8.4
```

如果达不到可接受结果，则不建议继续保留 SPDConv。

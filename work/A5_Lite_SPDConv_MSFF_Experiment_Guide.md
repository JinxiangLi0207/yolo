# A5-Lite 实验指导文档：YOLO11n + SPDConv-OnlyP2 + MSFF-Lite

## 0. 实验结论背景

当前已有实验结果表明：

- A1-OnlyP2：`SPDConv` 仅替换 Backbone 的 `P2/4` 下采样 Conv，能够提升 `scallop` Recall，但整体 mAP 没有超过 A0。
- A2-Lite：`MSFF` 仅加入 `P3/8` 检测分支前，能够提升整体 Recall，mAP@50 基本接近 A0，但 mAP@50:95 略降。
- A4：`MPDIoU` Loss 实验失败，必须恢复原始 YOLO11 Loss 后再进行 A5-Lite。

A5-Lite 的目的不是继续堆模块，而是验证：

```text
SPDConv-OnlyP2 的浅层小目标信息保留能力
+
MSFF-Lite 的 P3 小目标分支特征增强能力
=
是否能形成互补
```

---

## 1. 实验编号与模型定义

| 项目 | 内容 |
|---|---|
| 实验编号 | A5-Lite |
| 模型名称 | YOLO11n + SPDConv-OnlyP2 + MSFF-Lite |
| 数据集 | DUO |
| 修改 1 | Backbone P2/4 下采样 Conv 替换为 SPDConv |
| 修改 2 | P3/8 检测分支前加入 MSFF-Lite |
| Loss | 恢复原始 YOLO11 Loss，不使用 MPDIoU |
| Epochs | 100 |
| Batch | 16，若显存不足改为 8 或 4 |
| imgsz | 640 |
| Optimizer | SGD |
| lr0 | 0.01 |
| Pretrained | yolo11n.pt |

---

## 2. 必须先恢复原始 Loss

A4 的 MPDIoU 实验已经证明不适合当前 DUO 小目标检测任务。A5-Lite 必须使用原始 YOLO11 Loss。

### 2.1 Windows PowerShell 恢复命令

在你的 Ultralytics 源码根目录下执行：

```powershell
Copy-Item ultralytics/utils/metrics.py.bak_a4_mpdiou ultralytics/utils/metrics.py -Force
Copy-Item ultralytics/utils/loss.py.bak_a4_mpdiou ultralytics/utils/loss.py -Force
```

### 2.2 Linux / Git Bash 恢复命令

```bash
cp ultralytics/utils/metrics.py.bak_a4_mpdiou ultralytics/utils/metrics.py
cp ultralytics/utils/loss.py.bak_a4_mpdiou ultralytics/utils/loss.py
```

### 2.3 如果备份文件不存在

如果 `.bak_a4_mpdiou` 文件不存在，使用 Git 恢复：

```bash
git checkout -- ultralytics/utils/metrics.py
git checkout -- ultralytics/utils/loss.py
```

Windows PowerShell 同样可以执行：

```powershell
git checkout -- ultralytics/utils/metrics.py
git checkout -- ultralytics/utils/loss.py
```

### 2.4 恢复后快速检查

执行：

```bash
python - <<'PY'
from pathlib import Path

files = [
    Path('ultralytics/utils/metrics.py'),
    Path('ultralytics/utils/loss.py'),
]

for f in files:
    text = f.read_text(encoding='utf-8', errors='ignore')
    print(f'{f}:')
    print('  contains MPDIoU:', 'MPDIoU' in text or 'mpdiou' in text.lower())
PY
```

期望输出：

```text
contains MPDIoU: False
```

如果仍然出现 `True`，说明 MPDIoU 代码还没有完全恢复干净。

---

## 3. 恢复 Loss 后验证 A0 权重

恢复 Loss 后，先验证 A0 权重，确认环境已经恢复正常。

```bash
yolo detect val \
  model=runs/detect/train20/weights/best.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A5_restore_loss_check
```

期望结果接近 A0：

```text
mAP@50    ≈ 0.849
mAP@50:95 ≈ 0.656
```

如果验证结果仍然接近 A4 的低结果，例如 `mAP@50≈0.596`，不要继续 A5-Lite，先检查 `loss.py` 和 `metrics.py` 是否恢复正确。

---

## 4. A5-Lite 结构设计

A5-Lite 由两个已经验证过的候选模块组成：

```text
A5-Lite = YOLO11n + SPDConv-OnlyP2 + MSFF-Lite
```

### 4.1 SPDConv 插入位置

只替换 Backbone 中 `P2/4` 下采样层。

不要替换：

```text
P1/2 stem Conv
P3/8 下采样 Conv
P4/16 下采样 Conv
P5/32 下采样 Conv
```

### 4.2 MSFF 插入位置

只在 `P3/8` 检测分支前加入 MSFF。

不要在 `P4/16` 和 `P5/32` 检测分支前加入 MSFF，因为 A2-Full 已经证明 P3/P4/P5 全加会导致性能下降。

---

## 5. 创建 A5-Lite YAML

建议新建模型配置文件：

```text
ultralytics/cfg/models/11/yolo11n-a5-lite-spd-msff.yaml
```

可以从你已经完成的 A2-Lite 或 A1-OnlyP2 YAML 复制，然后合并两个修改。

### 5.1 Backbone 修改示例

原始 YOLO11n Backbone 通常类似：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]      # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]     # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]     # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]     # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]    # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
```

A5-Lite 中只修改第 1 层：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, SPDConv, [128, 3, 2]]    # 1-P2/4, A5-Lite: only this downsample Conv is replaced
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

### 5.2 Head / Neck 中加入 MSFF-Lite

A2-Lite 已经实现了 P3-only MSFF。A5-Lite 直接沿用 A2-Lite 的 MSFF 插入方式。

目标结构应为：

```text
P3/8 feature → MSFF → Detect
P4/16 feature → Detect
P5/32 feature → Detect
```

不要改成：

```text
P3 → MSFF → Detect
P4 → MSFF → Detect
P5 → MSFF → Detect
```

如果你的 A2-Lite YAML 中 MSFF 插入后修改了 Detect 的输入索引，那么 A5-Lite 需要保持同样的索引逻辑。

---

## 6. 模型构建检查

运行：

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolo11n-a5-lite-spd-msff.yaml')
model.info()
print('A5-Lite model build success.')
PY
```

如果出现 `KeyError: SPDConv` 或 `KeyError: MSFF`，说明模块没有在 `tasks.py` 或 `modules/__init__.py` 中注册。

如果出现通道数错误，检查：

```text
1. SPDConv 是否被加入 parse_model 的 base_modules
2. MSFF 是否使用了正确的输入通道
3. Detect 层输入索引是否对应 MSFF 后的 P3、原始 P4、原始 P5
```

---

## 7. 前向传播测试

```bash
python - <<'PY'
import torch
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolo11n-a5-lite-spd-msff.yaml').model
model.eval()

x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    y = model(x)

print('Forward success.')
print(type(y))
PY
```

---

## 8. Smoke Test：1 epoch 试跑

正式训练前先跑 1 epoch：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-a5-lite-spd-msff.yaml \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=1 \
  batch=4 \
  device=0 \
  optimizer=SGD \
  lr0=0.01 \
  pretrained=yolo11n.pt \
  cache=False \
  amp=True \
  workers=2 \
  name=A5_lite_smoke_test
```

确认以下内容：

```text
1. 模型能正常构建
2. 数据能正常读取
3. loss 不为 nan
4. 显存不爆
5. 日志中能看到 Params 和 GFLOPs
```

---

## 9. 正式训练命令

如果 smoke test 正常，运行 100 epoch 训练：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-a5-lite-spd-msff.yaml \
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
  name=A5_lite_yolo11n_spdP2_msffP3
```

如果 8GB 显存不足：

```text
第一步：batch=8
第二步：workers=2
第三步：batch=4
```

不要改变：

```text
imgsz=640
epochs=100
optimizer=SGD
lr0=0.01
```

否则不能和 A0/A1/A2 公平对比。

---

## 10. 验证命令

训练完成后执行：

```bash
yolo detect val \
  model=runs/detect/A5_lite_yolo11n_spdP2_msffP3/weights/best.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A5_lite_val
```

如果验证爆显存，改为：

```text
batch=8
```

---

## 11. 结果记录模板

新建实验报告：

```text
experiments/A5_Lite_SPDConv_MSFF_Report.md
```

记录以下内容：

```markdown
# A5-Lite YOLO11n + SPDConv-OnlyP2 + MSFF-Lite 实验报告

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | A5-Lite |
| 模型名称 | YOLO11n + SPDConv-OnlyP2 + MSFF-Lite |
| 数据集 | DUO |
| SPDConv 位置 | Backbone P2/4 下采样 Conv |
| MSFF 位置 | P3/8 检测分支前 |
| Loss | 原始 YOLO11 Loss |
| Epochs | 100 |
| Batch | 16 |
| imgsz | 640 |
| Optimizer | SGD |
| lr0 | 0.01 |
| Pretrained | yolo11n.pt |

## 模型复杂度

| 指标 | A0 YOLO11n | A1-OnlyP2 | A2-Lite | A5-Lite |
|---|---:|---:|---:|---:|
| Params | 2.58M | 2.60M | 2.59M |  |
| GFLOPs | 6.3 | 7.0 | 6.4 |  |
| Model size | 5.5MB | 5.5MB | 5.5MB |  |
| FPS | ~455 | ~455 | ~400 |  |

## 整体结果

| 指标 | A0 YOLO11n | A1-OnlyP2 | A2-Lite | A5-Lite | A5-A0 差值 |
|---|---:|---:|---:|---:|---:|
| Precision | 0.848 | 0.844 | 0.842 |  |  |
| Recall | 0.762 | 0.763 | 0.778 |  |  |
| mAP@50 | 0.849 | 0.840 | 0.848 |  |  |
| mAP@50:95 | 0.656 | 0.654 | 0.651 |  |  |

## 各类别结果

| 类别 | A0 mAP@50 | A5 mAP@50 | 差值 | A0 Recall | A5 Recall | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.860 |  |  | 0.786 |  |  |
| echinus | 0.925 |  |  | 0.845 |  |  |
| scallop | 0.684 |  |  | 0.552 |  |  |
| starfish | 0.927 |  |  | 0.865 |  |  |

## 判断结论

- 若 mAP@50 ≥ 0.850，说明 A5-Lite 超过 A0，可保留。
- 若 mAP@50:95 ≥ 0.656，说明定位质量至少不低于 A0，可保留。
- 若 Recall ≥ 0.780 且 mAP@50 ≥ 0.846，可作为召回增强型模型保留。
- 若 scallop Recall ≥ 0.590 且 scallop mAP@50 ≥ 0.670，可作为小目标优化型候选模型。
- 若 mAP@50 < 0.845 且 mAP@50:95 < 0.650，说明组合无效，应放弃 SPDConv。
```

---

## 12. A5-Lite 成功标准

A5-Lite 满足任意一个条件即可进入后续实验：

| 条件 | 标准 | 说明 |
|---|---:|---|
| 整体 mAP@50 超过 A0 | ≥ 0.850 | 最理想 |
| 整体 mAP@50:95 不低于 A0 | ≥ 0.656 | 定位质量保持 |
| 召回增强且 mAP 基本不掉 | Recall ≥ 0.780 且 mAP@50 ≥ 0.846 | 可作为召回增强模型 |
| scallop 改善明显 | scallop Recall ≥ 0.590 且 scallop mAP@50 ≥ 0.670 | 可作为小目标增强模型 |

A5-Lite 失败标准：

```text
mAP@50 < 0.845
且
mAP@50:95 < 0.650
```

如果失败，说明 SPDConv 与 MSFF-Lite 组合没有形成互补，应放弃 SPDConv。

---

## 13. A5-Lite 之后的决策

### 情况 1：A5-Lite 有效

如果 A5-Lite 有效，下一步做：

```text
A7 = A5-Lite + GSConv
```

目的：

```text
在保持 A5-Lite 精度的基础上，用 GSConv 降低计算量。
```

### 情况 2：A5-Lite 无效

如果 A5-Lite 无效，放弃 SPDConv，下一步做：

```text
A6 = YOLO11n + MSFF-Lite + GSConv
```

此时论文路线改为：

```text
P3 小目标分支特征增强 + Neck 轻量化
```

### 情况 3：A5-Lite 只提升 Recall，但 mAP 继续下降

这种情况下不要急着保留 A5-Lite。先比较：

```text
A2-Lite 是否比 A5-Lite 更稳
```

如果 A2-Lite 的 mAP 更高、GFLOPs 更低，则最终模型优先从 A2-Lite 继续发展。

---

## 14. 本实验不要做的事

不要加入 MPDIoU。

不要加入 Wise-IoU 或 SIoU。

不要加入 GSConv。

不要使用 MSFF-Full。

不要替换 P3/8 下采样 Conv 为 SPDConv。

不要改数据增强策略。

不要改 imgsz、epochs、optimizer、lr0 后直接和 A0 对比。

A5-Lite 的核心目标只有一个：

```text
验证 SPDConv-OnlyP2 与 MSFF-Lite 是否能互补。
```

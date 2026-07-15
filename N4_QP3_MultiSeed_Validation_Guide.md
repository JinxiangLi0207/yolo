# N4-QP3 多随机种子验证指导

## 1. 公平校准结论

N4 和 C1A 使用同一批 `quality_power` 后，seed 0 结果如下：

| power | N4 mAP50-95 | C1A mAP50-95 | C1A - N4 |
|---:|---:|---:|---:|
| 1.00 | 0.664253 | 0.662783 | -0.001470 |
| 1.25 | 0.666306 | 0.665257 | -0.001049 |
| 1.50 | 0.668597 | 0.666864 | -0.001733 |
| 1.75 | 0.670932 | 0.668753 | -0.002179 |
| 2.00 | 0.672657 | 0.670497 | -0.002160 |
| 2.50 | 0.675928 | 0.673893 | -0.002035 |
| 3.00 | **0.678852** | 0.677000 | -0.001852 |

C1A 在所有相同 power 对照中均未超过 N4，因此不进入 seeds 1/2。C1A 保留为结构消融：它能够提高 mAP50 和 scallop AP，但不能提高总体 mAP50-95。

## 2. N4-QP3 当前性能

N4 seed 0、power=3.0：

| Precision | Recall | mAP50 | mAP50-95 | fitness | scallop mAP50 | scallop mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.868320 | 0.745971 | 0.844523 | **0.678852** | **0.695419** | 0.665 | 0.526 |

相对 YOLO11n seed 0 的三位小数结果：

```text
Precision       +0.037
Recall          -0.032
mAP50           +0.001
mAP50-95        +0.022
scallop mAP50   -0.006
scallop mAP50-95 +0.016
```

`power=3.0` 暂定为 N4 的固定推理参数，不再扫描更高指数。对应配置：

```text
ultralytics/cfg/models/11/yolo11n-quality-n4-qp3.yaml
```

## 3. 下一步：现有权重免训练复验

`quality_power` 只影响推理排序，不参与训练损失。先复用已经完成的 N4 100 轮 seed 1/2 权重进行验证，不重新训练。

### 3.1 训练1：验证 seed 2

```bash
python - <<'PY'
from pathlib import Path

from ultralytics import YOLO

root = Path("/root/yolo/runs/detect")
matches = sorted(root.glob("*N4*e100*seed2*/weights/best.pt")) + sorted(
    root.glob("*n4*e100*seed2*/weights/best.pt")
)
if len(matches) != 1:
    raise RuntimeError(f"Expected one N4 e100 seed2 weight, found {len(matches)}: {matches}")

weight = str(matches[0])
print(f"Using: {weight}")
model = YOLO(weight)
model.model.model[-1].quality_power = 3.0
metrics = model.val(
    data="ultralytics/cfg/datasets/DUO.yaml",
    batch=96,
    imgsz=640,
    workers=8,
    device=0,
    plots=False,
    seed=2,
    deterministic=True,
    name="D2_N4_qp3_e100_seed2",
)
print({key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()})
PY
```

### 3.2 训练2：验证 seed 1

```bash
python - <<'PY'
from pathlib import Path

from ultralytics import YOLO

root = Path("/root/yolo/runs/detect")
matches = sorted(root.glob("*N4*e100*seed1*/weights/best.pt")) + sorted(
    root.glob("*n4*e100*seed1*/weights/best.pt")
)
if len(matches) != 1:
    raise RuntimeError(f"Expected one N4 e100 seed1 weight, found {len(matches)}: {matches}")

weight = str(matches[0])
print(f"Using: {weight}")
model = YOLO(weight)
model.model.model[-1].quality_power = 3.0
metrics = model.val(
    data="ultralytics/cfg/datasets/DUO.yaml",
    batch=96,
    imgsz=640,
    workers=8,
    device=0,
    plots=False,
    seed=1,
    deterministic=True,
    name="D2_N4_qp3_e100_seed1",
)
print({key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()})
PY
```

如果某台服务器检测到零个或多个候选权重，先执行：

```bash
find /root/yolo/runs/detect -path "*n4*e100*seed1*/weights/best.pt" -o \
  -path "*n4*e100*seed2*/weights/best.pt" -o \
  -path "*N4*e100*seed1*/weights/best.pt" -o \
  -path "*N4*e100*seed2*/weights/best.pt"
```

然后人工指定对应 seed 的正确 100 轮权重。不要使用 50 轮权重，也不要跨 seed 代替。

## 4. 最终统计

### 4.1 当前已完成结果

| seed | 模型 | Precision | Recall | mAP50 | mAP50-95 | scallop mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | YOLO11n | 0.831 | 0.778 | 0.844 | 0.657 | 0.510 |
| 0 | N4-QP3 | 0.868320 | 0.745971 | 0.844523 | **0.678852** | **0.526** |
| 1 | YOLO11n | 0.837 | 0.763 | 0.839 | 0.657 | **0.509** |
| 1 | N4-QP3 | 0.851804 | 0.755526 | 0.843370 | **0.675873** | 0.508 |
| 2 | YOLO11n | 0.835 | 0.779 | 0.849 | 0.659 | **0.524** |
| 2 | N4-QP3 | 0.855858 | 0.740582 | 0.846178 | **0.676792** | 0.514 |

配对增益：

```text
seed 0: mAP50-95 +0.021852
seed 1: mAP50-95 +0.018873
seed 2: mAP50-95 +0.017792
三种子平均增益 +0.019506 +/- 0.002103
```

三个 seed 均超过 `+0.017`，总体 mAP50-95 增益具有良好重复性。Recall 三种子平均下降约 0.026；scallop mAP50-95 的配对变化为 `+0.016/-0.001/-0.010`，稀有类别结论不稳定。

### 4.2 seed 1 权重已找回

训练2只找到：

```text
/root/yolo/runs/detect/F1_n4_full_e100_b96_seed2/weights/best.pt
```

随后在训练1找到 seed 1：

```text
/root/yolo/runs/detect/F1_n4_full_e100_b96_seed1/weights/best.pt
```

因此不需要重新训练 seed 1。直接在训练1使用该权重固定 `quality_power=3.0` 验证，命令见本文件第 3.2 节；`seed=1` 必须保持不变。

以下重新训练命令仅作为权重损坏时的备用方案，当前不要执行。先检查配置：

```bash
python -c "from ultralytics import YOLO; m=YOLO('ultralytics/cfg/models/11/yolo11n-quality-n4-qp3.yaml'); print(type(m.model.model[-1]).__name__, m.model.model[-1].quality_power)"
```

期望输出包含：

```text
QualityDetect 3.0
```

然后执行：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-quality-n4-qp3.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 epochs=100 imgsz=640 workers=8 device=0 \
  pretrained=weights/yolo11n.pt cache=True \
  seed=1 deterministic=True \
  quality_loss_gain=0.5 rcqfl=False sqr=False \
  name=D2_N4_qp3_e100_b96_seed1
```

该训练会让 `best.pt` 按 QP3 验证指标选择，比复用旧 power 下选择的 checkpoint 更严格。不要用 seed 2 权重代替 seed 1，也不要从已有 N4 `best.pt` 微调。

### 4.3 三种子统计

三种子统计已经完成，标准差采用样本标准差（`n-1`）：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n | 0.834333 ± 0.003055 | **0.773333 ± 0.008963** | 0.844000 ± 0.005000 | 0.657667 ± 0.001155 |
| N4-QP3 | **0.858661 ± 0.008607** | 0.747360 ± 0.007568 | **0.844690 ± 0.001411** | **0.677172 ± 0.001525** |
| 配对变化 | **+0.024327 ± 0.011652** | -0.025974 ± 0.016337 | +0.000690 ± 0.003599 | **+0.019506 ± 0.002103** |

逐类三种子均值：

| 类别 | Baseline mAP50 | N4-QP3 mAP50 | 变化 | Baseline mAP50-95 | N4-QP3 mAP50-95 | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.857 | 0.869 | +0.012 | 0.629 | 0.654 | **+0.025** |
| echinus | 0.919 | 0.926 | +0.007 | 0.745 | 0.777 | **+0.032** |
| scallop | 0.673 | 0.659 | -0.014 | 0.514 | 0.516 | +0.002 |
| starfish | 0.926 | 0.925 | -0.001 | 0.742 | 0.762 | **+0.020** |

正式报告应包含：

```text
Precision mean +/- std
Recall mean +/- std
mAP50 mean +/- std
mAP50-95 mean +/- std
各类别 AP mean +/- std
相对同 seed YOLO11n 的配对差值
```

N4-QP3 晋级检查：

```text
平均 mAP50-95 提升 >= 0.015
至少 2/3 个 seed 的 mAP50-95 提升 >= 0.010
平均 mAP50 相对 YOLO11n 的下降不超过 0.005
scallop mAP50-95 不低于 YOLO11n
```

四项均已满足，N4-QP3 正式通过当前开发协议下的三种子晋级检查。

## 5. 后续算法方向

若 N4-QP3 多种子稳定，当前论文主干可写成“定位质量监督 + 质量感知置信度校准”。下一结构创新不再使用 MSFF，而应针对 N4-QP3 的 Recall 下降设计残差质量融合或尺度自适应质量融合：

```text
score = class_score * ((1 - lambda) + lambda * quality_score**3)
```

目标是在保留 mAP50-95 增益的同时，将 Recall 恢复到基准附近。该实验应在 N4-QP3 多种子结果确认后再实现。

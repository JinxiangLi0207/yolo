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

收到 seed 1/2 结果后，计算 seeds 0/1/2 的：

```text
Precision mean +/- std
Recall mean +/- std
mAP50 mean +/- std
mAP50-95 mean +/- std
各类别 AP mean +/- std
相对同 seed YOLO11n 的配对差值
```

建议 N4-QP3 晋级标准：

```text
平均 mAP50-95 提升 >= 0.015
至少 2/3 个 seed 的 mAP50-95 提升 >= 0.010
平均 mAP50 不低于 YOLO11n 超过 0.005
scallop mAP50-95 不低于 YOLO11n
```

## 5. 后续算法方向

若 N4-QP3 多种子稳定，当前论文主干可写成“定位质量监督 + 质量感知置信度校准”。下一结构创新不再使用 MSFF，而应针对 N4-QP3 的 Recall 下降设计残差质量融合或尺度自适应质量融合：

```text
score = class_score * ((1 - lambda) + lambda * quality_score**3)
```

目标是在保留 mAP50-95 增益的同时，将 Recall 恢复到基准附近。该实验应在 N4-QP3 多种子结果确认后再实现。

# N6 分支自适应质量融合实验指导

## 1. 实验动机

当前 C1A seed 0 的全局 power 扫描表明，不存在一个统一指数能够同时使 Recall 和 mAP50-95 超过 YOLO11n：

| C1A power | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| YOLO11n baseline | **0.778** | 0.844 | 0.657 |
| 0.0 | **0.780** | **0.851** | 0.652 |
| 0.5 | 0.773 | **0.853** | **0.658** |
| 1.5 | 0.766 | **0.854** | **0.667** |
| 3.0 | 0.744 | **0.853** | **0.677** |

低 power 保留更多候选和 Recall，高 power 改善高 IoU 排序。统一 power 将三个尺度绑定在同一个精度与召回权衡中，因此 N6 为 P3/P4/P5 设置独立指数：

```text
P3 small-object branch: low quality power
P4/P5 branches: quality_power = 3.0
```

目标是在 P3 保留小目标候选，同时保留 P4/P5 的 QP3 高 IoU 增益。

## 2. 实现说明

`QualityDetect.quality_power` 现在支持两种形式：

```python
head.quality_power = 3.0              # 原标量行为，向后兼容
head.quality_power = (0.5, 3.0, 3.0) # P3/P4/P5 分支自适应
```

推理阶段在拼接三个尺度之前分别计算：

```text
score_l = class_score_l * quality_score_l ** power_l
```

该改动不增加参数量、GFLOPs 或训练损失，不改变现有 checkpoint 结构。旧的 N4、C1A 权重可以直接复用。

## 3. 实验原则

1. 本阶段只做免训练验证，不启动新训练。
2. N4 与 C1A 必须扫描相同 power 组合。
3. power 顺序固定为 `[P3, P4, P5]`。
4. 仅改变推理融合指数，其他验证参数完全一致。
5. 先使用 seed 0 筛选，候选通过后再验证 seeds 1/2。
6. 不继续搜索 P4/P5，固定为已经验证的 3.0。

## 4. 候选组合

```text
[3.0, 3.0, 3.0]  # 标量 QP3 等价对照
[0.0, 3.0, 3.0]  # P3 不使用质量分数
[0.5, 3.0, 3.0]
[1.0, 3.0, 3.0]
[1.5, 3.0, 3.0]
[2.0, 3.0, 3.0]
```

`[3,3,3]` 的结果必须与已有标量 `power=3.0` 基本一致，否则说明实现或尺度顺序存在错误，应立即停止实验。

## 5. 上传代码后的烟雾测试

两台服务器执行 `git pull` 后，任选一台运行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt"
model = YOLO(weight)
head = model.model.model[-1]
head.quality_power = (0.5, 3.0, 3.0)
print(type(head).__name__, head.quality_power, head.nl)
PY
```

期望输出：

```text
QualityDetect (0.5, 3.0, 3.0) 3
```

如果权重路径不同，只修改 `weight`。

## 6. 训练1：C1A seed 0 扫描

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/C1A_strict_p3_msff_quality_e100_b96_seed0/weights/best.pt"
power_sets = (
    (3.0, 3.0, 3.0),
    (0.0, 3.0, 3.0),
    (0.5, 3.0, 3.0),
    (1.0, 3.0, 3.0),
    (1.5, 3.0, 3.0),
    (2.0, 3.0, 3.0),
)

for powers in power_sets:
    tag = "_".join(f"{power:g}".replace(".", "p") for power in powers)
    print(f"\n===== C1A powers={powers} =====")
    model = YOLO(weight)
    model.model.model[-1].quality_power = powers
    metrics = model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        seed=0,
        deterministic=True,
        name=f"N6_C1A_seed0_powers_{tag}",
    )
    exact = {key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()}
    print(f"EXACT C1A powers={powers}: {exact}")
PY
```

## 7. 训练2：N4 seed 0 扫描

训练2当前 N4 seed 0 权重路径为：

```text
/root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt
```

执行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt"
power_sets = (
    (3.0, 3.0, 3.0),
    (0.0, 3.0, 3.0),
    (0.5, 3.0, 3.0),
    (1.0, 3.0, 3.0),
    (1.5, 3.0, 3.0),
    (2.0, 3.0, 3.0),
)

for powers in power_sets:
    tag = "_".join(f"{power:g}".replace(".", "p") for power in powers)
    print(f"\n===== N4 powers={powers} =====")
    model = YOLO(weight)
    model.model.model[-1].quality_power = powers
    metrics = model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        seed=0,
        deterministic=True,
        name=f"N6_N4_seed0_powers_{tag}",
    )
    exact = {key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()}
    print(f"EXACT N4 powers={powers}: {exact}")
PY
```

## 8. 结果记录表

| 模型 | P3/P4/P5 power | Precision | Recall | mAP50 | mAP50-95 | scallop mAP50 | scallop mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| N4 | 3/3/3 | | | | | | |
| N4 | 0/3/3 | | | | | | |
| N4 | 0.5/3/3 | | | | | | |
| N4 | 1/3/3 | | | | | | |
| N4 | 1.5/3/3 | | | | | | |
| N4 | 2/3/3 | | | | | | |
| C1A | 3/3/3 | | | | | | |
| C1A | 0/3/3 | | | | | | |
| C1A | 0.5/3/3 | | | | | | |
| C1A | 1/3/3 | | | | | | |
| C1A | 1.5/3/3 | | | | | | |
| C1A | 2/3/3 | | | | | | |

## 9. 判定规则

YOLO11n seed 0 参考值：

```text
Recall = 0.778
mAP50 = 0.844
mAP50-95 = 0.657
scallop mAP50-95 = 0.510
```

### S 级：实现目标

```text
Recall > 0.778
mAP50-95 > 0.657
mAP50 >= 0.844
```

三项必须同时满足，证明分支自适应 power 打破了统一 power 的精度与召回权衡。

### A 级：保留候选

```text
Recall >= 0.760
mAP50-95 >= 0.674
mAP50 >= 0.844
scallop mAP50-95 >= 0.510
```

若没有 S 级结果，选择 mAP50-95 最高的 A 级候选进入下一阶段。

### 停止条件

1. `[3,3,3]` 不能复现标量 QP3 时停止，先修复实现。
2. 所有候选 mAP50-95 均低于 0.674 且 Recall 低于 0.760 时，停止分支 power 路线。
3. C1A 未超过相同 tuple 的 N4 时，不训练 C1A seeds 1/2。

## 10. 后续分支

### N4 分支获胜

直接使用现有 N4 seeds 1/2 权重验证相同 tuple，无需重新训练。三种子稳定后，再生成固定 YAML。

### C1A 分支获胜

只有 C1A 同时超过相同 tuple 的 N4，才为 C1A 补跑 seeds 1/2，并生成固定 YAML。

### 没有候选通过

停止 MSFF 和分支 power，转向残差质量融合：

```text
score = class_score * ((1 - lambda) + lambda * quality_score ** 3)
```

## 11. 论文表述

如果实验成功，可将该机制暂定为：

```text
Scale-Adaptive Localization Quality Calibration (SA-LQC)
尺度自适应定位质量校准
```

核心表述：统一质量指数对所有尺度施加相同抑制，容易过度压制 P3 小目标候选；SA-LQC 对高分辨率小目标分支采用较弱质量校准，对 P4/P5 保持较强高 IoU 排序，从而改善精度与召回之间的权衡。

## 12. 实验边界

当前筛选仍使用已参与开发的 DUO test，因此只能用于方向判断。正式论文必须在独立 dev_val 上选择三分支 power，并冻结后在 test 上评估一次。不要继续扩大 P4/P5 搜索范围，也不要添加类别级 power，以免形成无法解释的测试集调参。

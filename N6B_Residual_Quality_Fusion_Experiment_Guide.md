# N6-B 全尺度残差质量融合实验指导

## 1. 实验目标

N4-QP3 使用以下置信度融合：

```text
final_score = class_score * quality_score ** 3
```

该策略使三种子平均 mAP50-95 提高 `0.019506`，但 Recall 平均下降 `0.025974`。N6-A 为 P3/P4/P5 设置不同 power 后，低 power 分支会在全局排序中获得额外分数偏置，未能同时保持 Recall 和 mAP50-95。

N6-B 不再改变不同检测尺度之间的相对校准方式，而是在所有尺度上使用相同的残差融合系数：

```text
final_score = class_score * ((1 - lambda) + lambda * quality_score ** 3)
```

其中：

```text
lambda = 0: 只使用分类分数
lambda = 1: 完全等价于 N4-QP3
0 < lambda < 1: 保留质量排序，同时减弱对低质量候选的过度抑制
```

本实验复用现有 N4 权重，只改变推理阶段排序，不重新训练，不增加参数量和 GFLOPs。

## 2. 实现说明

`QualityDetect` 新增运行时属性：

```python
head.quality_power = 3.0
head.quality_mix = 0.75
```

默认 `quality_mix=1.0`，旧 YAML 和旧 checkpoint 保持原行为。加载修改前保存的旧权重时，代码也会自动回退到 `1.0`。

## 3. 实验协议

```text
Dataset: DUO
Weights: 已有 N4 100-epoch best.pt
imgsz: 640
batch: 96
quality_power: 3.0
quality_mix: 0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0
Training: none
```

第一轮同时使用 seed 0 和 seed 1 扫描。根据两种子的共同趋势选择候选后，再用 seed 2 做确认，避免根据一个 seed 的偶然波动定参数。

## 4. 上传后的兼容性检查

在任意服务器执行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt"
model = YOLO(weight)
head = model.model.model[-1]
head.quality_power = 3.0
head.quality_mix = 0.75
print(type(head).__name__, head.quality_power, head.quality_mix)
PY
```

期望输出：

```text
QualityDetect 3.0 0.75
```

如果权重路径不同，只修改 `weight`。若不是 `QualityDetect`，立即停止实验并检查权重。

同时确认 Python 使用的是当前仓库代码，并且 `forward()` 中包含 `quality_mix`：

```bash
python - <<'PY'
import inspect
import ultralytics
from ultralytics.nn.modules.head import QualityDetect

print("ultralytics:", ultralytics.__file__)
source = inspect.getsource(QualityDetect.forward)
print("quality_mix active:", "quality_mix" in source)
print(source)
PY
```

必须看到：

```text
ultralytics: /root/yolo/ultralytics/__init__.py
quality_mix active: True
```

若路径不是 `/root/yolo/ultralytics`，或输出为 `False`，不要开始扫描。

### 4.1 端点检查

完整扫描前先只运行 `quality_mix=0` 和 `quality_mix=1`。`mix=1` 应复现 QP3；`mix=0` 应产生不同结果。若两者所有指标完全相同，说明新融合代码未生效，本次实验必须标记为无效。

## 5. 训练2：seed 0 扫描

训练2已知 seed 0 权重路径为：

```text
/root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt
```

执行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt"
mixes = (0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0)

for mix in mixes:
    print(f"\n===== N6-B seed0 quality_mix={mix:g} =====")
    model = YOLO(weight)
    head = model.model.model[-1]
    head.quality_power = 3.0
    head.quality_mix = mix
    metrics = model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        seed=0,
        deterministic=True,
        name=f"N6B_N4_e100_seed0_mix{mix:g}",
    )
    exact = {key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()}
    print(f"EXACT seed0 mix={mix:g}: {exact}")
PY
```

2026-07-15 首次 seed 0 扫描因训练2未拉取最新代码而无效。同步代码后的重跑已通过端点检查并完成，结果见 `work/N6B_Residual_Quality_Fusion_Experiment_Report.md`。

## 6. 训练1：seed 1 扫描

训练1已知 seed 1 权重路径为：

```text
/root/yolo/runs/detect/F1_n4_full_e100_b96_seed1/weights/best.pt
```

执行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/F1_n4_full_e100_b96_seed1/weights/best.pt"
mixes = (0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0)

for mix in mixes:
    print(f"\n===== N6-B seed1 quality_mix={mix:g} =====")
    model = YOLO(weight)
    head = model.model.model[-1]
    head.quality_power = 3.0
    head.quality_mix = mix
    metrics = model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        seed=1,
        deterministic=True,
        name=f"N6B_N4_e100_seed1_mix{mix:g}",
    )
    exact = {key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()}
    print(f"EXACT seed1 mix={mix:g}: {exact}")
PY
```

## 7. 一致性检查

`quality_mix=1.0` 必须复现已有 N4-QP3 结果：

| Seed | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0 | 0.868320 | 0.745971 | 0.844523 | 0.678852 |
| 1 | 0.851804 | 0.755526 | 0.843370 | 0.675873 |

允许日志显示造成的末位差异，但精确指标应基本一致。若不一致，实验无效，先检查代码版本、权重和数据路径。

## 8. 晋级规则

以各 seed 对应的 YOLO11n 为配对基准，并优先比较 seed 0/1 的平均值。

### S 级：理想候选

```text
mean mAP50-95 >= 0.674
mean Recall >= 0.760
mean mAP50 >= 0.842
两个 seed 的 mAP50-95 均高于各自 baseline
```

### A 级：可继续候选

```text
mean mAP50-95 >= 0.672
mean Recall >= 0.755
mean mAP50 >= 0.842
```

如果多个候选通过，先选择 mAP50-95 最高者；当差距小于 `0.001` 时，选择 Recall 更高者。只将一个固定 `quality_mix` 交给 seed 2 验证，不根据 seed 2 再调参。

## 9. seed 2 确认命令（本次不执行）

本命令仅保留为实验协议模板。由于 seed 0/1 没有候选通过晋级条件，本次禁止执行。仅当未来在独立 `dev_val` 上重新获得通过条件的唯一候选时，才将 `BEST_MIX` 替换后使用：

```bash
BEST_MIX=0.75 python - <<'PY'
import os
from ultralytics import YOLO

mix = float(os.environ["BEST_MIX"])
weight = "/root/yolo/runs/detect/F1_n4_full_e100_b96_seed2/weights/best.pt"
model = YOLO(weight)
head = model.model.model[-1]
head.quality_power = 3.0
head.quality_mix = mix
metrics = model.val(
    data="ultralytics/cfg/datasets/DUO.yaml",
    batch=96,
    imgsz=640,
    workers=8,
    device=0,
    plots=False,
    seed=2,
    deterministic=True,
    name=f"N6B_N4_e100_seed2_mix{mix:g}",
)
print({key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()})
PY
```

## 10. 停止条件与后续

若没有候选达到 A 级：

```text
N6-B failed
停止所有后训练 power/mix 搜索
保留 N4-QP3 作为当前最优模型
下一步转向训练阶段的质量监督改进
```

训练阶段优先分析 P3/P4/P5 质量预测与真实 IoU 的相关性，再决定是采用分尺度质量损失权重、正样本质量目标平滑，还是质量分支蒸馏。不得继续在 DUO test 上细化 `quality_mix` 小数，否则会形成明显的测试集调参。

### 10.1 最终状态

有效 seed 0/1 扫描没有候选达到 S 级或 A 级，N6-B 已停止，不运行 seed 2。当前最优模型仍为 N4-QP3，后续转向训练阶段质量监督诊断与改进。

## 11. 结果记录表

| Seed | quality_mix | Precision | Recall | mAP50 | mAP50-95 | scallop mAP50 | scallop mAP50-95 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.0 | | | | | | | |
| 0 | 0.25 | | | | | | | |
| 0 | 0.5 | | | | | | | |
| 0 | 0.625 | | | | | | | |
| 0 | 0.75 | | | | | | | |
| 0 | 0.875 | | | | | | | |
| 0 | 1.0 | 0.868320 | 0.745971 | 0.844523 | 0.678852 | 0.665 | 0.526 | QP3 control |
| 1 | 0.0 | | | | | | | |
| 1 | 0.25 | | | | | | | |
| 1 | 0.5 | | | | | | | |
| 1 | 0.625 | | | | | | | |
| 1 | 0.75 | | | | | | | |
| 1 | 0.875 | | | | | | | |
| 1 | 1.0 | 0.851804 | 0.755526 | 0.843370 | 0.675873 | | | QP3 control |

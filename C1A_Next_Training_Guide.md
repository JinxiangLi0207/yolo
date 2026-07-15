# C1A 下一步训练指导

## 1. 当前结论

C1A strict P3-only 在 DUO、100 epochs、batch=96、seed=0 下得到：

| 模型 | Precision | Recall | mAP50 | mAP50-95 | scallop mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0.831 | 0.778 | 0.844 | 0.657 | 0.510 |
| N4，quality_power=1.0 | 0.859 | 0.760 | 0.848 | **0.664** | 0.501 |
| C1A，quality_power=1.0 | **0.862** | **0.762** | **0.854** | 0.662 | **0.521** |

C1A 已经恢复 N4 损失的 mAP50 和 scallop AP，但 mAP50-95 仍比 N4 低 0.002。C1B 已终止，不再训练。

## 2. 实验目标

下一步先回答：C1A 的 mAP50-95 差距是否来自推理阶段质量分数融合过强或过弱。

```text
final_score = class_score * quality_score ** quality_power
```

`quality_power` 只在推理阶段生效，不参与训练损失，因此可以复用同一个 `best.pt`，无需重新训练。

## 3. D1：质量融合指数扫描

### 3.1 服务器安排

- 训练1：执行 C1A power sweep。
- 训练2：暂时不启动新的 100 轮任务，等待 D1 结果。
- 预计只需完成 6 次验证，远快于一次训练。

### 3.2 执行命令

在训练1的 `/root/yolo` 目录执行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/C1A_strict_p3_msff_quality_e100_b96_seed0/weights/best.pt"
powers = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)

for power in powers:
    print(f"\n===== C1A quality_power={power:g} =====")
    model = YOLO(weight)
    model.model.model[-1].quality_power = power
    model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        seed=0,
        deterministic=True,
        name=f"D1_C1A_e100_seed0_power{power:g}",
    )
PY
```

如果服务器上的 DUO 配置不是上述路径，只修改 `data=`，其他验证参数不变。

### 3.3 记录表

| quality_power | Precision | Recall | mAP50 | mAP50-95 | scallop P | scallop R | scallop mAP50 | scallop mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.843 | **0.780** | 0.851 | 0.652 | 0.798 | **0.608** | 0.699 | 0.515 |
| 0.25 | 0.845 | 0.777 | 0.852 | 0.655 | 0.802 | 0.599 | 0.699 | 0.517 |
| 0.50 | 0.846 | 0.773 | 0.853 | 0.658 | 0.795 | 0.594 | 0.699 | 0.519 |
| 0.75 | 0.846 | 0.773 | **0.854** | 0.660 | 0.791 | 0.594 | 0.699 | 0.522 |
| 1.00 | **0.860** | 0.760 | **0.854** | 0.663 | **0.806** | 0.585 | 0.700 | 0.523 |
| 1.25 | 0.852 | 0.764 | **0.854** | **0.665** | 0.790 | 0.585 | **0.701** | **0.525** |

不要只记录总体 mAP50-95。必须同时记录 mAP50、Recall 和 scallop AP，避免通过牺牲小目标检出来换取单一指标。

### 3.4 D1 结果分析

从 `quality_power=0.0` 到 `1.25`：

```text
总体 mAP50-95：0.652 -> 0.665，提升 0.013
总体 mAP50：   0.851 -> 0.854，提升 0.003
总体 Recall：  0.780 -> 0.764，下降 0.016
scallop mAP50-95：0.515 -> 0.525，提升 0.010
```

质量融合对高 IoU 排序和 scallop AP 有稳定正作用。`power=1.25` 已满足原定四项晋级条件，并相对 N4 获得约 `mAP50-95 +0.001`、`mAP50 +0.006`、`Recall +0.004` 和 `scallop mAP50-95 +0.024`。

但当前最优值位于扫描边界，而且 `0.665` 与 N4 的 `0.664` 都是三位小数结果。不能据此立即锁定 1.25，需要执行 D1b 高区间扫描并输出六位小数。

### 3.5 D1b：高区间精细扫描

训练1执行：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/C1A_strict_p3_msff_quality_e100_b96_seed0/weights/best.pt"
powers = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)

for power in powers:
    print(f"\n===== C1A quality_power={power:g} =====")
    model = YOLO(weight)
    model.model.model[-1].quality_power = power
    metrics = model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        seed=0,
        deterministic=True,
        name=f"D1b_C1A_e100_seed0_power{power:g}",
    )
    exact = {key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()}
    print(f"EXACT power={power:g}: {exact}")
PY
```

如果 `2.5` 仍是最高点，只追加 `3.0` 一项，不继续无限扩大搜索范围。若相邻指数的未四舍五入 mAP50-95 差值不超过 `0.001`，选择较小的指数，以保留 Recall 并减少校准敏感性。

### 3.6 D1b 结果

| power | Precision | Recall | mAP50 | mAP50-95 | fitness | scallop mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | **0.860056** | 0.760335 | 0.853885 | 0.662783 | 0.681893 | 0.523 |
| 1.25 | 0.851814 | 0.763738 | **0.854150** | 0.665257 | 0.684147 | 0.525 |
| 1.50 | 0.845842 | **0.766495** | 0.853883 | 0.666864 | 0.685566 | 0.526 |
| 1.75 | 0.849449 | 0.764584 | 0.853760 | 0.668753 | 0.687254 | 0.527 |
| 2.00 | 0.853832 | 0.762232 | 0.853773 | 0.670497 | 0.688824 | 0.529 |
| 2.50 | 0.859187 | 0.754412 | 0.853292 | **0.673893** | **0.691832** | **0.534** |

从 power 1.0 到 2.5，mAP50-95 提高 `0.011110`，scallop mAP50-95 提高 `0.011`，mAP50 基本不变。`power=2.5` 的 Recall 比预设 0.755 下限低 `0.000588`，属于边界结果；`power=2.0` 是当前满足全部预设条件的最优平衡点。

但当前还不能证明 C1A 超过 N4。原因是 C1A 已针对 power 进行校准，而此前 N4 的正式对照只使用 `power=1.0`。必须让 N4 使用相同 power 区间验证，排除收益仅来自后处理校准的可能。

### 3.7 D1c：边界点与 N4 公平校准

训练1只补测 C1A `power=3.0`：

```bash
python - <<'PY'
from ultralytics import YOLO

weight = "/root/yolo/runs/detect/C1A_strict_p3_msff_quality_e100_b96_seed0/weights/best.pt"
model = YOLO(weight)
model.model.model[-1].quality_power = 3.0
metrics = model.val(
    data="ultralytics/cfg/datasets/DUO.yaml",
    batch=96,
    imgsz=640,
    workers=8,
    device=0,
    plots=False,
    seed=0,
    deterministic=True,
    name="D1c_C1A_e100_seed0_power3",
)
print({key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()})
PY
```

训练2使用 N4 seed 0 的 100 轮 `best.pt` 执行同区间扫描。先用 `find` 确认实际路径，不要误用 50 轮或其他 seed：

```bash
find /root/yolo/runs/detect -path "*N4*seed0*/weights/best.pt" -o \
  -path "*n4*seed0*/weights/best.pt"
```

确认路径后执行：

```bash
python - <<'PY'
from pathlib import Path

from ultralytics import YOLO

root = Path("/root/yolo/runs/detect")
matches = sorted(root.glob("*N4*seed0*/weights/best.pt")) + sorted(root.glob("*n4*seed0*/weights/best.pt"))
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one N4 seed0 weight, found {len(matches)}: {matches}")
weight = str(matches[0])
print(f"Using N4 weight: {weight}")
powers = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

for power in powers:
    print(f"\n===== N4 quality_power={power:g} =====")
    model = YOLO(weight)
    model.model.model[-1].quality_power = power
    metrics = model.val(
        data="ultralytics/cfg/datasets/DUO.yaml",
        batch=96,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        seed=0,
        deterministic=True,
        name=f"D1c_N4_e100_seed0_power{power:g}",
    )
    exact = {key: f"{float(value):.6f}" for key, value in metrics.results_dict.items()}
    print(f"EXACT N4 power={power:g}: {exact}")
PY
```

当前训练2实际找到的文件为：

```text
/root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt
```

脚本会自动使用该文件。若检测到零个或多个匹配项则主动停止，此时应人工确认哪一个是 N4 的 100 轮 seed 0 权重；不能拿 seed 1/2 或 50 轮权重与 C1A seed 0 横向比较。

### 3.8 C1A power=3.0 结果

| power | Precision | Recall | mAP50 | mAP50-95 | fitness | scallop mAP50 | scallop mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.00 | 0.869024 | 0.743898 | 0.853475 | **0.677000** | **0.694647** | 0.704 | **0.539** |

相对 YOLO11n seed 0：

```text
Precision       +0.038024
Recall          -0.034102
mAP50           +0.009475
mAP50-95        +0.020000
scallop mAP50   +0.033
scallop mAP50-95 +0.029
```

`power=3.0` 是当前最大 AP 候选；`power=2.0` 仍是满足原 Recall 约束的均衡候选。虽然 3.0 的 mAP50-95 仍高于 2.5，但不再继续扫描更高 power，原因是 Recall 已降至 0.743898，而且继续在当前 test 划分上扩大搜索会增加后处理过拟合风险。

当前 C1A 候选固定为：

```text
最大 AP：power=3.0，mAP50-95=0.677000
均衡方案：power=2.0，mAP50-95=0.670497，Recall=0.762232
```

最终选择等待 N4 公平扫描结果。如果 N4 在高 power 下获得相同幅度的提升，则不能将 C1A 的全部增益归因于 MSFF。

### 3.9 N4 公平扫描结论

N4 在所有相同 power 对照中均高于 C1A 的 mAP50-95：

| power | N4 | C1A | C1A - N4 |
|---:|---:|---:|---:|
| 1.00 | 0.664253 | 0.662783 | -0.001470 |
| 1.25 | 0.666306 | 0.665257 | -0.001049 |
| 1.50 | 0.668597 | 0.666864 | -0.001733 |
| 1.75 | 0.670932 | 0.668753 | -0.002179 |
| 2.00 | 0.672657 | 0.670497 | -0.002160 |
| 2.50 | 0.675928 | 0.673893 | -0.002035 |
| 3.00 | **0.678852** | 0.677000 | -0.001852 |

C1A 没有通过主指标晋级条件，停止 seeds 1/2。其正向价值保留为：同 power 下 mAP50 和 scallop AP 更高；负向结果是总体 mAP50-95 稳定低于 N4。这一结果可作为论文中的 MSFF 位置与模块交互消融。

## 4. D1 晋级规则

按以下优先级选择候选指数：

```text
1. mAP50-95 高于经过同等 power 校准的 N4 seed 0
2. mAP50 >= 0.850
3. scallop mAP50-95 >= 0.510
4. Recall >= 0.755
```

### 情况 A：D1c 后 C1A 满足全部条件

选择 mAP50-95 最高的指数。若多个指数的 mAP50-95 差值不超过 0.001，优先选择 Recall 和 scallop AP 更高者；仍相同则选择更接近 1.0 的指数，减少额外调参解释。

然后进入 D2 多随机种子训练。C1A 的候选 power 暂定为 2.0、2.5 或 3.0，最终选择必须结合 N4 的公平校准结果。

### 情况 B：最高 mAP50-95 为 0.663 或 0.664

视为与 N4 持平，不立即开展多种子。先检查未四舍五入的 `results_dict` 或验证输出，确认差异是否真实存在。若仍未超过 N4，进入尺度自适应质量融合设计。

### 情况 C：所有指数都低于 0.664

停止原始 C1A 的多种子训练，不再延长到 150 epochs，也不恢复 C1B。下一实验改为尺度自适应质量融合，使 P3 与 P4/P5 使用不同的质量融合强度。

## 5. D2：C1A 多随机种子复现（取消）

C1A 未通过与校准后 N4 的公平比较，本节命令不再执行，仅保留为历史方案。当前多种子任务改为复验 N4-QP3 的已有 seed 1/2 权重，详见 `N4_QP3_MultiSeed_Validation_Guide.md`。

```text
ultralytics/cfg/models/11/yolo11n-msff-quality-c1-qp<POWER>.yaml
```

该 YAML 只允许修改最后一层：

```yaml
- [[17, 20, 23], 1, QualityDetect, [nc, <POWER>]]
```

固定 YAML 提交到 Git 后，两台服务器拉取同一 commit，再并行训练。

### 5.1 训练1：seed=1

```bash
yolo detect train \
  model=<C1A_FINAL_YAML> \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 epochs=100 imgsz=640 workers=8 device=0 \
  pretrained=weights/yolo11n.pt cache=True \
  seed=1 deterministic=True \
  quality_loss_gain=0.5 rcqfl=False sqr=False \
  name=D2_C1A_e100_b96_seed1
```

### 5.2 训练2：seed=2

```bash
yolo detect train \
  model=<C1A_FINAL_YAML> \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 epochs=100 imgsz=640 workers=8 device=0 \
  pretrained=weights/yolo11n.pt cache=True \
  seed=2 deterministic=True \
  quality_loss_gain=0.5 rcqfl=False sqr=False \
  name=D2_C1A_e100_b96_seed2
```

注意：D2 必须从 `weights/yolo11n.pt` 重新训练，不能从 seed 0 的 `best.pt` 微调，也不能 `resume` 50 轮实验。

## 6. D2 最终判定

将 C1A seeds 0/1/2 与对应 seed 的 YOLO11n、N4 做配对比较，报告 `mean ± std`。

建议论文候选的最低标准：

```text
三种子平均 mAP50-95 >= N4 平均值 0.663
三种子平均 mAP50 > N4 平均值 0.843
三种子平均 scallop mAP50-95 >= YOLO11n
至少 2/3 个 seed 的 mAP50-95 高于对应 seed 的 YOLO11n
参数量和 GFLOPs 保持约 2.60M / 6.4
```

若平均 mAP50-95 不能超过 N4，但 mAP50 和 scallop AP 稳定更高，应将 C1A 定位为消融变体，而不是最终模型；最终模型继续采用尺度自适应质量融合。

## 7. 论文协议提醒

当前 `images/test` 被用作 Ultralytics `val` 并选择 `best.pt`。D1 可用于当前开发判断，但正式论文不能在 test 上反复选择 `quality_power`。结构确定后必须：

1. 从训练集划分 `dev_train/dev_val`；
2. 在 `dev_val` 固定 `quality_power`；
3. 冻结结构和超参数；
4. 最后仅在官方 test 上评价一次。

## 8. 当前立即执行项

```text
N4 公平扫描已完成，N4 power=3.0 达到 0.678852，C1A 未晋级。
训练1使用现有 N4 seed2 权重验证 power=3.0。
训练2使用现有 N4 seed1 权重验证 power=3.0。
不要启动 C1B。
不要启动 C1A seeds 1/2。
执行命令见 N4_QP3_MultiSeed_Validation_Guide.md。
```

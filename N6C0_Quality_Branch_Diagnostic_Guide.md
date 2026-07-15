# N6-C0 质量分支诊断实验指导

## 1. 实验目标

N4-QP3 已稳定提高 mAP50-95，但降低 Recall。N6-A 分尺度推理指数和 N6-B 残差推理融合均未打破该权衡，说明下一步必须检查训练阶段的质量监督。

当前正样本质量标签和权重为：

```text
quality_target = IoU(pred_box, matched_gt)
positive_weight = TaskAligned target score
```

可能存在两类问题：

1. P3、小目标或 scallop 的质量预测被系统性低估；
2. P3 难正样本的 Task-Aligned 权重偏低，导致质量分支学习不足。

N6-C0 不修改模型、不重新训练，只复用现有 N4 seed0/1 权重，在相同数据上统计匹配正样本的质量预测与真实 IoU。

## 2. 输出指标

脚本输出总体、P3/P4/P5、类别、尺度和“尺度×类别”分组指标：

```text
quality_mean           预测质量均值
iou_mean               匹配正样本真实 IoU 均值
bias_q_minus_iou       quality - IoU，负值表示系统性低估
mae                    质量预测绝对误差
pearson                质量预测与 IoU 的线性相关性
task_weight_mean       当前正样本质量损失权重均值
area_ratio_mean        匹配目标的图像面积占比
high_iou_low_q_rate    IoU>=0.7 但 quality<0.5 的比例
```

同时输出各尺度背景负样本的质量均值，以及 `quality>0.1/0.3/0.5` 的比例。

## 3. 实验原则

1. seed0 和 seed1 使用同一代码、数据和参数；
2. 本实验不扫描任何超参数；
3. 不运行 seed2；
4. 不根据单一 seed 决定损失改法；
5. 两个 seed 的分尺度趋势一致后，才实现 N6-C1；
6. 正式论文应在独立 `dev_val` 上做方法选择，官方 test 不应继续承担调参功能。

## 4. 训练2：分析 seed0

先确认已拉取包含 `build_diagnostic_criterion` 的最新脚本：

```bash
grep -n "build_diagnostic_criterion" quality_iou_diagnostic.py
```

若没有输出，说明服务器仍是旧代码，不要运行诊断。

```bash
cd /root/yolo
python quality_iou_diagnostic.py \
  --weights /root/yolo/runs/detect/F1_n4_full_native_e100_b96_seed0/weights/best.pt \
  --data ultralytics/cfg/datasets/DUO.yaml \
  --batch 96 --imgsz 640 --workers 8 --device 0 \
  --name N6C0_N4_e100_seed0
```

输出目录：

```text
/root/yolo/runs/quality_diagnostics/N6C0_N4_e100_seed0/
```

## 5. 训练1：分析 seed1

```bash
cd /root/yolo
python quality_iou_diagnostic.py \
  --weights /root/yolo/runs/detect/F1_n4_full_e100_b96_seed1/weights/best.pt \
  --data ultralytics/cfg/datasets/DUO.yaml \
  --batch 96 --imgsz 640 --workers 8 --device 0 \
  --name N6C0_N4_e100_seed1
```

## 6. 需要返回的结果

请保留控制台中从下面标记开始的全部内容：

```text
QUALITY_DIAGNOSTIC_SUMMARY
```

同时保留两个 CSV：

```text
quality_positive_calibration.csv
quality_negative_summary.csv
```

控制台摘要用于快速判断，CSV 用于论文统计和进一步分析“尺度×类别”组合。

## 7. 决策规则

### 路线 A：尺度感知质量目标平滑

当两个 seed 同时满足以下现象时选择：

```text
P3 bias_q_minus_iou <= -0.03
P3 pearson >= 0.35
P3 或 small 的 high_iou_low_q_rate 明显高于 P4/P5
```

说明质量分支能够区分定位质量，但系统性低估小目标。N6-C1 使用只作用于正样本的小目标质量目标平滑：

```text
smallness = exp(-area_ratio / tau)
q_target = IoU + alpha * smallness * (1 - IoU)
```

先固定 `tau`，只比较两个温和 `alpha`，禁止同时搜索多个参数。

### 路线 B：尺度均衡正样本质量损失

当两个 seed 同时满足以下现象时选择：

```text
P3 pearson < 0.35
P3 task_weight_mean 明显低于 P4/P5
P3 MAE 明显高于 P4/P5
```

说明 P3 质量分支学习不足。N6-C1 保持 IoU 标签不变，只对小目标正样本增加归一化损失权重：

```text
raw_weight = 1 + alpha * exp(-area_ratio / tau)
normalized_weight = raw_weight / weighted_mean(raw_weight)
```

归一化用于保持总质量损失尺度基本不变，避免把收益混同为简单增大 `quality_loss_gain`。

### 路线 C：不继续修改质量监督

如果 P3 与其他尺度的 bias、MAE、Pearson 和 task weight 没有稳定差异，则现有数据不支持“P3 质量监督不足”的假设。此时停止 N6-C，不应为了继续实验而随意加入类别权重或新损失。

## 8. 下一轮训练安排

返回 seed0/1 的诊断结果后再确定。预计只选择路线 A 或 B 中的一条：

```text
训练1：N6-C1 弱补偿，50 epochs，seed0
训练2：N6-C1 中等补偿，50 epochs，seed0
```

其他参数继续使用当前公平筛选协议：

```text
batch=96 epochs=50 imgsz=640 workers=8 device=0
pretrained=weights/yolo11n.pt cache=True
seed=0 deterministic=True
quality_loss_gain=0.5 rcqfl=False sqr=False
```

只有至少一个版本同时提高同协议 N4 的 mAP50-95，并明显恢复 Recall 或 scallop，才进入 100 epochs 和多种子验证。

## 9. 当前停止项

```text
MSFF/C1: stopped
N6-A per-level quality power: stopped
N6-B residual quality mix: stopped
quality_power/mix refinement on test: stopped
N6-C loss implementation: wait for N6-C0 evidence
```

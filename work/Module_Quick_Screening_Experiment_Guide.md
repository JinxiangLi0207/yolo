# 模块快速筛选实验指导文档

## 0. 实验目标

当前目标从“建立最强 baseline”切换为：

```text
快速筛选可提升 DUO 性能的模块，尽快找到可写论文的有效结构。
```

已有实验显示：

- SPDConv、MSFF、UFE、Gated-MSFF 等增强型模块多数能提高 Recall，但 mAP 和 Precision 不稳定。
- 因此本轮筛选优先选择定位/空间/背景抑制相关模块，而不是继续堆纯通道增强模块。

本轮快筛只做趋势判断，不作为最终论文结果。  
若 50 epochs 有苗头，再用 100 epochs 复跑确认。

---

## 1. 快筛基准

当前快速基准使用 A8-0：

| 实验 | 模型 | 训练设置 | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---:|---:|---:|---:|
| A8-0 | YOLO11n | 100 epochs, batch=96, imgsz=640, cache=True | 0.831 | 0.778 | 0.844 | 0.657 |

快筛成功标准至少满足一个：

```text
mAP@50 > 0.844
或 mAP@50:95 > 0.657
或 Precision >= 0.850 且 Recall >= 0.760
```

如果只提升 Recall，但 mAP@50 和 mAP@50:95 均下降，则不保留。

---

## 2. 本轮筛选模块

| 编号 | YAML | 模块 | 插入/替换位置 | 论文叙事方向 |
|---|---|---|---|---|
| S1 | `yolo11n-ca-p3-s1.yaml` | CA | P3 head 后 | 位置感知小目标检测 |
| S2 | `yolo11n-cbam-p3-s2.yaml` | CBAM | P3 head 后 | 背景抑制与空间注意力 |
| S3 | `yolo11n-simam-p3-s3.yaml` | SimAM | P3 head 后 | 无参数弱目标重标定 |
| S4 | `yolo11n-ema-p3-s4.yaml` | EMA | P3 head 后 | 多尺度注意力与背景抑制 |
| S5 | `yolo11n-c2f-msblock-p3-s5.yaml` | C2f_MSBlock | 替换 P3 head C3k2 | 多尺度局部上下文建模 |

说明：

- S1-S4 只在 P3 小目标分支后插入注意力模块。
- S5 不是插入注意力，而是替换 P3 head 的特征提取块，更像结构创新。
- 所有实验只改一个位置，便于快速判断模块有效性。

---

## 3. 统一快筛训练协议

所有快筛实验先统一使用：

```text
epochs=50
batch=96
imgsz=640
cache=True
pretrained=weights/yolo11n.pt
close_mosaic=默认值
cos_lr=默认值
```

命令模板：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/xxx.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 \
  epochs=50 \
  imgsz=640 \
  workers=8 \
  device=0 \
  pretrained=weights/yolo11n.pt \
  cache=True \
  name=xxx
```

---

## 4. 训练命令

### S1：CA(P3)

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ca-p3-s1.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 \
  epochs=50 \
  imgsz=640 \
  workers=8 \
  device=0 \
  pretrained=weights/yolo11n.pt \
  cache=True \
  name=S1_yolo11n_ca_p3_e50
```

### S2：CBAM(P3)

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-cbam-p3-s2.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 \
  epochs=50 \
  imgsz=640 \
  workers=8 \
  device=0 \
  pretrained=weights/yolo11n.pt \
  cache=True \
  name=S2_yolo11n_cbam_p3_e50
```

### S3：SimAM(P3)

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-simam-p3-s3.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 \
  epochs=50 \
  imgsz=640 \
  workers=8 \
  device=0 \
  pretrained=weights/yolo11n.pt \
  cache=True \
  name=S3_yolo11n_simam_p3_e50
```

### S4：EMA(P3)

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ema-p3-s4.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 \
  epochs=50 \
  imgsz=640 \
  workers=8 \
  device=0 \
  pretrained=weights/yolo11n.pt \
  cache=True \
  name=S4_yolo11n_ema_p3_e50
```

### S5：C2f_MSBlock(P3)

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-c2f-msblock-p3-s5.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 \
  epochs=50 \
  imgsz=640 \
  workers=8 \
  device=0 \
  pretrained=weights/yolo11n.pt \
  cache=True \
  name=S5_yolo11n_c2f_msblock_p3_e50
```

---

## 5. 两台服务器建议分配

如果有两台服务器并行：

| 服务器 | 任务 |
|---|---|
| 服务器 1 | S1 CA、S2 CBAM、S3 SimAM |
| 服务器 2 | S4 EMA、S5 C2f_MSBlock |

优先级：

```text
S1 CA > S4 EMA > S2 CBAM > S3 SimAM > S5 C2f_MSBlock
```

原因：

- CA 和 EMA 更可能兼顾定位和背景抑制。
- CBAM 有空间注意力，值得筛。
- SimAM 成本低，快速验证。
- C2f_MSBlock 结构变化较大，可能有效但风险更高。

---

## 6. 结果记录表

| 编号 | 模型 | Params | GFLOPs | Precision | Recall | mAP@50 | mAP@50:95 | scallop P | scallop R | scallop mAP50 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A8-0 | YOLO11n | 2.58M | 6.3 | 0.831 | 0.778 | 0.844 | 0.657 | 0.730 | 0.590 | 0.671 | 快速基准 |
| S1 | CA(P3) |  |  |  |  |  |  |  |  |  |  |
| S2 | CBAM(P3) |  |  |  |  |  |  |  |  |  |  |
| S3 | SimAM(P3) |  |  |  |  |  |  |  |  |  |  |
| S4 | EMA(P3) |  |  |  |  |  |  |  |  |  |  |
| S5 | C2f_MSBlock(P3) |  |  |  |  |  |  |  |  |  |  |

---

## 7. 筛选规则

### 7.1 直接保留

满足任一条件：

```text
mAP@50 >= 0.848
mAP@50:95 >= 0.660
Precision >= 0.850 且 Recall >= 0.760
scallop mAP@50 >= 0.683 且 Recall >= 0.580
```

下一步：

```text
用同一 YAML 跑 100 epochs 正式实验
```

### 7.2 边界保留

满足：

```text
mAP@50 接近 A8-0，但 Precision 明显提升
或 scallop 指标明显改善
```

下一步：

```text
可跑 100 epochs，但优先级低于直接保留模块
```

### 7.3 直接淘汰

出现任一情况：

```text
mAP@50 < 0.840
mAP@50:95 < 0.650
Recall < 0.740
scallop mAP@50 < 0.650
```

如果只提升 Recall，但 mAP 明显下降，也淘汰。

---

## 8. 100 epochs 复跑命令模板

筛出最好模块后，复跑 100 epochs：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/best.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  batch=96 \
  epochs=100 \
  imgsz=640 \
  workers=8 \
  device=0 \
  pretrained=weights/yolo11n.pt \
  cache=True \
  name=best_module_e100
```

100 epochs 后再决定是否：

```text
1. 包装成论文模块
2. 做插入位置消融
3. 做 imgsz=800 或 close_mosaic 调整
```

---

## 9. 论文包装建议

如果 S1 最好：

```text
UPA: Underwater Position Attention
```

叙事：

```text
利用坐标方向编码增强水下小目标的位置感知能力。
```

如果 S2 或 S4 最好：

```text
BSA: Background Suppression Attention
```

叙事：

```text
通过空间/多尺度注意力抑制水下复杂背景响应，提升检测精度。
```

如果 S5 最好：

```text
ULC: Underwater Local Context Block
```

叙事：

```text
在 P3 小目标分支建模多尺度局部上下文，提高弱纹理小目标表达。
```

注意：只有当 100 epochs 正式实验有效后，才进行论文命名包装。

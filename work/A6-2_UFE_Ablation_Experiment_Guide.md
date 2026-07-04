# A6-2 实验指导文档：UFE 消融与结构收敛

## 0. 实验背景

A6-1 已完成，结果如下：

| 实验 | 模型 | Precision | Recall | mAP@50 | mAP@50:95 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 3090 | YOLO11n | 0.867 | 0.753 | 0.848 | 0.658 | 2.58M | 6.3 |
| A6-1 | YOLO11n + UFE(P3) | 0.863 | 0.761 | 0.845 | 0.649 | 2.59M | 6.4 |

A6-1 的结论：

- UFE 计算开销很小，轻量性满足要求。
- Recall 有小幅提升，说明 UFE 对弱响应目标有一定增强作用。
- mAP@50 和 mAP@50:95 均下降，尤其 mAP@50:95 下降 0.009，说明当前增强方式可能影响定位质量或置信度排序。
- 因此，当前 UFE 不适合作为最终模型，也不建议马上进入 `YOLO11n + MSFF + UFE` 组合实验。

A6-2 的目标是通过消融实验找出 UFE 中导致 mAP@50:95 下降的结构，并收敛出更稳定的 UFE 版本。

---

## 1. 下一步实验总原则

下一步不要增加新模块，也不要和 MSFF 组合。  
只围绕 UFE 做减法消融：

```text
A6-1 原始 UFE
  = 局部细节增强
  + GAP/GMP 通道重标定
  + 空间弱目标门控
  + gamma 残差增强
```

A6-2 需要回答三个问题：

| 问题 | 对应实验 |
|---|---|
| max pooling 通道分支是否放大了水下背景噪声？ | A6-2-v1 |
| 空间弱目标门控是否干扰了边界框定位？ | A6-2-v2 |
| gamma 初始值是否过早改变了 baseline 特征分布？ | A6-2-v3 |

---

## 2. 实验编号与模型定义

| 实验编号 | 模型 | UFE 版本 | 核心改动 | 优先级 |
|---|---|---|---|---|
| A6-2-v1 | YOLO11n + UFE-GAP(P3) | UFE-v1 | 移除 GMP，仅保留 GAP 通道重标定 | 1 |
| A6-2-v2 | YOLO11n + UFE-DC(P3) | UFE-v2 | 移除空间门控，仅保留 Detail + Channel | 2 |
| A6-2-v3 | YOLO11n + UFE-G0(P3) | UFE-v3 | 保持 A6-1 结构，但 `init_gamma=0` | 3 |

建议执行顺序：

```text
先跑 A6-2-v1
如果 mAP@50:95 仍明显低于 A0，再跑 A6-2-v2
如果 A6-2-v1 或 A6-2-v2 有潜力，再补 A6-2-v3 判断 gamma 影响
```

---

## 3. 各版本设计说明

### 3.1 A6-2-v1：UFE-GAP

目的：判断 GMP 是否放大水下噪声。

A6-1 中通道门控使用：

```text
ChannelGate = MLP(GAP(X)) + MLP(GMP(X))
```

水下图像中存在高亮颗粒、反光、局部噪声点。GMP 可能过度响应这些局部强噪声，从而导致误增强。

A6-2-v1 改为：

```text
ChannelGate = MLP(GAP(X))
```

预期：

- Precision 和 mAP@50:95 应该回升。
- Recall 可能略低于 A6-1，但不能低于 A0 太多。
- 如果 A6-2-v1 优于 A6-1，说明 GMP 分支不适合当前 DUO 水下场景。

### 3.2 A6-2-v2：UFE-DC

目的：判断空间门控是否影响定位。

A6-1 中空间门控使用：

```text
SpatialGate = Conv(concat(avg_map, max_map))
Enhanced = Detail * ChannelGate * SpatialGate
```

如果空间门控产生的权重图过于粗糙，可能会削弱目标边缘或改变特征响应分布，导致 mAP@50:95 下降。

A6-2-v2 改为：

```text
Enhanced = Detail * ChannelGate
Y = X + gamma * Enhanced
```

预期：

- mAP@50:95 应该比 A6-1 更高。
- 如果 Recall 小幅下降但 mAP 回升，则说明空间门控过强。
- 如果 A6-2-v2 优于 A6-2-v1，可将最终 UFE 收敛为 Detail + Channel 的轻量结构。

### 3.3 A6-2-v3：UFE-G0

目的：判断残差增强强度初始化是否影响训练稳定性。

A6-1 中：

```text
gamma = 1e-3
```

A6-2-v3 改为：

```text
gamma = 0
```

这样训练开始时模型完全等价于 YOLO11n，UFE 分支从零影响开始学习。

预期：

- 训练更稳定。
- mAP@50:95 可能回升。
- 如果 A6-2-v3 优于 A6-1，说明 A6-1 的增强分支过早干扰了预训练特征。

---

## 4. 需要新增的模型配置

建议新增 3 个 YAML：

```text
ultralytics/cfg/models/11/yolo11n-ufe-gap-a6-2-v1.yaml
ultralytics/cfg/models/11/yolo11n-ufe-dc-a6-2-v2.yaml
ultralytics/cfg/models/11/yolo11n-ufe-g0-a6-2-v3.yaml
```

所有 YAML 的网络结构都保持 A6-1 的插入位置：

```yaml
- [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)
- [-1, 1, UFE_xxx, []]        # 17 UFE variant on P3
```

检测头保持：

```yaml
- [[17, 20, 23], 1, Detect, [nc]]
```

注意：A6-2 的重点是 UFE 结构消融，插入位置不要变化，否则无法与 A6-1 做干净对比。

---

## 5. 云服务器训练命令

### 5.1 A6-2-v1：UFE-GAP

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ufe-gap-a6-2-v1.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  pretrained=yolo11n.pt \
  name=A6_2_v1_yolo11n_ufe_gap_p3
```

### 5.2 A6-2-v2：UFE-DC

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ufe-dc-a6-2-v2.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  pretrained=yolo11n.pt \
  name=A6_2_v2_yolo11n_ufe_dc_p3
```

### 5.3 A6-2-v3：UFE-G0

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-ufe-g0-a6-2-v3.yaml \
  data=ultralytics/cfg/datasets/DUO.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  pretrained=yolo11n.pt \
  name=A6_2_v3_yolo11n_ufe_g0_p3
```

如果云服务器使用独立数据配置，将 `data=ultralytics/cfg/datasets/DUO.yaml` 替换为实际路径，例如：

```bash
data=/root/datasets/DUO.yaml
```

---

## 6. 验证命令

训练完成后分别验证 best.pt：

```bash
yolo detect val \
  model=runs/detect/A6_2_v1_yolo11n_ufe_gap_p3/weights/best.pt \
  data=ultralytics/cfg/datasets/DUO.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A6_2_v1_yolo11n_ufe_gap_p3_val
```

```bash
yolo detect val \
  model=runs/detect/A6_2_v2_yolo11n_ufe_dc_p3/weights/best.pt \
  data=ultralytics/cfg/datasets/DUO.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A6_2_v2_yolo11n_ufe_dc_p3_val
```

```bash
yolo detect val \
  model=runs/detect/A6_2_v3_yolo11n_ufe_g0_p3/weights/best.pt \
  data=ultralytics/cfg/datasets/DUO.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A6_2_v3_yolo11n_ufe_g0_p3_val
```

---

## 7. 成功标准

A6-2 的目标不是追求复杂结构，而是找出比 A6-1 更稳定的 UFE。

### 7.1 主成功标准

至少满足以下任一条件：

| 条件 | 标准 |
|---|---|
| mAP 恢复 | `mAP@50 >= 0.848` 且 `mAP@50:95 >= 0.655` |
| 召回增强 | `Recall >= 0.765` 且 `mAP@50 >= 0.848` |
| 弱类增强 | `scallop Recall >= 0.570` 且 `scallop mAP@50 >= 0.683` |

### 7.2 复杂度约束

| 指标 | 要求 |
|---|---|
| Params | 不超过 2.8M |
| GFLOPs | 不超过 6.8G |
| 模型大小 | 不明显超过 5.7MB |
| 推理速度 | 3090 上单张总耗时不明显高于 A6-1 |

### 7.3 淘汰标准

如果出现以下情况，直接不保留：

| 情况 | 判断 |
|---|---|
| `mAP@50 < 0.845` | 不优于 A6-1 |
| `mAP@50:95 < 0.649` | 定位质量进一步下降 |
| `Recall < 0.753` | 连 A0 云端 Recall 都不如 |
| scallop mAP@50 明显低于 0.670 | 弱类检测继续恶化 |

---

## 8. 实验记录表

### 8.1 整体指标

| 实验 | 模型 | Precision | Recall | mAP@50 | mAP@50:95 | Params | GFLOPs | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A0 | YOLO11n | 0.867 | 0.753 | 0.848 | 0.658 | 2.58M | 6.3 | baseline |
| A6-1 | YOLO11n + UFE | 0.863 | 0.761 | 0.845 | 0.649 | 2.59M | 6.4 | 边界有效 |
| A6-2-v1 | YOLO11n + UFE-GAP |  |  |  |  |  |  |  |
| A6-2-v2 | YOLO11n + UFE-DC |  |  |  |  |  |  |  |
| A6-2-v3 | YOLO11n + UFE-G0 |  |  |  |  |  |  |  |

### 8.2 各类别指标

| 实验 | 类别 | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---:|---:|---:|---:|
| A6-2-v1 | holothurian |  |  |  |  |
| A6-2-v1 | echinus |  |  |  |  |
| A6-2-v1 | scallop |  |  |  |  |
| A6-2-v1 | starfish |  |  |  |  |
| A6-2-v2 | holothurian |  |  |  |  |
| A6-2-v2 | echinus |  |  |  |  |
| A6-2-v2 | scallop |  |  |  |  |
| A6-2-v2 | starfish |  |  |  |  |
| A6-2-v3 | holothurian |  |  |  |  |
| A6-2-v3 | echinus |  |  |  |  |
| A6-2-v3 | scallop |  |  |  |  |
| A6-2-v3 | starfish |  |  |  |  |

---

## 9. 结果解释规则

| 结果现象 | 说明 | 下一步 |
|---|---|---|
| A6-2-v1 明显优于 A6-1 | GMP 分支可能放大噪声 | 后续 UFE 去掉 GMP |
| A6-2-v2 明显优于 A6-1 | 空间门控可能干扰定位 | 后续 UFE 保留 Detail + Channel |
| A6-2-v3 明显优于 A6-1 | gamma 初始化过强 | 后续使用 gamma=0 |
| 三个版本都不如 A0 | 当前 UFE 路线暂时不成立 | 回到 A2-Lite 或做水下数据增强 |
| 某版本 mAP 接近 A0 且 Recall 高于 A0 | 可进入 A7 组合实验 | 尝试 `MSFF(P3) + 最优 UFE` |

---

## 10. A6-2 后续决策

如果 A6-2 找到有效版本，例如：

```text
A6-2-v2: mAP@50 >= 0.848, mAP@50:95 >= 0.655, Recall > A0
```

则进入 A7：

```text
A7 = YOLO11n + MSFF(P3) + best-UFE(P3)
```

如果 A6-2 没有任何版本达到成功标准，则暂时放弃 UFE 结构主线，转向：

```text
A8 = YOLO11n + MSFF(P3) + underwater degradation augmentation
```

即把创新重点从结构模块转向水下退化鲁棒训练策略。

---

## 11. 推荐执行顺序

最推荐的实际执行顺序：

```text
1. 先实现并训练 A6-2-v1：UFE-GAP
2. 如果 A6-2-v1 的 mAP@50:95 没有恢复，训练 A6-2-v2：UFE-DC
3. 如果某个版本表现接近 A0，再训练 A6-2-v3：gamma=0
4. 选出 best-UFE 后，再决定是否进入 A7
```

不要一次性提交最终结论。A6-2 的意义是把 UFE 从“想法模块”收敛成“有证据支撑的论文模块”。

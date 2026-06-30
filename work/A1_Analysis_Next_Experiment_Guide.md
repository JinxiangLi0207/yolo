# A1 SPDConv 实验分析与下一步实验指导文档

文件名建议：`A1_Analysis_Next_Experiment_Guide.md`

---

## 1. 当前实验结论

你已经完成：

```text
A0: YOLO11n baseline
A1: YOLO11n + SPDConv
```

A1 的改动为：

```text
Backbone P2/4、P3/8 两处下采样 Conv 替换为 SPDConv
```

A1 相比 A0 的主要结果如下：

| 指标 | A0 YOLO11n | A1 YOLO11n + SPDConv | 变化 |
|---|---:|---:|---:|
| Precision | 0.848 | 0.842 | -0.006 |
| Recall | 0.762 | 0.774 | +0.012 |
| mAP@50 | 0.849 | 0.841 | -0.008 |
| mAP@50:95 | 0.656 | 0.654 | -0.002 |
| Params | 2.58M | 2.71M | +4.8% |
| GFLOPs | 6.3 | 8.4 | +33.3% |
| FPS | ~455 | ~417 | -38 |

各类别中，最值得关注的是 scallop：

| 类别 | A0 Recall | A1 Recall | 变化 | A0 mAP@50 | A1 mAP@50 | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| scallop | 0.552 | 0.592 | +0.040 | 0.684 | 0.662 | -0.022 |

---

## 2. 对 A1 的判断

A1 是一个“边界有效”的实验，不建议直接作为最终模型，但建议暂时保留为候选模块。

### 2.1 积极信号

SPDConv 明显提高了整体 Recall 和 scallop Recall：

```text
整体 Recall: 0.762 → 0.774
scallop Recall: 0.552 → 0.592
```

这说明 SPDConv 对“减少漏检”是有帮助的，尤其是对 DUO 中最弱的 scallop 类别。

### 2.2 消极信号

A1 的整体精度下降：

```text
mAP@50: 0.849 → 0.841
mAP@50:95: 0.656 → 0.654
```

同时 GFLOPs 增加明显：

```text
6.3G → 8.4G
```

说明当前 P2/4 + P3/8 两处替换可能过强。它增加了小目标候选响应，但也可能引入更多误检，导致 Precision 和 AP 下降。

### 2.3 当前结论

```text
SPDConv 可以暂时保留，但不能直接认定有效。
```

后续应通过两条线验证：

```text
1. A2: 单独验证 MSFF 是否能提升特征区分能力；
2. A1-OnlyP2: 降低 SPDConv 插入强度，观察是否保留 Recall 提升并恢复 mAP。
```

---

## 3. 下一步优先级

当前不建议直接做：

```text
A5 = YOLO11n + SPDConv + MSFF
```

因为 A1 本身还不是稳定正收益。直接组合会导致问题难以定位。

推荐下一步顺序为：

```text
A2: YOLO11n + MSFF
A1-OnlyP2: YOLO11n + SPDConv，仅替换 P2/4
A5: YOLO11n + 最优 SPDConv 版本 + MSFF
```

优先做 A2。A2 可以判断 MSFF 是否能解决 A1 中“Recall 提高但 mAP 下降”的问题。

---

# 4. A2 实验：YOLO11n + MSFF

## 4.1 实验目标

A2 的目标是验证 MSFF 是否能增强检测头前的多尺度特征表达，使模型更好地区分目标和复杂水下背景。

实验编号：

```text
A2
```

模型：

```text
YOLO11n + MSFF
```

注意：A2 不加入 SPDConv、不加入 GSConv、不改 Loss。

---

## 4.2 为什么下一步做 MSFF

A1 中 SPDConv 提升了 scallop Recall，但 mAP 下降。这个现象说明模型可能检测出了更多目标，但误检或排序质量变差。

MSFF 的目标是增强特征区分度，理论上可能带来：

```text
1. 保持或提升 Recall；
2. 减少误检；
3. 提升 Precision；
4. 提升 mAP@50 和 mAP@50:95；
5. 对 scallop 这类低样本、小目标类别更有帮助。
```

因此 A2 是当前最合理的下一步。

---

## 4.3 MSFF 推荐插入位置

MSFF 应放在 Detect Head 前，而不是 Backbone 或 Neck 的中间。

推荐结构：

```text
P3 → MSFF → Detect
P4 → MSFF → Detect
P5 → MSFF → Detect
```

但考虑你的显存约 8GB，建议分两个版本：

| 实验编号 | 插入方式 | 目的 |
|---|---|---|
| A2-Lite | 只在 P3 检测分支前加入 MSFF | 低显存、低风险，先验证小目标分支效果 |
| A2-Full | 在 P3/P4/P5 三个检测分支前加入 MSFF | 更完整验证 MSFF |

推荐先做：

```text
A2-Lite
```

如果 A2-Lite 不爆显存且结果有正向趋势，再做 A2-Full。

---

# 5. A2-Lite 实现建议

## 5.1 添加 MSFF 模块代码

打开：

```text
ultralytics/nn/modules/conv.py
```

在文件中添加如下 MSFF-Lite 模块。

> 说明：这是适合 YOLO11n 的轻量适配版 MSFF，用于 A2 快速验证。它不是直接照搬 UWNet 的完整实现，而是保留“多尺度深度卷积 + 隐式乘法增强 + 残差连接”的核心思想，便于在 8GB 显存环境下测试。

```python
class MSFF(nn.Module):
    """
    MSFF-Lite: Multi-Scale Implicit Feature Fusion module.

    Recommended position:
        before Detect head inputs.

    Input:
        x: [B, C, H, W]

    Output:
        y: [B, C, H, W]

    Design:
        1. average pooling to reduce local redundancy
        2. multi-scale depthwise separable spatial branches
        3. element-wise multiplication for implicit feature interaction
        4. sigmoid gating
        5. residual enhancement
    """

    def __init__(self, c1, c2=None, act=True):
        super().__init__()
        c2 = c1 if c2 is None else c2

        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.dw3_h = nn.Conv2d(c1, c1, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=c1, bias=False)
        self.dw3_v = nn.Conv2d(c1, c1, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=c1, bias=False)

        self.dw5_h = nn.Conv2d(c1, c1, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=c1, bias=False)
        self.dw5_v = nn.Conv2d(c1, c1, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=c1, bias=False)

        self.dw7_h = nn.Conv2d(c1, c1, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=c1, bias=False)
        self.dw7_v = nn.Conv2d(c1, c1, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=c1, bias=False)

        self.proj = Conv(c1, c2, k=1, s=1, act=act)

    def forward(self, x):
        z = self.avg(x)

        b3 = self.dw3_v(self.dw3_h(z))
        b5 = self.dw5_v(self.dw5_h(z))
        b7 = self.dw7_v(self.dw7_h(z))

        gate = torch.sigmoid(b3) * torch.sigmoid(b5) * torch.sigmoid(b7)
        y = x + x * gate

        return self.proj(y)
```

确认 `conv.py` 顶部已有：

```python
import torch
import torch.nn as nn
```

---

## 5.2 导出 MSFF

打开：

```text
ultralytics/nn/modules/__init__.py
```

把 `MSFF` 加入 `.conv` 的 import 列表，例如：

```python
from .conv import (
    Conv,
    ...
    MSFF,
)
```

如果有 `__all__`，也加入：

```python
"MSFF",
```

---

## 5.3 在 tasks.py 注册 MSFF

打开：

```text
ultralytics/nn/tasks.py
```

在模块导入区加入：

```python
MSFF,
```

然后在 `parse_model()` 的 `base_modules` 中加入：

```python
MSFF,
```

目标是让 YAML 中的：

```yaml
- [-1, 1, MSFF, [256]]
```

可以被解析成：

```python
MSFF(c1, c2)
```

如果你的 Ultralytics 版本没有 `base_modules`，就搜索包含 `Conv` 的解析分支，把 `MSFF` 加入和 `Conv` 同一组。

---

# 6. A2-Lite YAML 修改方式

复制一份 YOLO11 YAML：

```bash
cp ultralytics/cfg/models/11/yolo11.yaml ultralytics/cfg/models/11/yolo11n-msff-a2-lite.yaml
```

Windows PowerShell：

```powershell
Copy-Item ultralytics/cfg/models/11/yolo11.yaml ultralytics/cfg/models/11/yolo11n-msff-a2-lite.yaml
```

打开：

```text
ultralytics/cfg/models/11/yolo11n-msff-a2-lite.yaml
```

找到最后的 Detect 层，原始结构通常类似：

```yaml
- [[16, 19, 22], 1, Detect, [nc]]
```

A2-Lite 只在 P3 分支前加入 MSFF。

修改为类似：

```yaml
- [16, 1, MSFF, [256]]          # P3 MSFF
- [[23, 19, 22], 1, Detect, [nc]]
```

注意：

```text
1. 这里的 16、19、22、23 只是常见 YOLO11n YAML 的层号；
2. 你的实际层号必须以当前 yolo11.yaml 为准；
3. 原 Detect 输入的第一个分支通常是 P3，小目标检测分支；
4. 插入 MSFF 后，Detect 的 P3 输入要改成新生成的 MSFF 层。
```

如果你的原 Detect 是：

```yaml
- [[A, B, C], 1, Detect, [nc]]
```

那么 A2-Lite 改为：

```yaml
- [A, 1, MSFF, [对应A的通道数]]
- [[新MSFF层号, B, C], 1, Detect, [nc]]
```

---

# 7. A2-Full YAML 修改方式

如果 A2-Lite 成功，再创建 A2-Full：

```bash
cp ultralytics/cfg/models/11/yolo11.yaml ultralytics/cfg/models/11/yolo11n-msff-a2-full.yaml
```

原始 Detect：

```yaml
- [[16, 19, 22], 1, Detect, [nc]]
```

改为：

```yaml
- [16, 1, MSFF, [256]]          # P3 MSFF
- [19, 1, MSFF, [512]]          # P4 MSFF
- [22, 1, MSFF, [1024]]         # P5 MSFF
- [[23, 24, 25], 1, Detect, [nc]]
```

如果你的层号不同，以原始 Detect 输入为准。

---

# 8. A2 Smoke Test

先跑 1 epoch 确认模型能构建、训练和验证。

A2-Lite：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-msff-a2-lite.yaml \
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
  name=A2_msff_lite_smoke
```

如果成功，再跑正式 100 epochs。

---

# 9. A2 正式训练命令

为了和 A0、A1 公平比较，训练参数保持一致：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-msff-a2-lite.yaml \
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
  name=A2_yolo11n_msff_lite
```

如果 8GB 显存爆显存：

```text
batch=8
workers=2
```

如果还爆显存：

```text
batch=4
```

A2-Full 只有在 A2-Lite 成功且显存允许时再跑：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-msff-a2-full.yaml \
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
  workers=2 \
  name=A2_yolo11n_msff_full
```

---

# 10. A2 结果记录模板

新建：

```text
experiments/A2_MSFF_Report.md
```

记录：

```markdown
# A2 YOLO11n + MSFF 实验报告

## 实验设置

| 项目 | 内容 |
|---|---|
| 实验编号 | A2 |
| 模型 | YOLO11n + MSFF |
| MSFF版本 | Lite / Full |
| 插入位置 | P3 only / P3-P4-P5 |
| 数据集 | DUO |
| Epochs | 100 |
| Batch |  |
| imgsz | 640 |
| optimizer | SGD |
| lr0 | 0.01 |
| pretrained | yolo11n.pt |

## 模型复杂度

| 指标 | A0 YOLO11n | A1 SPDConv | A2 MSFF |
|---|---:|---:|---:|
| Params | 2.58M | 2.71M |  |
| GFLOPs | 6.3 | 8.4 |  |
| FPS | 455 | 417 |  |

## 整体结果

| 指标 | A0 YOLO11n | A1 SPDConv | A2 MSFF | A2-A0 |
|---|---:|---:|---:|---:|
| Precision | 0.848 | 0.842 |  |  |
| Recall | 0.762 | 0.774 |  |  |
| mAP@50 | 0.849 | 0.841 |  |  |
| mAP@50:95 | 0.656 | 0.654 |  |  |

## 各类别结果

| 类别 | A0 mAP@50 | A1 mAP@50 | A2 mAP@50 | A0 Recall | A1 Recall | A2 Recall |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.860 | 0.852 |  | 0.786 | 0.787 |  |
| echinus | 0.925 | 0.925 |  | 0.845 | 0.851 |  |
| scallop | 0.684 | 0.662 |  | 0.552 | 0.592 |  |
| starfish | 0.927 | 0.927 |  | 0.865 | 0.865 |  |

## 判断结论

- 如果 A2 的 mAP@50 和 mAP@50:95 均超过 A0，则 MSFF 是有效模块；
- 如果 A2 的 Precision 提升，但 Recall 下降，可以考虑后续与 SPDConv 组合；
- 如果 A2 对 scallop mAP 或 Recall 有明显提升，优先保留；
- 如果 A2 整体下降且 scallop 无提升，则不保留当前 MSFF 插入方式；
- 如果 A2-Lite 有效，再尝试 A2-Full；
- 如果 A2-Full 比 A2-Lite 更差，最终保留 A2-Lite。
```

---

# 11. A2 成功标准

A2 推荐保留条件：

```text
条件 1：mAP@50:95 比 A0 提升 >= 0.3%
或
条件 2：mAP@50 比 A0 提升 >= 0.5%
或
条件 3：scallop mAP@50 提升 >= 2%
或
条件 4：scallop Recall 提升 >= 2%，且整体 mAP 不明显下降
```

更理想的结果：

```text
A2 mAP@50 >= 0.855
A2 mAP@50:95 >= 0.662
scallop mAP@50 >= 0.700
scallop Recall >= 0.580
```

---

# 12. A2 后的决策树

## 情况 1：A2 明显优于 A0

例如：

```text
A2 mAP@50 > 0.849
A2 mAP@50:95 > 0.656
```

下一步：

```text
A5 = YOLO11n + MSFF + SPDConv
```

但优先使用 A1-OnlyP2，而不是当前 P2+P3 两处 SPDConv。

---

## 情况 2：A2 提升 Precision，但 Recall 不如 A1

这说明：

```text
MSFF 更擅长减少误检；
SPDConv 更擅长减少漏检。
```

下一步：

```text
A5 = SPDConv + MSFF
```

这是最值得期待的组合，因为二者可能互补。

---

## 情况 3：A2 整体下降

不要直接做 A5-Full。

先做：

```text
A1-OnlyP2
```

验证是否可以保留 SPDConv 的 scallop Recall 提升，同时降低 GFLOPs 和 mAP 损失。

---

## 情况 4：A2-Lite 有效，但 A2-Full 无效

最终只保留 P3-only MSFF。

这种情况很常见，因为 P3 是小目标分支，P4/P5 加强不一定对 DUO 小目标有帮助，反而可能干扰语义表达。

---

# 13. A1-OnlyP2 补充实验

A1 当前替换了 P2/4 和 P3/8 两处下采样，GFLOPs 增加较大，mAP 略降。

建议补充一个轻量版本：

```text
A1-OnlyP2 = YOLO11n + SPDConv，仅替换 P2/4
```

目的：

```text
1. 降低 GFLOPs；
2. 检查 scallop Recall 是否仍能提升；
3. 判断 P3/8 替换是否导致 mAP 下降。
```

A1-OnlyP2 成功标准：

```text
mAP@50:95 >= 0.656
或
scallop Recall >= 0.580 且 mAP@50 不低于 0.845
```

如果 A1-OnlyP2 比 A1 更好，后续 A5 使用 A1-OnlyP2 版本。

---

# 14. 当前推荐实验路线

最终建议你按以下顺序执行：

```text
1. A2-Lite: YOLO11n + MSFF(P3-only)
2. A2-Full: YOLO11n + MSFF(P3/P4/P5)，仅在 A2-Lite 有效或显存允许时执行
3. A1-OnlyP2: YOLO11n + SPDConv(P2-only)
4. A5: YOLO11n + 最优 SPDConv 版本 + 最优 MSFF 版本
```

不要跳过 A2 直接做 A5。

---

# 15. 当前阶段不要做的事

暂时不要加入 GSConv。

暂时不要修改 IoU Loss。

暂时不要调整数据增强。

暂时不要换 YOLO11s/m。

暂时不要改 epochs、imgsz、optimizer。

原因：

```text
你现在还在验证单个模块是否有效。
如果同时改多个变量，后续无法解释提升来源。
```

---

# 16. 最终建议

A1 结果说明 SPDConv 具有一定价值，但当前插入方式并不理想。

最合理判断是：

```text
SPDConv 有助于提高小目标召回率；
但 P2/4 + P3/8 两处替换导致整体 mAP 略降、计算量增加偏大；
需要通过 MSFF 和 A1-OnlyP2 进一步验证。
```

下一步请优先执行：

```text
A2-Lite = YOLO11n + MSFF(P3-only)
```

执行完成后，把 A2 报告发给我，再决定：

```text
A2-Full
A1-OnlyP2
A5 组合实验
```

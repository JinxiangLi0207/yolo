# A1 实验：在 YOLO11n 中插入 SPDConv 模块

## 1. 实验目标

本实验编号为 **A1**。

目标是在 A0 baseline 的基础上，将 YOLO11n Backbone 中部分普通下采样卷积替换为 **SPDConv**，用于减少水下小目标在下采样过程中的空间细节损失。

A0 baseline 为：

```text
YOLO11n
mAP@50 = 0.849
mAP@50:95 = 0.656
Params = 2.58M
GFLOPs = 6.3
```

A1 目标：

```text
YOLO11n + SPDConv
```

重点观察：

```text
1. 总体 mAP@50 是否超过 0.849
2. 总体 mAP@50:95 是否超过 0.656
3. scallop 类别 Recall 是否提升
4. scallop 类别 mAP@50 是否提升
5. Params 和 GFLOPs 是否可控
```

---

## 2. SPDConv 插入原则

SPDConv 不建议一开始全网络替换。

本实验只在 **Backbone 浅层下采样位置** 插入 SPDConv。

推荐优先替换：

```text
Backbone layer 1: P2/4 下采样
Backbone layer 3: P3/8 下采样
```

不建议第一轮替换：

```text
layer 0: P1/2 stem Conv
layer 5: P4/16 下采样
layer 7: P5/32 下采样
```

原因：

```text
1. layer 0 直接处理 RGB 输入，替换后可能影响底层纹理稳定性；
2. P4/P5 已经是较深层语义特征，小目标细节大多已经在浅层损失；
3. 8GB 显存下，第一轮应优先使用轻量替换方案；
4. A1 只需要验证 SPDConv 是否对 DUO 小目标有效，不要一次改太多。
```

---

## 3. 添加 SPDConv 代码

打开文件：

```text
ultralytics/nn/modules/conv.py
```

在 `Conv` 类后面添加以下代码。

```python
class SPDConv(nn.Module):
    """
    SPDConv: Space-to-Depth Convolution.

    This module replaces a stride-2 convolution with:
    1. space-to-depth rearrangement with block size 2
    2. normal convolution with stride 1

    Input:
        x: [B, C, H, W]

    Output:
        y: [B, C_out, H/2, W/2]

    Notes:
        - H and W should be divisible by 2.
        - This module is intended to replace downsampling Conv layers with stride=2.
        - In YOLO yaml, use the same argument style as Conv: [c2, k, s].
    """

    default_act = Conv.default_act

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()

        if s != 2:
            raise ValueError(
                f"SPDConv is designed to replace stride-2 Conv only, but got s={s}."
            )

        self.block_size = 2
        self.conv = Conv(
            c1 * self.block_size * self.block_size,
            c2,
            k=k,
            s=1,
            p=p,
            g=g,
            d=d,
            act=act,
        )

    def forward(self, x):
        b, c, h, w = x.shape

        if h % 2 != 0 or w % 2 != 0:
            raise RuntimeError(
                f"SPDConv requires even H and W, but got H={h}, W={w}."
            )

        # Space-to-depth, block size = 2
        x = torch.cat(
            [
                x[..., 0::2, 0::2],
                x[..., 1::2, 0::2],
                x[..., 0::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            dim=1,
        )

        return self.conv(x)
```

确认 `conv.py` 文件顶部已有：

```python
import torch
import torch.nn as nn
```

如果没有，就补上。

---

## 4. 导出 SPDConv 模块

打开文件：

```text
ultralytics/nn/modules/__init__.py
```

找到类似下面的导入：

```python
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
```

把 `SPDConv` 加进去：

```python
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SPDConv,
    SpatialAttention,
)
```

如果文件中有 `__all__`，也把 `SPDConv` 加进去：

```python
__all__ = (
    ...
    "SPDConv",
    ...
)
```

---

## 5. 在 tasks.py 中注册 SPDConv

打开文件：

```text
ultralytics/nn/tasks.py
```

找到从 `ultralytics.nn.modules` 导入模块的位置。

例如：

```python
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3k2,
    Concat,
    Conv,
    Detect,
    ...
)
```

加入：

```python
SPDConv,
```

即类似：

```python
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3k2,
    Concat,
    Conv,
    SPDConv,
    Detect,
    ...
)
```

然后在 `parse_model()` 函数中查找：

```python
base_modules = frozenset({
```

或类似包含 `Conv`, `C3k2`, `C2f`, `SPPF` 的模块集合。

把 `SPDConv` 加进去：

```python
base_modules = frozenset({
    ...
    Conv,
    SPDConv,
    ...
})
```

如果你的 Ultralytics 版本没有 `base_modules`，就搜索：

```python
if m in {
```

找到包含 `Conv` 的模块解析分支，把 `SPDConv` 加到和 `Conv` 同一组的模块里。

目标是让 `SPDConv` 按照普通 `Conv` 的方式解析通道：

```python
c1 = ch[f]
c2 = args[0]
args = [c1, c2, *args[1:]]
```

否则 YAML 中写 `SPDConv` 会出现通道解析错误。

---

## 6. 创建 YOLO11n-SPDConv 模型 YAML

找到原始 YOLO11 配置文件，通常在：

```text
ultralytics/cfg/models/11/yolo11.yaml
```

复制一份：

```bash
cp ultralytics/cfg/models/11/yolo11.yaml ultralytics/cfg/models/11/yolo11n-spd-a1.yaml
```

Windows PowerShell：

```powershell
Copy-Item ultralytics/cfg/models/11/yolo11.yaml ultralytics/cfg/models/11/yolo11n-spd-a1.yaml
```

---

## 7. 修改 YAML 中的 Backbone

打开：

```text
ultralytics/cfg/models/11/yolo11n-spd-a1.yaml
```

找到 Backbone 部分。

原始结构通常类似：

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

### 7.1 A1-Lite 推荐版

先只替换 P2/4 和 P3/8 两处下采样：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, SPDConv, [128, 3, 2]]    # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, SPDConv, [256, 3, 2]]    # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
```

不要修改 Head 的层号。

因为只替换模块类型，不改变输出特征图尺寸，也不改变层数量，所以 Head 中的索引一般不需要改。

---

## 8. 快速检查模块是否能被识别

运行：

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-spd-a1.yaml")
print(model.model)
model.info()
PY
```

如果出现：

```text
KeyError: 'SPDConv'
```

说明 `tasks.py` 没有正确导入或注册 `SPDConv`。

如果出现：

```text
NameError: name 'SPDConv' is not defined
```

说明 `ultralytics/nn/modules/__init__.py` 或 `tasks.py` 没有导入成功。

如果出现通道错误，例如：

```text
expected input channel ...
```

说明 `parse_model()` 没有把 `SPDConv` 放进和 `Conv` 一样的模块解析分支。

---

## 9. 测试前向传播

运行：

```bash
python - <<'PY'
import torch
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11n-spd-a1.yaml").model
model.eval()

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    y = model(x)

print("Forward success.")
print(type(y))
PY
```

如果前向传播成功，说明结构基本可用。

---

## 10. 查看模型参数量和 GFLOPs

运行：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-spd-a1.yaml \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  epochs=1 \
  batch=4 \
  device=0 \
  name=A1_spd_smoke_test
```

只跑 1 个 epoch，目的是确认：

```text
1. 模型可以构建；
2. 数据可以正常读取；
3. loss 正常下降；
4. 没有显存爆炸；
5. 日志中可以看到 Params 和 GFLOPs。
```

---

## 11. 正式 A1 训练命令

为了和 A0 公平对比，A1 使用相同训练轮数和主要超参数。

推荐命令：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-spd-a1.yaml \
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
  name=A1_yolo11n_spd
```

如果 8GB 显存爆显存，改成：

```bash
batch=8
```

如果还爆显存：

```bash
batch=4
workers=2
```

---

## 12. 训练后验证命令

训练完成后运行：

```bash
yolo detect val \
  model=runs/detect/A1_yolo11n_spd/weights/best.pt \
  data=datasets/DUO-YOLO/duo.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=A1_yolo11n_spd_val
```

如果验证爆显存：

```bash
batch=8
```

---

## 13. A1 实验记录模板

训练完成后，新建：

```text
experiments/A1_SPDConv_Report.md
```

记录以下内容：

```markdown
# A1 YOLO11n + SPDConv 实验报告

## 实验设置

| 项目 | 内容 |
|---|---|
| 实验编号 | A1 |
| 模型 | YOLO11n + SPDConv |
| 数据集 | DUO |
| 修改位置 | Backbone P2/4、P3/8 下采样 Conv 替换为 SPDConv |
| Epochs | 100 |
| Batch | 16 |
| imgsz | 640 |
| optimizer | SGD |
| lr0 | 0.01 |
| pretrained | yolo11n.pt |

## 模型复杂度

| 指标 | A0 YOLO11n | A1 YOLO11n+SPDConv |
|---|---:|---:|
| Params | 2.58M |  |
| GFLOPs | 6.3 |  |
| Model size | 5.5MB |  |

## 整体结果

| 指标 | A0 YOLO11n | A1 YOLO11n+SPDConv | 差值 |
|---|---:|---:|---:|
| Precision | 0.848 |  |  |
| Recall | 0.762 |  |  |
| mAP@50 | 0.849 |  |  |
| mAP@50:95 | 0.656 |  |  |

## 各类别结果

| 类别 | A0 mAP@50 | A1 mAP@50 | 差值 | A0 Recall | A1 Recall | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| holothurian | 0.860 |  |  | 0.786 |  |  |
| echinus | 0.925 |  |  | 0.845 |  |  |
| scallop | 0.684 |  |  | 0.552 |  |  |
| starfish | 0.927 |  |  | 0.865 |  |  |

## 判断结论

- 如果整体 mAP@50 和 mAP@50:95 均提升，保留 SPDConv。
- 如果整体 mAP 略降，但 scallop Recall 和 mAP 明显提升，暂时保留，后续与 MSFF 组合再观察。
- 如果整体 mAP 下降且 scallop 没提升，不保留当前插入位置。
- 如果 Params/GFLOPs 增加过多，尝试只替换 P2/4 一处。
```

---

## 14. 如果 A1 效果不好，尝试 A1-OnlyP2

如果 A1-Lite 的 P2/4 + P3/8 替换效果不好，可以创建另一个 YAML：

```text
yolo11n-spd-a1-p2only.yaml
```

只替换 layer 1：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, SPDConv, [128, 3, 2]]    # 1-P2/4
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

训练命令：

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11n-spd-a1-p2only.yaml \
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
  name=A1_yolo11n_spd_p2only
```

---

## 15. 常见错误处理

### 15.1 KeyError: SPDConv

原因：

```text
tasks.py 没有导入 SPDConv
```

处理：

```text
检查 ultralytics/nn/tasks.py 的 import 列表。
```

### 15.2 通道数不匹配

原因：

```text
parse_model() 没有把 SPDConv 当作 Conv 类模块处理。
```

处理：

```text
把 SPDConv 加入 base_modules，或者加入包含 Conv 的解析分支。
```

### 15.3 RuntimeError: SPDConv requires even H and W

原因：

```text
输入特征图 H 或 W 不是偶数。
```

处理：

```text
确认 imgsz=640；
确认 SPDConv 只替换 stride=2 的下采样 Conv；
不要把 SPDConv 放在任意奇怪位置。
```

### 15.4 训练显存不足

处理顺序：

```text
1. batch=16 改 batch=8
2. batch=8 改 batch=4
3. workers=4 改 workers=2
4. 只替换 P2/4 一处
```

---

## 16. A1 是否成功的判断标准

A1 可以进入后续组合实验的条件：

```text
条件 1：mAP@50 或 mAP@50:95 至少一个提升；
条件 2：scallop Recall 或 scallop mAP@50 有明显提升；
条件 3：Params 和 GFLOPs 没有不可接受地增加；
条件 4：训练稳定，没有明显 loss 异常。
```

推荐保留标准：

```text
mAP@50:95 提升 >= 0.3%
或 scallop Recall 提升 >= 2%
```

如果 A1 整体提升不大，但 scallop 类别提升明显，也可以继续保留到 A5：

```text
YOLO11n + SPDConv + MSFF
```

因为 SPDConv 可能主要改善小目标细节，而 MSFF 后续可能进一步增强多尺度特征表达。

---

## 17. 本阶段不要做的事

不要同时加入 MSFF。

不要同时加入 GSConv。

不要同时修改 Loss。

不要调整数据增强策略。

不要更换 batch、epoch、imgsz 后直接和 A0 对比。

不要在同一次实验中替换所有下采样卷积。

A1 的核心目标只有一个：

```text
验证 SPDConv 替换浅层下采样卷积是否对 DUO 小目标检测有效。
```

"""
YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection
核心模块实现

包含：
1. MS-Block: 多尺度卷积块，使用多个分支提取不同尺度特征
2. InceptionDWConvBlock: 基于深度可分离卷积的Inception风格模块
3. C2f_MSBlock: 集成MS-Block的C2f结构
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class DWConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, s, k // 2 if isinstance(k, int) else [x // 2 for x in k],
                              groups=c1, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MSBlock(nn.Module):
    """
    Multi-Scale Block (MS-Block)

    论文: YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection

    核心思想：
    - 使用多个分支提取不同尺度的特征
    - 分支1: 1x1卷积 (捕获点级特征)
    - 分支2: 3x3卷积 (捕获小尺度特征)
    - 分支3: 5x5卷积 (使用两个3x3串联实现，捕获中尺度特征，减少参数)
    - 最后通过1x1卷积融合多尺度特征

    Args:
        c1: 输入通道数 (由attn_modules自动注入)
        c2: 输出通道数 (YAML中指定)
        n: 堆叠次数
        shortcut: 是否使用残差连接
    """

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.c = c2 // 4  # 每个分支的通道数
        self.cv1 = Conv(c1, c2, 1)  # 输入卷积

        # 多分支卷积
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                Conv(self.c, self.c, 3),
                Conv(self.c, self.c, 3)
            ),  # 分支1: 两个3x3串联 (等效5x5)
            nn.Sequential(
                Conv(self.c, self.c, 3),
                Conv(self.c, self.c, 3)
            ),  # 分支2: 两个3x3串联 (等效5x5)
            nn.Sequential(
                Conv(self.c, self.c, 3),
                Conv(self.c, self.c, 3)
            ),  # 分支3: 两个3x3串联 (等效5x5)
        ])

        self.cv3 = Conv(c2, c2, 1)  # 输出融合卷积
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        """前向传播"""
        y = self.cv1(x)

        # 分通道处理
        y_split = torch.split(y, self.c, dim=1)

        # 多分支特征提取
        y1 = self.cv2[0](y_split[0])
        y2 = self.cv2[1](y_split[1])
        y3 = self.cv2[2](y_split[2])
        y4 = y_split[3]  # 直接传递

        # 拼接并融合
        y = self.cv3(torch.cat([y1, y2, y3, y4], dim=1))

        # 残差连接
        return y + x if self.shortcut else y


class InceptionDWConvBlock(nn.Module):
    """
    Inception风格的深度可分离卷积块

    使用不同大小的深度卷积核提取多尺度特征：
    - 分支1: 3x3 深度卷积
    - 分支2: 5x5 深度卷积 (使用两个3x3串联实现)
    - 分支3: 7x7 深度卷积 (使用三个3x3串联实现)

    Args:
        c1: 输入通道数 (由attn_modules自动注入)
        c2: 输出通道数 (YAML中指定)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 4

        # 输入投影
        self.cv1 = Conv(c1, c2, 1)

        # 多尺度深度卷积分支
        self.dw_conv = nn.ModuleList([
            DWConv(self.c, self.c, 3),  # 3x3
            nn.Sequential(
                DWConv(self.c, self.c, 3),
                DWConv(self.c, self.c, 3)
            ),  # 5x5 (两个3x3)
            nn.Sequential(
                DWConv(self.c, self.c, 3),
                DWConv(self.c, self.c, 3),
                DWConv(self.c, self.c, 3)
            ),  # 7x7 (三个3x3)
        ])

        # 输出投影
        self.cv2 = Conv(c2, c2, 1)

    def forward(self, x):
        # 输入投影
        y = self.cv1(x)

        # 分通道处理
        y_split = torch.split(y, self.c, dim=1)

        # 多尺度特征提取
        y1 = self.dw_conv[0](y_split[0])
        y2 = self.dw_conv[1](y_split[1])
        y3 = self.dw_conv[2](y_split[2])
        y4 = y_split[3]  # 直接传递

        # 拼接并融合
        return self.cv2(torch.cat([y1, y2, y3, y4], dim=1))


class C2f_MSBlock(nn.Module):
    """
    C2f结构集成MS-Block

    将MS-Block集成到C2f结构中，保持C2f的梯度流特性

    Args:
        c1: 输入通道数 (由attn_modules自动注入)
        c2: 输出通道数 (YAML中指定)
        n: MS-Block堆叠次数
        shortcut: 是否使用残差连接
        g: 分组卷积的组数
        e: 通道扩展系数
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # MS-Block模块
        self.m = nn.ModuleList(
            MSBlock(self.c, self.c, n=1, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x):
        """前向传播"""
        # 分支处理
        y = list(self.cv1(x).chunk(2, 1))

        # 通过MS-Block
        for m in self.m:
            y.append(m(y[-1]))

        # 融合
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用split的前向传播（更节省内存）"""
        y = list(self.cv1(x).split((self.c, self.c), 1))

        for m in self.m:
            y.append(m(y[-1]))

        return self.cv2(torch.cat(y, 1))


class MSBlock_S(nn.Module):
    """
    轻量级MS-Block (Small版本)

    使用更少的分支和更小的计算量

    Args:
        c1: 输入通道数 (由attn_modules自动注入)
        c2: 输出通道数 (YAML中指定)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2

        # 输入卷积
        self.cv1 = Conv(c1, c2, 1)

        # 双分支卷积
        self.cv2 = nn.ModuleList([
            Conv(self.c, self.c, 3),  # 3x3
            nn.Sequential(
                Conv(self.c, self.c, 3),
                Conv(self.c, self.c, 3)
            ),  # 5x5
        ])

        # 输出融合
        self.cv3 = Conv(c2, c2, 1)

    def forward(self, x):
        y = self.cv1(x)
        y_split = torch.split(y, self.c, dim=1)

        y1 = self.cv2[0](y_split[0])
        y2 = self.cv2[1](y_split[1])

        return self.cv3(torch.cat([y1, y2], dim=1))


if __name__ == "__main__":
    # 测试代码
    x = torch.randn(2, 64, 80, 80)

    # 测试MSBlock
    ms = MSBlock(64, 64, n=1)
    y = ms(x)
    print(f"MSBlock: {x.shape} -> {y.shape}")

    # 测试InceptionDWConvBlock
    inception = InceptionDWConvBlock(64, 64)
    y = inception(x)
    print(f"InceptionDWConvBlock: {x.shape} -> {y.shape}")

    # 测试C2f_MSBlock
    c2f_ms = C2f_MSBlock(64, 64, n=2)
    y = c2f_ms(x)
    print(f"C2f_MSBlock: {x.shape} -> {y.shape}")

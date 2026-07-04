import torch
import torch.nn as nn


class UFE(nn.Module):
    """
    Underwater Feature Enhancement module.

    UFE is a lightweight residual enhancement block for underwater small-object
    detection. It keeps the input/output shape unchanged and combines local
    detail enhancement, channel recalibration, and spatial weak-target gating.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        detail_kernel: int = 5,
        init_gamma: float = 1e-3,
    ):
        """
        Args:
            channels: Input channels, injected by parse_model.
            reduction: Channel gate reduction ratio.
            spatial_kernel: Kernel size for spatial weak-target gate.
            detail_kernel: Larger depthwise kernel for local context.
            init_gamma: Initial residual enhancement strength.
        """
        super().__init__()
        assert channels > 0
        assert spatial_kernel in {3, 5, 7}
        assert detail_kernel in {3, 5, 7}

        hidden = max(1, channels // reduction)
        spatial_padding = spatial_kernel // 2
        detail_padding = detail_kernel // 2

        self.detail_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.detail_kxk = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=detail_kernel,
                padding=detail_padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.detail_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.spatial_gate = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.detail_fuse(self.detail_3x3(x) + self.detail_kxk(x))

        channel_attn = self.sigmoid(self.channel_gate(self.avg_pool(x)) + self.channel_gate(self.max_pool(x)))

        avg_map = detail.mean(dim=1, keepdim=True)
        max_map = detail.amax(dim=1, keepdim=True)
        spatial_attn = self.sigmoid(self.spatial_gate(torch.cat((avg_map, max_map), dim=1)))

        enhanced = detail * channel_attn * spatial_attn
        return x + self.gamma * enhanced

import torch
import torch.nn as nn


class _UFEBase(nn.Module):
    """
    Base implementation for Underwater Feature Enhancement variants.

    All variants keep the input/output shape unchanged and share the same local
    detail branch. Ablations toggle the channel max-pooling branch, spatial gate,
    or residual initialization.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        detail_kernel: int = 5,
        init_gamma: float = 1e-3,
        use_max_pool: bool = True,
        use_spatial_gate: bool = True,
    ):
        """
        Args:
            channels: Input channels, injected by parse_model.
            reduction: Channel gate reduction ratio.
            spatial_kernel: Kernel size for spatial weak-target gate.
            detail_kernel: Larger depthwise kernel for local context.
            init_gamma: Initial residual enhancement strength.
            use_max_pool: Whether to use GMP in the channel gate.
            use_spatial_gate: Whether to use the spatial weak-target gate.
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
        self.use_max_pool = use_max_pool
        self.use_spatial_gate = use_spatial_gate
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

        channel_logits = self.channel_gate(self.avg_pool(x))
        if self.use_max_pool:
            channel_logits = channel_logits + self.channel_gate(self.max_pool(x))
        channel_attn = self.sigmoid(channel_logits)

        enhanced = detail * channel_attn
        if self.use_spatial_gate:
            avg_map = detail.mean(dim=1, keepdim=True)
            max_map = detail.amax(dim=1, keepdim=True)
            spatial_attn = self.sigmoid(self.spatial_gate(torch.cat((avg_map, max_map), dim=1)))
            enhanced = enhanced * spatial_attn

        return x + self.gamma * enhanced


class UFE(_UFEBase):
    """
    Original A6-1 UFE: detail branch + GAP/GMP channel gate + spatial gate.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        detail_kernel: int = 5,
        init_gamma: float = 1e-3,
    ):
        super().__init__(
            channels,
            reduction=reduction,
            spatial_kernel=spatial_kernel,
            detail_kernel=detail_kernel,
            init_gamma=init_gamma,
            use_max_pool=True,
            use_spatial_gate=True,
        )


class UFE_GAP(_UFEBase):
    """
    A6-2-v1 UFE-GAP: remove the GMP branch from channel recalibration.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        detail_kernel: int = 5,
        init_gamma: float = 1e-3,
    ):
        super().__init__(
            channels,
            reduction=reduction,
            spatial_kernel=spatial_kernel,
            detail_kernel=detail_kernel,
            init_gamma=init_gamma,
            use_max_pool=False,
            use_spatial_gate=True,
        )


class UFE_DC(_UFEBase):
    """
    A6-2-v2 UFE-DC: keep detail + channel recalibration and remove the spatial gate.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        detail_kernel: int = 5,
        init_gamma: float = 1e-3,
    ):
        super().__init__(
            channels,
            reduction=reduction,
            spatial_kernel=spatial_kernel,
            detail_kernel=detail_kernel,
            init_gamma=init_gamma,
            use_max_pool=True,
            use_spatial_gate=False,
        )


class UFE_G0(_UFEBase):
    """
    A6-2-v3 UFE-G0: keep original UFE structure but initialize gamma to zero.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        detail_kernel: int = 5,
        init_gamma: float = 0.0,
    ):
        super().__init__(
            channels,
            reduction=reduction,
            spatial_kernel=spatial_kernel,
            detail_kernel=detail_kernel,
            init_gamma=init_gamma,
            use_max_pool=True,
            use_spatial_gate=True,
        )

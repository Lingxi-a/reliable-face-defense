from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DiscriminatorEngineeringConfig:
    # Paper states binary classifier with N ConvBlocks, trained from scratch.
    # Exact architecture remains paper-unreported/ambiguous.
    num_blocks: int = 5
    base_channels: int = 32
    max_channels: int = 256
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1


class DiscConvBlock(nn.Module):
    # Paper ConvBlock contract: CNN + BN + ReLU.
    def __init__(self, in_ch, out_ch, cfg):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            cfg.kernel_size, cfg.stride, cfg.padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class NullSwapDiscriminator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or DiscriminatorEngineeringConfig()
        blocks = []
        in_ch = 3
        ch = self.cfg.base_channels
        for _ in range(self.cfg.num_blocks):
            out_ch = min(ch, self.cfg.max_channels)
            blocks.append(DiscConvBlock(in_ch, out_ch, self.cfg))
            in_ch = out_ch
            ch *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

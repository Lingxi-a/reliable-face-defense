from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NullSwapPaperConfig:
    input_size: int = 256
    L: int = 4
    M: int = 3
    N: int = 5


@dataclass(frozen=True)
class NullSwapEngineeringConfig:
    # Paper-unreported implementation details.
    base_channels: int = 64
    bottleneck_ratio: int = 4
    se_reduction: int = 16

    id_conv_kernel: int = 3
    id_conv_stride: int = 2
    id_conv_padding: int = 1
    maxpool_kernel: int = 2
    maxpool_stride: int = 2

    feature_kernels: Tuple[int, int, int] = (3, 3, 3)
    feature_strides: Tuple[int, int, int] = (2, 2, 1)
    feature_paddings: Tuple[int, int, int] = (1, 1, 1)

    upsample_scale: int = 2
    upsample_mode: str = "bilinear"

    noise_alpha_init: float = 1.0
    noise_beta_init: float = 0.1
    noise_eta_init: float = 0.0
    cloak_gamma_init: float = 1.0


class ConvBlock(nn.Module):
    # Paper contract: CNN + BN + ReLU.
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s


class SEResBlock(nn.Module):
    # Paper contract: ResNet bottleneck + squeeze-and-excitation.
    def __init__(self, channels, bottleneck_ratio=4, se_reduction=16):
        super().__init__()
        hidden = max(channels // bottleneck_ratio, 1)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        y = self.se(y)
        return self.relu(identity + y)


class DeConvBlock(nn.Module):
    # Paper text: upsampling operation + BN + ReLU.
    def __init__(self, channels, scale_factor=2, mode="bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            x = F.interpolate(
                x, scale_factor=self.scale_factor,
                mode=self.mode, align_corners=False
            )
        else:
            x = F.interpolate(
                x, scale_factor=self.scale_factor,
                mode=self.mode
            )
        return self.relu(self.bn(x))


class IdentityExtraction(nn.Module):
    # ConvBlock -> MaxPool -> L x SEResBlock.
    def __init__(self, paper, eng):
        super().__init__()
        c = eng.base_channels
        self.conv = ConvBlock(
            3, c,
            eng.id_conv_kernel,
            eng.id_conv_stride,
            eng.id_conv_padding,
        )
        self.pool = nn.MaxPool2d(
            eng.maxpool_kernel,
            eng.maxpool_stride,
        )
        self.se_blocks = nn.ModuleList([
            SEResBlock(c, eng.bottleneck_ratio, eng.se_reduction)
            for _ in range(paper.L)
        ])

    def forward(self, x):
        x = self.pool(self.conv(x))
        for block in self.se_blocks:
            x = block(x)
        return x


class PerturbationBlock(nn.Module):
    # ConvBlock -> M x SEResBlock -> adaptive random-noise addition.
    def __init__(self, paper, eng, feature_hw):
        super().__init__()
        c = eng.base_channels
        self.conv = ConvBlock(c, c)
        self.se_blocks = nn.ModuleList([
            SEResBlock(c, eng.bottleneck_ratio, eng.se_reduction)
            for _ in range(paper.M)
        ])
        self.noise_alpha = nn.Parameter(torch.tensor(float(eng.noise_alpha_init)))
        self.noise_beta = nn.Parameter(torch.tensor(float(eng.noise_beta_init)))
        # Paper states eta is learnable but does not report its shape/init.
        self.noise_eta = nn.Parameter(
            torch.full(
                (1, c, feature_hw, feature_hw),
                float(eng.noise_eta_init),
            )
        )

    def forward(self, x):
        x = self.conv(x)
        for block in self.se_blocks:
            x = block(x)
        base_noise = torch.randn_like(x)
        adaptive_noise = self.noise_beta * (
            self.noise_alpha * base_noise + self.noise_eta
        )
        return x + adaptive_noise, adaptive_noise


class FeatureBlock(nn.Module):
    # 3 x ConvBlock -> N x SEResBlock.
    def __init__(self, paper, eng):
        super().__init__()
        c = eng.base_channels
        ks = eng.feature_kernels
        st = eng.feature_strides
        pd = eng.feature_paddings
        self.conv1 = ConvBlock(3, c, ks[0], st[0], pd[0])
        self.conv2 = ConvBlock(c, c, ks[1], st[1], pd[1])
        self.conv3 = ConvBlock(c, c, ks[2], st[2], pd[2])
        self.se_blocks = nn.ModuleList([
            SEResBlock(c, eng.bottleneck_ratio, eng.se_reduction)
            for _ in range(paper.N)
        ])

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        for block in self.se_blocks:
            x = block(x)
        return x


class CloakingBlock(nn.Module):
    """
    Paper macro-sequence:
    gamma*perturbation -> concat shallow feature -> SEResBlock
    -> DeConvBlock -> ConvBlock -> DeConvBlock
    -> concat original -> ConvBlock -> ConvBlock -> ConvBlock.
    """
    def __init__(self, eng):
        super().__init__()
        c = eng.base_channels
        fused = 2 * c

        self.cloak_gamma = nn.Parameter(torch.tensor(float(eng.cloak_gamma_init)))
        self.se = SEResBlock(fused, eng.bottleneck_ratio, eng.se_reduction)
        self.deconv1 = DeConvBlock(fused, eng.upsample_scale, eng.upsample_mode)
        self.mid_conv = ConvBlock(fused, fused)
        self.deconv2 = DeConvBlock(fused, eng.upsample_scale, eng.upsample_mode)

        self.final_conv1 = ConvBlock(fused + 3, c)
        self.final_conv2 = ConvBlock(c, c)
        self.final_conv3 = ConvBlock(c, 3)

    def forward(self, perturbation, shallow_feature, original):
        if perturbation.shape != shallow_feature.shape:
            raise ValueError("Perturbation and shallow feature shapes must match.")
        x = torch.cat(
            [self.cloak_gamma * perturbation, shallow_feature],
            dim=1,
        )
        x = self.se(x)
        x = self.deconv1(x)
        x = self.mid_conv(x)
        x = self.deconv2(x)
        if x.shape[-2:] != original.shape[-2:]:
            raise ValueError("Feature reconstruction did not recover 256x256.")
        x = torch.cat([x, original], dim=1)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return self.final_conv3(x)


class NullSwapGenerator(nn.Module):
    def __init__(self, paper=None, engineering=None):
        super().__init__()
        self.paper = paper or NullSwapPaperConfig()
        self.engineering = engineering or NullSwapEngineeringConfig()

        feature_hw = self.paper.input_size // (
            self.engineering.id_conv_stride
            * self.engineering.maxpool_stride
        )
        self.id_extraction = IdentityExtraction(self.paper, self.engineering)
        self.perturbation = PerturbationBlock(
            self.paper, self.engineering, feature_hw
        )
        self.feature = FeatureBlock(self.paper, self.engineering)
        self.cloaking = CloakingBlock(self.engineering)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[-2:] != (256, 256):
            raise ValueError(f"Expected BCHW RGB 256x256, got {tuple(x.shape)}")
        identity_feature = self.id_extraction(x)
        perturbation, adaptive_noise = self.perturbation(identity_feature)
        shallow_feature = self.feature(x)
        protected_raw = self.cloaking(
            perturbation, shallow_feature, x
        )
        return {
            "protected_raw": protected_raw,
            "identity_feature": identity_feature,
            "perturbation": perturbation,
            "adaptive_noise": adaptive_noise,
            "shallow_feature": shallow_feature,
        }

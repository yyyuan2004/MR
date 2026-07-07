"""Optional compact U-Net for learned artifact removal.

Not used by the default pipeline. Provided as a starting point for training a
post-processor on zero-filled or ridge reconstructions.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Two-level U-Net with skip connections for small (e.g. 64x64) images."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = _conv_block(in_channels, c)
        self.enc2 = _conv_block(c, 2 * c)
        self.bottleneck = _conv_block(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=2, stride=2)
        self.dec2 = _conv_block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, kernel_size=2, stride=2)
        self.dec1 = _conv_block(2 * c, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        # Residual output: the network predicts a correction to its input.
        return x + self.head(d1)

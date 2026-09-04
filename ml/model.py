"""
Siamese U-Net Architecture (FC-Siam-diff) for Bi-Temporal Change Detection in PyTorch.

Architecture Summary:
- Shared-weight twin encoder extracts hierarchical multi-scale feature maps from T-0 and T-1.
- Absolute feature difference skip connections: D^(l) = |F_T0^(l) - F_T1^(l)|.
- Decoder with transposed convolutions upsamples difference features back to original pixel grid.
- Output: 1-channel change logits map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class SiameseUNetDiff(nn.Module):
    """
    Fully Convolutional Siamese Difference Network (FC-Siam-diff).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        b = base_channels

        # --- Twin Shared-Weight Encoder ---
        self.inc = DoubleConv(in_channels, b)         # 32
        self.down1 = DoubleConv(b, b * 2)             # 64
        self.down2 = DoubleConv(b * 2, b * 4)         # 128
        self.down3 = DoubleConv(b * 4, b * 8)         # 256
        self.down4 = DoubleConv(b * 8, b * 16)        # 512 (bottleneck)

        self.pool = nn.MaxPool2d(2)

        # --- Decoder (Upsampling difference features) ---
        self.up1 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 8 + b * 8, b * 8)

        self.up2 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4 + b * 4, b * 4)

        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 2 + b * 2, b * 2)

        self.up4 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b + b, b)

        # Final 1x1 classification projection
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def encode(self, x: torch.Tensor):
        """Pass single image through the encoder hierarchy."""
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))
        return x1, x2, x3, x4, x5

    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with twin shared encoders and difference skip connections.

        Args:
            img_t0: Tensor (B, 3, H, W) - T-0 Reference capture
            img_t1: Tensor (B, 3, H, W) - T-1 Target capture

        Returns:
            logits: Tensor (B, 1, H, W)
        """
        # 1. Twin feature extraction with shared weights
        t0_1, t0_2, t0_3, t0_4, t0_5 = self.encode(img_t0)
        t1_1, t1_2, t1_3, t1_4, t1_5 = self.encode(img_t1)

        # 2. Compute absolute feature differences at each hierarchy level
        diff_5 = torch.abs(t0_5 - t1_5)
        diff_4 = torch.abs(t0_4 - t1_4)
        diff_3 = torch.abs(t0_3 - t1_3)
        diff_2 = torch.abs(t0_2 - t1_2)
        diff_1 = torch.abs(t0_1 - t1_1)

        # 3. Decoder with skip connections from difference maps
        x = self.up1(diff_5)
        x = torch.cat([x, diff_4], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, diff_3], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, diff_2], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x, diff_1], dim=1)
        x = self.dec4(x)

        logits = self.outc(x)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_layers import SpectralConv2d


class SmoothBranch(nn.Module):
    """FNO主干：捕捉全局低频解"""

    def __init__(self, modes1=12, modes2=12, width=64, in_channels=1, out_channels=1):
        super().__init__()
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for coordinates

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, grid=None):
        B, _, H, W = x.shape

        if grid is None:
            grid = self._make_grid(B, H, W, x.device)

        # [B, C+2, H, W] -> [B, H, W, C+2] -> lift -> [B, width, H, W]
        x = torch.cat([x.permute(0, 2, 3, 1), grid.permute(0, 2, 3, 1)], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)

        x1 = F.gelu(self.conv0(x) + self.w0(x))
        x2 = F.gelu(self.conv1(x1) + self.w1(x1))
        x3 = F.gelu(self.conv2(x2) + self.w2(x2))
        x4 = self.conv3(x3) + self.w3(x3)

        x4 = x4.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x4))
        x = self.fc2(x)

        return x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]

    def _make_grid(self, B, H, W, device):
        gx = torch.linspace(-1, 1, W, device=device)
        gy = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)
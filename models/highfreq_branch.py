import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_layers import SpectralConv2d


class HighFreqBranch(nn.Module):

    def __init__(self, modes1=16, modes2=16, width=32, in_channels=1, out_channels=1):
        super().__init__()

        # 高频谱层：保留比平滑分支更多的模态
        self.spectral1 = SpectralConv2d(in_channels, width, modes1, modes2)
        self.spectral2 = SpectralConv2d(width, width, modes1, modes2)

        # 局部卷积层：捕捉局部高频振荡
        self.local_conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.local_conv2 = nn.Conv2d(width, width, 3, padding=1)

        # 融合与投影
        self.fusion = nn.Conv2d(width, width, 1)
        self.proj = nn.Conv2d(width, out_channels, 1)

        # 提升层
        self.fc0 = nn.Linear(in_channels + 2, width)

    def forward(self, x, grid=None):

        batch, _, h, w = x.shape

        if grid is None:
            grid = self._make_grid(batch, h, w, x.device)


        x_perm = torch.cat([x.permute(0, 2, 3, 1), grid.permute(0, 2, 3, 1)], dim=-1)
        x_perm = self.fc0(x_perm).permute(0, 3, 1, 2)


        spec = self.spectral1(x_perm)
        spec = F.gelu(spec)
        spec = self.spectral2(spec)


        local = self.local_conv1(x)
        local = F.gelu(local)
        local = self.local_conv2(local)


        if local.shape[1] != spec.shape[1]:
            local = nn.Conv2d(local.shape[1], spec.shape[1], 1).to(local.device)(local)


        fused = self.fusion(spec + local)
        fused = F.gelu(fused)

        return self.proj(fused)

    def _make_grid(self, batch, h, w, device):
        x = torch.linspace(-1, 1, w, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        grid_x, grid_y = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)
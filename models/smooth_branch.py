import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_layers import SpectralConv2d


class SmoothBranch(nn.Module):

    def __init__(self, modes1=12, modes2=12, width=64, in_channels=1, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # 提升层：输入 → 高维特征空间
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 坐标

        # 4层傅里叶层（论文推荐4层效果最好[citation:2]）
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        # 局部线性层（1x1卷积，残差连接用）
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # 投影层：高维特征 → 输出
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, grid=None):
        """
        x: [batch, channels, height, width] 源项
        grid: [batch, 2, height, width] 坐标网格（如未提供则自动生成）
        """
        batch, _, h, w = x.shape

        # 生成坐标网格[citation:2]
        if grid is None:
            grid = self._make_grid(batch, h, w, x.device)

        # 拼接源项和坐标，调整维度 [b, c, h, w] → [b, h, w, c]
        x = torch.cat([x.permute(0, 2, 3, 1), grid.permute(0, 2, 3, 1)], dim=-1)

        # 提升层
        x = self.fc0(x)  # [b, h, w, width]
        x = x.permute(0, 3, 1, 2)  # [b, width, h, w]

        # 4层傅里叶-局部交替处理
        x1 = self.conv0(x) + self.w0(x)
        x1 = F.gelu(x1)

        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)

        x3 = self.conv2(x2) + self.w2(x2)
        x3 = F.gelu(x3)

        x4 = self.conv3(x3) + self.w3(x3)

        # 投影层
        x4 = x4.permute(0, 2, 3, 1)  # [b, h, w, width]
        x = F.gelu(self.fc1(x4))
        x = self.fc2(x)

        return x.permute(0, 3, 1, 2)  # [b, out_channels, h, w]

    def _make_grid(self, batch, h, w, device):
        """生成归一化坐标网格 [-1, 1]"""
        x = torch.linspace(-1, 1, w, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        grid_x, grid_y = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, h, w]
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)
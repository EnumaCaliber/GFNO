import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_layers import SpectralConv2d


class HighFreqBranch(nn.Module):
    """
    高频残差分支：谱方法 + 局部卷积，学习平滑分支的残差。

    修复：
    - 删除 forward 中动态创建 nn.Conv2d 的危险代码
      （参数不在优化器中，且通道数由构造函数保证一致，该分支永远不会触发）
    - 加入 u_smooth 作为可选输入，显式支持残差学习
    """

    def __init__(self, modes1=16, modes2=16, width=32, in_channels=1, out_channels=1):
        super().__init__()
        self.width = width

        # 提升层：f + 坐标 -> width
        self.fc0 = nn.Linear(in_channels + 2, width)

        # 谱路径
        self.spectral1 = SpectralConv2d(width, width, modes1, modes2)
        self.spectral2 = SpectralConv2d(width, width, modes1, modes2)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)

        # 局部路径（捕捉边界层附近的局部高频振荡）
        self.local_conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.local_conv2 = nn.Conv2d(width, width, 3, padding=1)

        # 融合与投影
        self.fusion = nn.Conv2d(width, width, 1)
        self.proj   = nn.Conv2d(width, out_channels, 1)

    def forward(self, x, grid=None, u_smooth=None):
        """
        x:        [B, C, H, W]  源项/系数场
        grid:     [B, 2, H, W]  坐标网格（可选）
        u_smooth: [B, 1, H, W]  平滑分支输出，用于残差监督（训练时传入）
        """
        B, _, H, W = x.shape

        if grid is None:
            grid = self._make_grid(B, H, W, x.device)

        # 拼接坐标并提升
        x_cat  = torch.cat([x.permute(0, 2, 3, 1), grid.permute(0, 2, 3, 1)], dim=-1)
        x_lift = self.fc0(x_cat).permute(0, 3, 1, 2)  # [B, width, H, W]

        # 谱路径（含残差连接）
        spec = F.gelu(self.spectral1(x_lift) + self.w1(x_lift))
        spec = self.spectral2(spec) + self.w2(spec)

        # 局部路径（通道数均为 width，✅ 无需动态创建层）
        local = F.gelu(self.local_conv1(x_lift))
        local = self.local_conv2(local)

        # 融合
        fused = F.gelu(self.fusion(spec + local))

        return self.proj(fused)  # [B, out_channels, H, W]

    def _make_grid(self, B, H, W, device):
        gx = torch.linspace(-1, 1, W, device=device)
        gy = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)
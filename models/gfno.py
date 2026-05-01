import torch
import torch.nn as nn
from .smooth_branch import SmoothBranch
from .boundary_branch import BoundaryGreenBranch
from .highfreq_branch import HighFreqBranch
from .fusion import AdaptiveFusion


class GFNO(nn.Module):


    def __init__(self,
                 smooth_modes=(12, 12),
                 smooth_width=64,
                 high_modes=(16, 16),
                 high_width=32,
                 n_boundary_points=200,
                 fusion_type='attention'):
        super().__init__()

        # 三分支
        self.smooth_branch = SmoothBranch(
            modes1=smooth_modes[0], modes2=smooth_modes[1],
            width=smooth_width, in_channels=1, out_channels=1
        )

        self.boundary_branch = BoundaryGreenBranch(
            n_boundary_points=n_boundary_points,
            hidden_dim=64, out_channels=1
        )

        self.high_branch = HighFreqBranch(
            modes1=high_modes[0], modes2=high_modes[1],
            width=high_width, in_channels=1, out_channels=1
        )

        # 自适应融合
        self.fusion = AdaptiveFusion(1, fusion_type)

    def forward(self, f, boundary_info=None, grid=None, interior_coords=None):
        """
        f: [batch, 1, h, w] 源项/系数场
        boundary_info: [batch, n_points, 3] (x, y, g_value) 边界条件
        grid: [batch, 2, h, w] 坐标网格
        interior_coords: [batch, n_interior, 2] 内部点坐标
        """
        batch, _, h, w = f.shape

        # 生成坐标网格
        if grid is None:
            grid = self._make_grid(batch, h, w, f.device)

        # 生成内部点坐标（用于边界分支）
        if interior_coords is None:
            interior_coords = self._make_interior_coords(h, w, f.device)
            interior_coords = interior_coords.unsqueeze(0).repeat(batch, 1, 1)

        # 三分支并行计算
        u_smooth = self.smooth_branch(f, grid)

        if boundary_info is not None:
            u_boundary = self.boundary_branch(boundary_info, interior_coords, (h, w))
        else:
            u_boundary = torch.zeros_like(u_smooth)

        u_high = self.high_branch(f, grid)

        # 自适应融合
        u = self.fusion(u_smooth, u_boundary, u_high)

        return u, {'smooth': u_smooth, 'boundary': u_boundary, 'high': u_high}

    def _make_grid(self, batch, h, w, device):
        x = torch.linspace(-1, 1, w, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        grid_x, grid_y = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)

    def _make_interior_coords(self, h, w, device):
        """生成内部点坐标网格"""
        x = torch.linspace(-1, 1, w, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        grid_x, grid_y = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return coords  # [h*w, 2]
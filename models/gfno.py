import torch
import torch.nn as nn
import torch.nn.functional as F
from .smooth_branch import SmoothBranch
from .boundary_branch import BoundaryGreenBranch
from .highfreq_branch import HighFreqBranch
from .fusion import AdaptiveFusion


class GFNO(nn.Module):
    """
    Green-augmented Fourier Neural Operator

    三分支分解：
        u(x) ≈ u_smooth(x)    # FNO 主干，低频全局解
              + u_boundary(x)  # 格林函数分支，边界条件修正
              + u_high(x)      # 高频残差分支

    支持非规则域（通过 domain_mask 将域外置零）
    """

    def __init__(self,
                 smooth_modes=(12, 12),
                 smooth_width=64,
                 high_modes=(16, 16),
                 high_width=32,
                 n_boundary_points=64,
                 fusion_type='attention'):
        super().__init__()

        self.smooth_branch = SmoothBranch(
            modes1=smooth_modes[0], modes2=smooth_modes[1],
            width=smooth_width, in_channels=1, out_channels=1
        )
        self.boundary_branch = BoundaryGreenBranch(
            n_boundary_points=n_boundary_points,
            hidden_dim=64, out_channels=1, coarse_factor=4
        )
        self.high_branch = HighFreqBranch(
            modes1=high_modes[0], modes2=high_modes[1],
            width=high_width, in_channels=1, out_channels=1
        )
        self.fusion = AdaptiveFusion(in_channels=1, fusion_type=fusion_type)

    def forward(self, f, boundary_info=None, grid=None, domain_mask=None):
        """
        f:             [B, 1, H, W]      源项/系数场
        boundary_info: [B, n_bc, 3]      边界条件 (x_norm, y_norm, g)，坐标 [-1,1]
        grid:          [B, 2, H, W]      坐标网格（可选，自动生成）
        domain_mask:   [H, W] bool       True = 域内点（可选，用于非规则域）

        returns:
            u:      [B, 1, H, W]         最终预测
            branches: dict               各分支输出（用于损失计算）
        """
        B, _, H, W = f.shape

        if grid is None:
            grid = self._make_grid(B, H, W, f.device)

        # ── 三分支并行 ──────────────────────────────────────────
        u_smooth = self.smooth_branch(f, grid)

        if boundary_info is not None:
            interior_coords = self._make_interior_coords(H, W, f.device)
            interior_coords = interior_coords.unsqueeze(0).expand(B, -1, -1)
            u_boundary = self.boundary_branch(boundary_info, interior_coords, (H, W))
        else:
            u_boundary = torch.zeros_like(u_smooth)

        u_high = self.high_branch(f, grid, u_smooth=u_smooth)

        # ── 自适应融合 ─────────────────────────────────────────
        u = self.fusion(u_smooth, u_boundary, u_high)

        # ── 非规则域：域外置零 ─────────────────────────────────
        if domain_mask is not None:
            mask = domain_mask.to(f.device).unsqueeze(0).unsqueeze(0).float()
            u = u * mask

        return u, {'smooth': u_smooth, 'boundary': u_boundary, 'high': u_high}

    def _make_grid(self, B, H, W, device):
        gx = torch.linspace(-1, 1, W, device=device)
        gy = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)

    def _make_interior_coords(self, H, W, device):
        gx = torch.linspace(-1, 1, W, device=device)
        gy = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [H*W, 2]
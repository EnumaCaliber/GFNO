import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryGreenBranch(nn.Module):
    """
    边界格林分支：用离散格林函数建模边界条件对内部的影响。

    修复：
    1. interior_coords 维度解包错误 (n_int, _) -> (_, n_int, _)
    2. 内存爆炸：改为在粗网格上计算，再双线性上采样到目标分辨率
       [B, n_bc, H*W, D] (7GB) -> [B, n_bc, (H/4)*(W/4), D] (~140MB)
    """

    def __init__(self, n_boundary_points=64, hidden_dim=64, out_channels=1, coarse_factor=4):
        super().__init__()
        self.n_boundary_points = n_boundary_points
        self.coarse_factor = coarse_factor

        # 边界点编码器: [x, y, g] -> D
        self.boundary_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 格林核网络: (边界特征 D, 内部坐标 2, 距离 1) -> out_channels
        self.green_kernel = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_channels)
        )

        # 可学习的距离衰减尺度（物理归纳偏置：边界影响随距离衰减）
        self.distance_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, boundary_info, interior_coords, interior_shape):
        """
        boundary_info:   [B, n_bc, 3]      (x_norm, y_norm, g_value)，坐标在 [-1,1]
        interior_coords: [B, n_int, 2]     内部点坐标（此参数保留接口，实际用粗网格）
        interior_shape:  (H, W)            目标输出分辨率
        """
        batch, n_bc, _ = boundary_info.shape
        # ✅ 修复 Bug 1：原代码 n_int, _ = interior_coords.shape 会在三维张量上报错
        _, n_int, _ = interior_coords.shape

        H, W = interior_shape

        # === 在粗网格上计算，减少内存占用 ===
        H_c = max(H // self.coarse_factor, 4)
        W_c = max(W // self.coarse_factor, 4)
        n_coarse = H_c * W_c

        # 粗网格坐标 [n_coarse, 2]
        coarse_coords = self._make_coarse_coords(H_c, W_c, boundary_info.device)
        coarse_coords = coarse_coords.unsqueeze(0).expand(batch, -1, -1)  # [B, n_coarse, 2]

        # 1. 编码边界点
        boundary_feats = self.boundary_encoder(boundary_info)  # [B, n_bc, D]

        # 2. 计算边界点 -> 粗网格点 的欧氏距离
        bc_coords = boundary_info[..., :2]                   # [B, n_bc, 2]
        bc_exp    = bc_coords.unsqueeze(2)                   # [B, n_bc, 1, 2]
        coarse_exp = coarse_coords.unsqueeze(1)              # [B, 1, n_coarse, 2]

        diff     = bc_exp - coarse_exp                       # [B, n_bc, n_coarse, 2]
        distance = torch.sqrt((diff ** 2).sum(-1) + 1e-8)   # [B, n_bc, n_coarse]

        # 3. 距离先验衰减（格林函数的物理归纳偏置）
        dist_weight = torch.exp(-self.distance_scale.abs() * distance)  # [B, n_bc, n_coarse]

        # 4. 拼接格林核输入
        bf_exp   = boundary_feats.unsqueeze(2).expand(-1, -1, n_coarse, -1)  # [B, n_bc, n_coarse, D]
        ic_exp   = coarse_exp.expand(-1, n_bc, -1, -1)                       # [B, n_bc, n_coarse, 2]
        dist_exp = distance.unsqueeze(-1)                                     # [B, n_bc, n_coarse, 1]

        kernel_input = torch.cat([bf_exp, ic_exp, dist_exp], dim=-1)         # [B, n_bc, n_coarse, D+3]

        # 5. 格林核前向 + 距离加权
        raw_contrib = self.green_kernel(kernel_input)                         # [B, n_bc, n_coarse, C]
        contribution = raw_contrib * dist_weight.unsqueeze(-1)               # [B, n_bc, n_coarse, C]

        # 6. 离散边界积分（均匀权重求和）
        u_coarse = contribution.sum(dim=1) / n_bc                            # [B, n_coarse, C]

        # 7. 重塑并上采样到目标分辨率
        C = u_coarse.shape[-1]
        u_coarse = u_coarse.permute(0, 2, 1).reshape(batch, C, H_c, W_c)    # [B, C, H_c, W_c]
        u_boundary = F.interpolate(u_coarse, size=(H, W), mode='bilinear', align_corners=True)

        return u_boundary  # [B, C, H, W]

    def _make_coarse_coords(self, H, W, device):
        """生成粗网格归一化坐标 [-1, 1]"""
        gx = torch.linspace(-1, 1, W, device=device)
        gy = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [H*W, 2]
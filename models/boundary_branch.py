import torch
import torch.nn as nn



class BoundaryGreenBranch(nn.Module):


    def __init__(self, n_boundary_points=200, hidden_dim=64, out_channels=1):
        super().__init__()
        self.n_boundary_points = n_boundary_points


        self.boundary_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # [x, y, g_value]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )


        self.green_kernel = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, boundary_info, interior_coords, interior_shape):

        batch, n_bc, _ = boundary_info.shape
        n_int, _ = interior_coords.shape

        # 1. 编码边界点
        boundary_feats = self.boundary_encoder(boundary_info)  # [b, n_bc, D]

        # 2. 扩展以便计算所有边界点-内部点对
        # boundary_feats: [b, n_bc, D] → [b, n_bc, 1, D]
        # interior_coords: [b, 1, n_int, 2]
        boundary_feats_exp = boundary_feats.unsqueeze(2)  # [b, n_bc, 1, D]
        interior_coords_exp = interior_coords.unsqueeze(1)  # [b, 1, n_int, 2]
        interior_coords_exp = interior_coords_exp.expand(-1, n_bc, -1, -1)

        # 3. 拼接：边界特征 + 内部点坐标
        kernel_input = torch.cat([
            boundary_feats_exp.expand(-1, -1, n_int, -1),
            interior_coords_exp
        ], dim=-1)  # [b, n_bc, n_int, D+2]

        # 4. 计算每个边界点对每个内部点的贡献
        contribution = self.green_kernel(kernel_input)  # [b, n_bc, n_int, 1]

        # 5. 边界积分（离散求和），简单平均权重
        u_boundary = contribution.sum(dim=1) / n_bc  # [b, n_int, 1]

        # 6. 重塑为网格形状 [b, 1, h, w]
        h, w = interior_shape
        u_boundary = u_boundary.reshape(batch, 1, h, w)

        return u_boundary

    def sample_boundary_points(self, boundary_function, device, n_points=None):
        """从给定的边界函数采样边界点"""
        n_points = n_points or self.n_boundary_points
        # 简化为单位正方形边界采样
        t = torch.linspace(0, 1, n_points, device=device)
        # 边界参数化：四条边
        points = []
        values = []
        for ti in t:
            # 底边 y=0
            points.append([2 * ti - 1, -1])
            values.append(boundary_function(2 * ti - 1, -1))
            # 顶边 y=1
            points.append([2 * ti - 1, 1])
            values.append(boundary_function(2 * ti - 1, 1))
            # 左边 x=-1
            points.append([-1, 2 * ti - 1])
            values.append(boundary_function(-1, 2 * ti - 1))
            # 右边 x=1
            points.append([1, 2 * ti - 1])
            values.append(boundary_function(1, 2 * ti - 1))

        points = torch.tensor(points, device=device)
        values = torch.tensor(values, device=device).unsqueeze(-1)
        return torch.cat([points, values], dim=-1).unsqueeze(0)  # [1, 4*n_points, 3]
"""
非规则域 PDE 数据生成
支持：L形域、带圆孔域
求解：-Δu = f，Dirichlet 边界条件 u|_∂Ω = g(x,y)
"""

import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import os


# ── 域定义 ────────────────────────────────────────────────────────────────────

def make_l_shape_mask(N):
    """L形域：单位正方形去掉右上角 1/4"""
    mask = np.ones((N, N), dtype=bool)
    mask[N // 2:, N // 2:] = False
    return mask


def make_circle_hole_mask(N):
    """带圆孔的正方形域：去掉中心圆（半径 ≈ N/6）"""
    mask = np.ones((N, N), dtype=bool)
    cx, cy, r = N // 2, N // 2, N // 6
    ii, jj = np.ogrid[:N, :N]
    mask[(ii - cx) ** 2 + (jj - cy) ** 2 <= r ** 2] = False
    return mask


# ── FD 求解器 ─────────────────────────────────────────────────────────────────

def solve_poisson_fd(f_field, mask, h, bc_func):
    """
    有限差分法在非规则域上求解 -Δu = f
    对域外点（mask=False）用 Dirichlet BC u = bc_func(x, y)
    """
    N = mask.shape[0]

    # 内部点线性索引映射
    idx_map = -np.ones((N, N), dtype=np.int32)
    interior = np.argwhere(mask)          # [n_int, 2]
    n_int = len(interior)
    idx_map[interior[:, 0], interior[:, 1]] = np.arange(n_int)

    h2 = h * h
    A   = lil_matrix((n_int, n_int), dtype=np.float64)
    rhs = np.zeros(n_int, dtype=np.float64)

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for k, (i, j) in enumerate(interior):
        A[k, k] = -4.0 / h2
        rhs[k]  = f_field[i, j]

        for di, dj in DIRS:
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N and mask[ni, nj]:
                A[k, idx_map[ni, nj]] = 1.0 / h2
            else:
                # 域外 → 边界条件贡献
                xb = np.clip(nj * h, 0.0, 1.0)
                yb = np.clip(ni * h, 0.0, 1.0)
                rhs[k] -= bc_func(xb, yb) / h2

    u_int = spsolve(A.tocsr(), rhs)
    u = np.zeros((N, N), dtype=np.float64)
    u[mask] = u_int
    return u


# ── 边界点提取 ────────────────────────────────────────────────────────────────

def extract_boundary_points(mask, h, bc_func, n_points=64):
    """
    提取域边界点（内部点中至少有一个邻居在域外的点）
    坐标归一化到 [-1, 1]，返回 [n_points, 3]
    """
    N = mask.shape[0]
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    pts  = []

    for i in range(N):
        for j in range(N):
            if not mask[i, j]:
                continue
            on_boundary = any(
                not (0 <= i + di < N and 0 <= j + dj < N and mask[i + di, j + dj])
                for di, dj in DIRS
            )
            if on_boundary:
                x, y = j * h, i * h
                g = bc_func(x, y)
                # 归一化到 [-1, 1]（与模型坐标系一致）
                pts.append([2 * x - 1, 2 * y - 1, g])

    pts = np.array(pts, dtype=np.float32)

    if len(pts) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    # 均匀下采样或重复填充到 n_points
    if len(pts) >= n_points:
        idx = np.round(np.linspace(0, len(pts) - 1, n_points)).astype(int)
        pts = pts[idx]
    else:
        reps = (n_points // len(pts)) + 1
        pts  = np.tile(pts, (reps, 1))[:n_points]

    return pts  # [n_points, 3]


# ── 随机场生成 ────────────────────────────────────────────────────────────────

def random_smooth_field(N, n_modes=6):
    """傅里叶基展开生成随机平滑场"""
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x, indexing='ij')
    field = np.zeros((N, N))
    for k in range(1, n_modes + 1):
        for l in range(1, n_modes + 1):
            amp = np.random.randn() / (k ** 2 + l ** 2)
            field += amp * np.sin(k * np.pi * X) * np.sin(l * np.pi * Y)
    return field


# ── 主生成函数 ────────────────────────────────────────────────────────────────

def generate_dataset(domain_type='l_shape', N=64, n_samples=500,
                     n_boundary_points=64, save_dir='./data'):
    print(f"\n{'='*50}")
    print(f"生成 {domain_type} 域数据集")
    print(f"分辨率: {N}×{N}，目标样本数: {n_samples}")
    print(f"{'='*50}")

    if domain_type == 'l_shape':
        mask = make_l_shape_mask(N)
    elif domain_type == 'circle_hole':
        mask = make_circle_hole_mask(N)
    else:
        mask = np.ones((N, N), dtype=bool)

    h = 1.0 / (N - 1)

    coeffs, solutions, boundaries = [], [], []

    for _ in tqdm(range(n_samples), desc=domain_type):
        # 随机系数场（模拟渗透率 a(x)，保证正定）
        a_field = np.exp(random_smooth_field(N, n_modes=5))
        a_field = np.clip(a_field, 0.1, 10.0)

        # 随机源项
        f_field = random_smooth_field(N, n_modes=4)
        f_field[~mask] = 0.0

        # 随机非零 Dirichlet BC（这是边界分支的学习目标）
        freq  = np.random.randint(1, 4)
        amp   = np.random.uniform(0.1, 1.0)
        phase = np.random.uniform(0, np.pi)
        def bc_func(x, y, _f=freq, _a=amp, _p=phase):
            return _a * np.sin(_f * np.pi * x + _p) * np.cos(_f * np.pi * y)

        try:
            u = solve_poisson_fd(f_field, mask, h, bc_func)
        except Exception:
            continue

        if np.any(np.isnan(u)) or np.any(np.isinf(u)) or np.abs(u).max() > 1e4:
            continue

        u[~mask] = 0.0
        a_field[~mask] = 0.0

        bc_pts = extract_boundary_points(mask, h, bc_func, n_boundary_points)

        coeffs.append(a_field.astype(np.float32))
        solutions.append(u.astype(np.float32))
        boundaries.append(bc_pts)

    n_success = len(coeffs)
    print(f"成功生成: {n_success}/{n_samples} 样本")

    coeffs    = torch.tensor(np.array(coeffs))
    solutions = torch.tensor(np.array(solutions))
    boundaries = torch.tensor(np.array(boundaries))
    mask_t    = torch.tensor(mask)

    n_train = int(0.8 * n_success)
    os.makedirs(save_dir, exist_ok=True)

    torch.save({'x': coeffs[:n_train], 'y': solutions[:n_train],
                'boundary': boundaries[:n_train], 'mask': mask_t},
               f'{save_dir}/{domain_type}_train.pt')
    torch.save({'x': coeffs[n_train:], 'y': solutions[n_train:],
                'boundary': boundaries[n_train:], 'mask': mask_t},
               f'{save_dir}/{domain_type}_test.pt')

    print(f"✅ 保存至 {save_dir}/{domain_type}_{{train,test}}.pt")
    print(f"   训练: {n_train} | 测试: {n_success - n_train}")
    return n_success


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='l_shape',
                   choices=['l_shape', 'circle_hole', 'square'])
    p.add_argument('--N',        type=int, default=64)
    p.add_argument('--n',        type=int, default=500)
    args = p.parse_args()
    generate_dataset(args.domain, N=args.N, n_samples=args.n)
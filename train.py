import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from models.gfno import GFNO


# ========== 配置参数 ==========
class Config:
    # 数据
    batch_size = 32
    train_ratio = 0.8

    # 模型参数[citation:2]
    smooth_modes = (12, 12)  # 保留的低频模态数（网格尺寸的1/8~1/4）
    smooth_width = 64  # 隐藏通道数
    high_modes = (16, 16)  # 高频分支保留更多模态
    high_width = 32

    # 训练参数
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 损失权重
    lambda_pde = 0.1  # PDE残差权重
    lambda_bc = 0.5  # 边界条件权重


def relative_l2_loss(pred, target):
    """相对L2损失：||pred - target||₂ / ||target||₂"""
    return torch.norm(pred - target) / torch.norm(target)


class CompositeLoss(nn.Module):
    """复合损失函数：MSE + PDE残差 + 边界约束"""

    def __init__(self, lambda_pde=0.1, lambda_bc=0.5):
        super().__init__()
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.mse = nn.MSELoss()

    def forward(self, pred, target, u_smooth=None, u_high=None, boundary_info=None):
        # 1. 数据拟合损失
        loss_data = self.mse(pred, target)

        # 2. 边界条件损失
        loss_bc = 0
        if boundary_info is not None:
            # 在边界采样点上计算误差
            loss_bc = self._compute_boundary_loss(pred, boundary_info)

        # 3. 正则化：控制各分支的平衡
        loss_reg = 0
        if u_smooth is not None and u_high is not None:
            # 鼓励高频分支学习残差（而非主信号）
            residual = target - u_smooth
            loss_reg = self.mse(u_high, residual.detach())

        return loss_data + self.lambda_bc * loss_bc + 0.05 * loss_reg

    def _compute_boundary_loss(self, pred, boundary_info):
        """在边界点上计算预测值与真实边界条件的差异"""
        # boundary_info: [n_points, 3] (x, y, g_true)
        # 需要在边界点上采样pred的值
        # 简化实现，实际需要双线性插值
        return 0  # placeholder


def train():
    # 1. 初始化模型
    model = GFNO(
        smooth_modes=Config.smooth_modes,
        smooth_width=Config.smooth_width,
        high_modes=Config.high_modes,
        high_width=Config.high_width,
        fusion_type='attention'
    ).to(Config.device)

    # 2. 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    criterion = CompositeLoss(lambda_pde=Config.lambda_pde, lambda_bc=Config.lambda_bc)

    # 3. 数据加载（需要实现自己的Dataset）
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # 4. 训练循环
    best_val_loss = float('inf')

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs}")
        for batch in pbar:
            f = batch['source'].to(Config.device)  # [b,1,h,w] 源项
            u_true = batch['solution'].to(Config.device)  # [b,1,h,w] 真实解
            boundary = batch.get('boundary', None)  # 边界条件
            boundary_info = boundary.to(Config.device) if boundary is not None else None

            # 前向传播
            u_pred, branch_outputs = model(f, boundary_info)

            # 计算损失
            loss = criterion(u_pred, u_true,
                             u_smooth=branch_outputs['smooth'],
                             u_high=branch_outputs['high'],
                             boundary_info=boundary_info)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        val_loss = validate(model, val_loader, criterion, Config.device)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gfno.pth')

    print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            f = batch['source'].to(device)
            u_true = batch['solution'].to(device)
            boundary = batch.get('boundary', None)
            boundary_info = boundary.to(device) if boundary is not None else None

            u_pred, _ = model(f, boundary_info)
            loss = criterion(u_pred, u_true, boundary_info=boundary_info)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def get_dataloaders(config):
    """加载64×64 Darcy Flow数据（适配Zenodo .pt格式）"""
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split

    class Darcy64Dataset(Dataset):
        def __init__(self, pt_path):
            # 加载 .pt 文件
            data = torch.load(pt_path)

            # 数据格式: {'x': coeff, 'y': solution}
            # 添加通道维度: [N, H, W] → [N, 1, H, W]
            self.coeff = data['x'].unsqueeze(1).float()
            self.solution = data['y'].unsqueeze(1).float()

        def __len__(self):
            return len(self.coeff)

        def __getitem__(self, idx):
            return {'source': self.coeff[idx], 'solution': self.solution[idx]}

    # 数据路径（64×64）
    train_path = './data/darcy_train_64.pt'
    test_path = './data/darcy_test_64.pt'

    full_train_dataset = Darcy64Dataset(train_path)
    test_dataset = Darcy64Dataset(test_path)

    # 划分训练/验证集
    train_size = int(config.train_ratio * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train()
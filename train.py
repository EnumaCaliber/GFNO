import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import os
import argparse
from models.gfno import GFNO


# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════

class Config:
    # 数据
    domain_type  = 'l_shape'   # 'l_shape' | 'circle_hole' | 'square'
    n_samples    = 500
    batch_size   = 16          # 减小 batch 适配边界分支显存
    train_ratio  = 0.8

    # 模型
    smooth_modes    = (12, 12)
    smooth_width    = 64
    high_modes      = (16, 16)
    high_width      = 32
    n_boundary_pts  = 64
    fusion_type     = 'attention'

    # 训练
    epochs       = 50
    lr           = 1e-3
    weight_decay = 1e-4
    grad_clip    = 1.0
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 损失权重
    lambda_reg   = 0.05        # 高频残差正则


# ══════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════

class IrregularDomainDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location='cpu', weights_only=False)

        self.coeff    = data['x'].unsqueeze(1).float()   # [N, 1, H, W]
        self.solution = data['y'].unsqueeze(1).float()   # [N, 1, H, W]
        self.boundary = data.get('boundary', None)       # [N, n_bc, 3] or None
        self.mask     = data.get('mask', None)           # [H, W] bool

        # 归一化（只在域内点统计）
        if self.mask is not None:
            m = self.mask.unsqueeze(0).unsqueeze(0).float()
            n_valid = m.sum()
            self.sol_mean = (self.solution * m).sum() / n_valid
            self.sol_std  = torch.sqrt(((self.solution * m - self.sol_mean * m) ** 2).sum() / n_valid) + 1e-8
        else:
            self.sol_mean = self.solution.mean()
            self.sol_std  = self.solution.std() + 1e-8

        self.solution = (self.solution - self.sol_mean) / self.sol_std

    def __len__(self):
        return len(self.coeff)

    def __getitem__(self, idx):
        item = {'source': self.coeff[idx], 'solution': self.solution[idx]}
        if self.boundary is not None:
            item['boundary'] = self.boundary[idx]
        if self.mask is not None:
            item['mask'] = self.mask
        return item


# ══════════════════════════════════════════════════════
#  损失函数
# ══════════════════════════════════════════════════════

def relative_l2(pred, target, mask=None):
    if mask is not None:
        m = mask.unsqueeze(0).unsqueeze(0).float().to(pred.device)
        pred, target = pred * m, target * m
    return torch.norm(pred - target) / (torch.norm(target) + 1e-8)


class CompositeLoss(nn.Module):
    def __init__(self, lambda_reg=0.05):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.mse = nn.MSELoss()

    def forward(self, pred, target, u_smooth=None, u_high=None, mask=None):
        if mask is not None:
            m = mask.unsqueeze(0).unsqueeze(0).float().to(pred.device)
            pred_m   = pred * m
            target_m = target * m
            smooth_m = u_smooth * m if u_smooth is not None else None
            high_m   = u_high   * m if u_high   is not None else None
        else:
            pred_m, target_m = pred, target
            smooth_m, high_m = u_smooth, u_high

        loss_data = self.mse(pred_m, target_m)

        # 高频分支残差正则：鼓励 u_high 逼近 (target - u_smooth)
        loss_reg = 0.0
        if smooth_m is not None and high_m is not None:
            residual = (target_m - smooth_m).detach()
            loss_reg = self.mse(high_m, residual)

        return loss_data + self.lambda_reg * loss_reg


# ══════════════════════════════════════════════════════
#  数据加载（自动生成）
# ══════════════════════════════════════════════════════

def get_dataloaders(cfg):
    train_path = f'./data/{cfg.domain_type}_train.pt'

    if not os.path.exists(train_path):
        print("数据文件不存在，正在生成...")
        import sys; sys.path.insert(0, '.')
        from data.generate_irregular import generate_dataset
        generate_dataset(cfg.domain_type, N=64, n_samples=cfg.n_samples)

    dataset = IrregularDomainDataset(train_path)
    n_train = int(cfg.train_ratio * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    kw = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **kw)
    return train_loader, val_loader


# ══════════════════════════════════════════════════════
#  训练 / 验证
# ══════════════════════════════════════════════════════

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_rel = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            f       = batch['source'].to(device)
            u_true  = batch['solution'].to(device)
            bc_info = batch.get('boundary', None)
            if bc_info is not None: bc_info = bc_info.to(device)
            mask    = batch.get('mask', None)
            if mask is not None: mask = mask[0].to(device)

            u_pred, branches = model(f, bc_info, domain_mask=mask)
            loss    = criterion(u_pred, u_true,
                                u_smooth=branches['smooth'],
                                u_high=branches['high'], mask=mask)
            rel_l2  = relative_l2(u_pred, u_true, mask)
            total_loss += loss.item()
            total_rel  += rel_l2.item()

    n = len(loader)
    return total_loss / n, total_rel / n


def train(cfg=None):
    if cfg is None:
        cfg = Config()

    print(f"设备: {cfg.device} | 域: {cfg.domain_type}")

    model = GFNO(
        smooth_modes=cfg.smooth_modes, smooth_width=cfg.smooth_width,
        high_modes=cfg.high_modes,   high_width=cfg.high_width,
        n_boundary_points=cfg.n_boundary_pts, fusion_type=cfg.fusion_type
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = CompositeLoss(lambda_reg=cfg.lambda_reg)

    train_loader, val_loader = get_dataloaders(cfg)

    best_val  = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        ep_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{cfg.epochs}")
        for batch in pbar:
            f       = batch['source'].to(cfg.device)
            u_true  = batch['solution'].to(cfg.device)
            bc_info = batch.get('boundary', None)
            if bc_info is not None: bc_info = bc_info.to(cfg.device)
            mask    = batch.get('mask', None)
            if mask is not None: mask = mask[0].to(cfg.device)

            u_pred, branches = model(f, bc_info, domain_mask=mask)
            loss = criterion(u_pred, u_true,
                             u_smooth=branches['smooth'],
                             u_high=branches['high'], mask=mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            ep_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        val_loss, val_rel = validate(model, val_loader, criterion, cfg.device)
        avg_train = sum(ep_losses) / len(ep_losses)

        print(f"  Train={avg_train:.5f} | Val={val_loss:.5f} | RelL2={val_rel:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': val_loss, 'val_rel_l2': val_rel,
            }, f'checkpoints/best_{cfg.domain_type}.pth')
            print(f"  ✅ 保存最佳模型 (val={val_loss:.5f})")

    print(f"\n训练完成！最佳验证损失: {best_val:.5f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='l_shape',
                   choices=['l_shape', 'circle_hole', 'square'])
    p.add_argument('--epochs',   type=int, default=50)
    p.add_argument('--batch',    type=int, default=16)
    p.add_argument('--n',        type=int, default=500)
    args = p.parse_args()

    cfg = Config()
    cfg.domain_type = args.domain
    cfg.epochs      = args.epochs
    cfg.batch_size  = args.batch
    cfg.n_samples   = args.n
    train(cfg)
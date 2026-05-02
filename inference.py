import torch
import numpy as np
import matplotlib.pyplot as plt
from models.gfno import GFNO
from torch.utils.data import DataLoader, Dataset
import h5py
import os


# ========== 配置 ==========
class InferenceConfig:
    smooth_modes = (12, 12)
    smooth_width = 64
    high_modes = (16, 16)
    high_width = 32
    fusion_type = 'attention'


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'best_gfno.pth'  # 训练好的模型权重

    use_boundary_branch = True  # 推理时是否使用边界分支（设为False则边界分支输出0）
    n_boundary_points = 200  # 边界采样点数


def load_model(config, checkpoint_path):
    """加载训练好的模型"""
    model = GFNO(
        smooth_modes=config.smooth_modes,
        smooth_width=config.smooth_width,
        high_modes=config.high_modes,
        high_width=config.high_width,
        fusion_type=config.fusion_type
    ).to(config.device)

    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"模型加载成功: {checkpoint_path}")
    return model


def make_interior_coords(h, w, device):
    """生成内部点坐标网格，归一化到 [-1, 1]"""
    x = torch.linspace(-1, 1, w, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    return coords.unsqueeze(0)  # [1, H*W, 2]


def predict(model, source, device, use_boundary=False, interior_coords=None):

    model.eval()

    # 确保输入形状正确
    original_shape = source.shape
    if source.dim() == 2:
        source = source.unsqueeze(0).unsqueeze(0)  # [H, W] → [1, 1, H, W]
    elif source.dim() == 3:
        if source.shape[0] == 1:
            source = source.unsqueeze(0)  # [1, H, W] → [1, 1, H, W]
        else:
            source = source.unsqueeze(1)  # [H, W, C] 的情况，按需调整
    elif source.dim() == 4 and source.shape[1] != 1:
        source = source.unsqueeze(1)

    source = source.to(device)
    batch, _, h, w = source.shape

    # 生成内部坐标（如果未提供）
    if interior_coords is None:
        interior_coords = make_interior_coords(h, w, device)

    # 扩展到batch维度
    batch_coords = interior_coords.expand(batch, -1, -1)

    with torch.no_grad():
        if use_boundary:
            # 使用边界分支（需要边界信息）
            # 注意：这里简化处理，实际使用时需要根据问题提供真实的边界条件
            boundary_info = None
            pred, branch_outputs = model(source, boundary_info=boundary_info, interior_coords=batch_coords)
        else:
            # 不使用边界分支
            pred, branch_outputs = model(source, boundary_info=None, interior_coords=batch_coords)

    # 移除batch和channel维度
    pred = pred.squeeze().cpu()
    # 如果原始输入是2D，确保输出也是2D
    if len(original_shape) == 2:
        pred = pred.squeeze()

    return pred, branch_outputs


def predict_batch(model, sources, batch_size=32, use_boundary=False):

    model.eval()
    device = next(model.parameters()).device
    _, _, h, w = sources.shape

    # 生成内部坐标
    interior_coords = make_interior_coords(h, w, device)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size].to(device)
            batch_coords = interior_coords.expand(batch.shape[0], -1, -1)

            if use_boundary:
                pred, _ = model(batch, boundary_info=None, interior_coords=batch_coords)
            else:
                pred, _ = model(batch, boundary_info=None, interior_coords=batch_coords)

            all_preds.append(pred.cpu())

    return torch.cat(all_preds, dim=0)


def visualize_prediction(source, true_solution, pred_solution, save_path=None, show=True):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 输入场（渗透率）
    im1 = axes[0, 0].imshow(source.squeeze(), cmap='viridis')
    axes[0, 0].set_title('Input Field (Permeability a(x))')
    plt.colorbar(im1, ax=axes[0, 0])

    # 真实解
    im2 = axes[0, 1].imshow(true_solution.squeeze(), cmap='plasma')
    axes[0, 1].set_title('Ground Truth (Pressure u(x))')
    plt.colorbar(im2, ax=axes[0, 1])

    # 预测解
    im3 = axes[1, 0].imshow(pred_solution.squeeze(), cmap='plasma')
    axes[1, 0].set_title('GFNO Prediction')
    plt.colorbar(im3, ax=axes[1, 0])

    # 绝对误差
    error = np.abs(true_solution.squeeze() - pred_solution.squeeze())
    im4 = axes[1, 1].imshow(error, cmap='hot')
    axes[1, 1].set_title(f'Absolute Error (max={error.max():.4f})')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return error


def evaluate_on_testset(model, test_loader, device, num_samples=None, use_boundary=False):

    model.eval()

    all_preds = []
    all_targets = []
    all_sources = []
    rel_l2_errors = []

    # 获取网格信息
    sample_batch = next(iter(test_loader))
    _, _, h, w = sample_batch['source'].shape
    interior_coords = make_interior_coords(h, w, device)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if num_samples is not None and i >= num_samples:
                break

            source = batch['source'].to(device)
            target = batch['solution'].to(device)
            batch_coords = interior_coords.expand(source.shape[0], -1, -1)

            if use_boundary:
                pred, _ = model(source, boundary_info=None, interior_coords=batch_coords)
            else:
                pred, _ = model(source, boundary_info=None, interior_coords=batch_coords)

            # 计算相对L2误差
            for j in range(len(pred)):
                error = torch.norm(pred[j] - target[j]) / torch.norm(target[j])
                rel_l2_errors.append(error.item())

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_sources.append(source.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    sources = torch.cat(all_sources, dim=0)

    # 计算详细指标
    rel_l2_errors = np.array(rel_l2_errors)
    abs_errors = torch.abs(preds - targets)

    metrics = {
        'Rel_L2_mean': np.mean(rel_l2_errors),
        'Rel_L2_std': np.std(rel_l2_errors),
        'Rel_L2_median': np.median(rel_l2_errors),
        'Rel_L2_max': np.max(rel_l2_errors),
        'MAE': abs_errors.mean().item(),
        'Max_AE': abs_errors.max().item(),
        'RMSE': torch.sqrt((abs_errors ** 2).mean()).item()
    }

    return sources, targets, preds, metrics


def print_metrics(metrics, title="评估指标"):
    """打印评估指标"""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"{'=' * 50}\n")


# ========== 数据加载器（适配64x64 Darcy数据）==========
class Darcy64Dataset(Dataset):
    """64×64 Darcy Flow 数据集（适配Zenodo .pt格式）"""

    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.coeff = data['x'].unsqueeze(1).float()  # [N, 1, H, W]
        self.solution = data['y'].unsqueeze(1).float()  # [N, 1, H, W]

    def __len__(self):
        return len(self.coeff)

    def __getitem__(self, idx):
        return {'source': self.coeff[idx], 'solution': self.solution[idx]}


def get_test_loader(data_dir='./data', batch_size=32):
    """获取测试数据加载器"""
    test_path = os.path.join(data_dir, 'darcy_test_64.pt')

    if not os.path.exists(test_path):
        # 尝试其他可能的文件名
        alt_paths = [
            os.path.join(data_dir, 'piececonst_64_test.pt'),
            os.path.join(data_dir, 'darcy_test.pt'),
            os.path.join(data_dir, 'test.pt')
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                test_path = alt
                break
        else:
            raise FileNotFoundError(f"测试数据未找到: {test_path}")

    test_dataset = Darcy64Dataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"测试集加载成功: {test_path}, 样本数: {len(test_dataset)}")
    return test_loader


# ========== 主程序 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("GFNO 推理与评估")
    print("=" * 60)

    # 1. 加载模型
    config = InferenceConfig()
    model = load_model(config, config.checkpoint_path)
    print(f"设备: {config.device}")

    # 2. 加载测试数据
    test_loader = get_test_loader('./data', batch_size=32)

    # 3. 在测试集上评估
    print("\n正在评估测试集...")
    sources, targets, preds, metrics = evaluate_on_testset(
        model, test_loader, config.device,
        num_samples=20,  # 评估20个样本
        use_boundary=config.use_boundary_branch
    )

    print_metrics(metrics, "GFNO 测试集评估结果")

    # 4. 可视化对比（前3个样本）
    print("\n正在生成可视化对比图...")
    for i in range(min(3, len(sources))):
        visualize_prediction(
            sources[i].squeeze().numpy(),
            targets[i].squeeze().numpy(),
            preds[i].squeeze().numpy(),
            save_path=f"gfno_prediction_sample_{i + 1}.png",
            show=True
        )

    # 5. 单样本详细分析
    print("\n" + "=" * 50)
    print("单样本详细分析")
    print("=" * 50)

    # 取第一个测试样本
    single_source = sources[0]
    single_true = targets[0]

    # 使用 predict 函数（会自动生成 interior_coords）
    pred, branch_outs = predict(
        model, single_source, config.device,
        use_boundary=config.use_boundary_branch
    )

    # 计算误差
    error = torch.abs(pred - single_true).numpy()
    rel_l2 = torch.norm(pred - single_true) / torch.norm(single_true)

    print(f"\n样本 1:")
    print(f"  输入场范围: [{single_source.min():.4f}, {single_source.max():.4f}]")
    print(f"  真实解范围: [{single_true.min():.4f}, {single_true.max():.4f}]")
    print(f"  预测解范围: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"  相对L2误差: {rel_l2:.6f}")
    print(f"  最大绝对误差: {error.max():.6f}")
    print(f"  平均绝对误差: {error.mean():.6f}")

    # 显示各分支输出统计
    print(f"\n三分支输出统计:")
    print(f"  平滑分支: min={branch_outs['smooth'].min():.4f}, max={branch_outs['smooth'].max():.4f}, "
          f"mean={branch_outs['smooth'].mean():.4f}")
    print(f"  边界分支: min={branch_outs['boundary'].min():.4f}, max={branch_outs['boundary'].max():.4f}, "
          f"mean={branch_outs['boundary'].mean():.4f}")
    print(f"  高频分支: min={branch_outs['high'].min():.4f}, max={branch_outs['high'].max():.4f}, "
          f"mean={branch_outs['high'].mean():.4f}")

    # 6. 批量预测示例
    print("\n" + "=" * 50)
    print("批量预测示例")
    print("=" * 50)

    batch_sources = sources[:8]  # 取8个样本
    batch_preds = predict_batch(model, batch_sources, batch_size=4)
    print(f"批量预测: 输入形状 {batch_sources.shape} → 输出形状 {batch_preds.shape}")

    print("\n推理完成！")
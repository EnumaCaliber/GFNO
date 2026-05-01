import torch
import numpy as np
import matplotlib.pyplot as plt
from models.gfno import GFNO
from torch.utils.data import DataLoader


# ========== 配置 ==========
class InferenceConfig:
    # 模型参数（必须与训练时一致）
    smooth_modes = (12, 12)
    smooth_width = 64
    high_modes = (16, 16)
    high_width = 32
    fusion_type = 'attention'

    # 数据参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'best_gfno.pth'  # 训练好的模型权重


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
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.eval()  # 切换到评估模式
    print(f"模型加载成功: {checkpoint_path}")
    return model


def predict(model, source, device):
    """
    单样本预测
    source: [1, H, W] 或 [1, 1, H, W] 渗透率场
    返回: [1, H, W] 压力场预测
    """
    model.eval()

    # 确保输入形状正确
    if source.dim() == 3:
        source = source.unsqueeze(0)  # [1, H, W] → [1, 1, H, W]
    elif source.dim() == 4 and source.shape[1] != 1:
        source = source.unsqueeze(1)  # [1, H, W] 或类似情况

    source = source.to(device)

    with torch.no_grad():
        pred, branch_outputs = model(source, boundary_info=None)

    return pred.cpu().squeeze(), branch_outputs


def predict_batch(model, sources, batch_size=32):
    """
    批量预测
    sources: [N, 1, H, W] 渗透率场
    returns: [N, 1, H, W] 压力场预测
    """
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size].to(device)
            pred, _ = model(batch, boundary_info=None)
            all_preds.append(pred.cpu())

    return torch.cat(all_preds, dim=0)


def visualize_prediction(source, true_solution, pred_solution, save_path=None):
    """
    可视化对比：输入场、真实解、预测解、误差
    """
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
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    plt.show()


def test_on_testset(model, test_loader, device, num_samples=5):
    """
    在测试集上评估并可视化
    """
    model.eval()

    # 收集预测结果
    all_preds = []
    all_targets = []
    all_sources = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            source = batch['source'].to(device)
            target = batch['solution'].to(device)

            pred, _ = model(source, boundary_info=None)

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_sources.append(source.cpu())

            if i >= num_samples - 1:
                break

    # 合并
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    sources = torch.cat(all_sources, dim=0)

    # 计算指标
    rel_l2_errors = []
    for i in range(len(preds)):
        error = torch.norm(preds[i] - targets[i]) / torch.norm(targets[i])
        rel_l2_errors.append(error.item())

    print(f"测试集评估 (共{len(preds)}个样本):")
    print(f"  平均相对L2误差: {np.mean(rel_l2_errors):.6f}")
    print(f"  中位数相对L2误差: {np.median(rel_l2_errors):.6f}")
    print(f"  最大相对L2误差: {np.max(rel_l2_errors):.6f}")
    print(f"  标准差: {np.std(rel_l2_errors):.6f}")

    return sources, targets, preds, rel_l2_errors


if __name__ == '__main__':
    # ========== 1. 加载模型 ==========
    config = InferenceConfig()
    model = load_model(config, config.checkpoint_path)

    # ========== 2. 加载测试数据 ==========
    from train import get_dataloaders, Config as TrainConfig

    # 使用训练脚本中的配置加载测试数据
    train_config = TrainConfig()
    train_config.data_path = './data'
    _, _, test_loader = get_dataloaders(train_config)

    print(f"测试集批次数: {len(test_loader)}")

    # ========== 3. 评估测试集 ==========
    sources, targets, preds, errors = test_on_testset(
        model, test_loader, config.device, num_samples=10
    )

    # ========== 4. 可视化几个样本 ==========
    for i in range(min(3, len(sources))):
        visualize_prediction(
            sources[i],
            targets[i],
            preds[i],
            save_path=f"prediction_sample_{i + 1}.png"
        )

    # ========== 5. 单样本推理示例 ==========
    print("\n" + "=" * 50)
    print("单样本推理示例")
    print("=" * 50)

    # 取第一个测试样本
    single_source = sources[0]
    single_true = targets[0]

    pred, branch_outs = predict(model, single_source, config.device)
    print(f"输入形状: {single_source.shape}")
    print(f"预测形状: {pred.shape}")
    print(f"相对L2误差: {errors[0]:.6f}")

    # 查看各分支的贡献
    print(f"\n各分支输出范围:")
    print(f"  平滑分支: [{branch_outs['smooth'].min():.4f}, {branch_outs['smooth'].max():.4f}]")
    print(f"  边界分支: [{branch_outs['boundary'].min():.4f}, {branch_outs['boundary'].max():.4f}]")
    print(f"  高频分支: [{branch_outs['high'].min():.4f}, {branch_outs['high'].max():.4f}]")
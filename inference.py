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
    return model


def predict(model, source, device):

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))


    im1 = axes[0, 0].imshow(source.squeeze(), cmap='viridis')
    axes[0, 0].set_title('Input Field (Permeability a(x))')
    plt.colorbar(im1, ax=axes[0, 0])


    im2 = axes[0, 1].imshow(true_solution.squeeze(), cmap='plasma')
    axes[0, 1].set_title('Ground Truth (Pressure u(x))')
    plt.colorbar(im2, ax=axes[0, 1])


    im3 = axes[1, 0].imshow(pred_solution.squeeze(), cmap='plasma')
    axes[1, 0].set_title('GFNO Prediction')
    plt.colorbar(im3, ax=axes[1, 0])


    error = np.abs(true_solution.squeeze() - pred_solution.squeeze())
    im4 = axes[1, 1].imshow(error, cmap='hot')
    axes[1, 1].set_title(f'Absolute Error (max={error.max():.4f})')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def test_on_testset(model, test_loader, device, num_samples=5):

    model.eval()


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


    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    sources = torch.cat(all_sources, dim=0)


    rel_l2_errors = []
    for i in range(len(preds)):
        error = torch.norm(preds[i] - targets[i]) / torch.norm(targets[i])
        rel_l2_errors.append(error.item())

    return sources, targets, preds, rel_l2_errors


if __name__ == '__main__':

    config = InferenceConfig()
    model = load_model(config, config.checkpoint_path)


    from train import get_dataloaders, Config as TrainConfig


    train_config = TrainConfig()
    train_config.data_path = './data'
    _, _, test_loader = get_dataloaders(train_config)


    sources, targets, preds, errors = test_on_testset(
        model, test_loader, config.device, num_samples=10
    )


    for i in range(min(3, len(sources))):
        visualize_prediction(
            sources[i],
            targets[i],
            preds[i],
            save_path=f"prediction_sample_{i + 1}.png"
        )




    single_source = sources[0]
    single_true = targets[0]
    pred, branch_outs = predict(model, single_source, config.device)


    print(f"  smooth branch: [{branch_outs['smooth'].min():.4f}, {branch_outs['smooth'].max():.4f}]")
    print(f"  boundary branch: [{branch_outs['boundary'].min():.4f}, {branch_outs['boundary'].max():.4f}]")
    print(f"  high frequency: [{branch_outs['high'].min():.4f}, {branch_outs['high'].max():.4f}]")
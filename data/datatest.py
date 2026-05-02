import torch
import matplotlib.pyplot as plt

# 加载数据
data = torch.load('darcy_train_64.pt')

# 画前3个样本的输入和输出
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i in range(3):
    # 左边：输入（渗透率场）
    im1 = axes[i, 0].imshow(data['x'][i], cmap='viridis')
    axes[i, 0].set_title(f'Sample {i + 1}: Input (Permeability a(x))')
    axes[i, 0].set_xlabel('x')
    axes[i, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[i, 0])

    # 右边：输出（压力场）
    im2 = axes[i, 1].imshow(data['y'][i], cmap='plasma')
    axes[i, 1].set_title(f'Sample {i + 1}: Output (Pressure u(x))')
    axes[i, 1].set_xlabel('x')
    axes[i, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[i, 1])

plt.tight_layout()
plt.savefig('darcy_samples.png', dpi=150)
plt.show()

print("图片已保存: darcy_samples.png")
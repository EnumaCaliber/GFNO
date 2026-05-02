import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """三分支自适应融合：attention / learnable / concat"""

    def __init__(self, in_channels=1, fusion_type='attention'):
        super().__init__()
        self.fusion_type = fusion_type
        total = in_channels * 3  # 三分支拼接后的通道数

        if fusion_type == 'attention':
            self.weight_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(total, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            )
        elif fusion_type == 'learnable':
            self.alpha = nn.Parameter(torch.ones(1) / 3)
            self.beta  = nn.Parameter(torch.ones(1) / 3)
            self.gamma = nn.Parameter(torch.ones(1) / 3)
        else:  # concat
            self.conv = nn.Conv2d(total, in_channels, 1)

    def forward(self, u_smooth, u_boundary, u_high):
        if self.fusion_type == 'attention':
            concat  = torch.cat([u_smooth, u_boundary, u_high], dim=1)  # [B, 3C, H, W]
            weights = self.weight_net(concat)                            # [B, 3]
            w0 = weights[:, 0:1, None, None]
            w1 = weights[:, 1:2, None, None]
            w2 = weights[:, 2:3, None, None]
            return w0 * u_smooth + w1 * u_boundary + w2 * u_high

        elif self.fusion_type == 'learnable':
            total = self.alpha.abs() + self.beta.abs() + self.gamma.abs() + 1e-8
            return (self.alpha.abs() / total * u_smooth +
                    self.beta.abs()  / total * u_boundary +
                    self.gamma.abs() / total * u_high)

        else:  # concat
            return self.conv(torch.cat([u_smooth, u_boundary, u_high], dim=1))
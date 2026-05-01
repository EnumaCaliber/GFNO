import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):

    def __init__(self, in_channels, fusion_type='attention'):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'attention':
            # 注意力式融合：根据输入特征动态生成权重
            self.weight_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels * 3, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            )
        elif fusion_type == 'learnable':
            # 可学习静态权重
            self.alpha = nn.Parameter(torch.ones(1) / 3)
            self.beta = nn.Parameter(torch.ones(1) / 3)
            self.gamma = nn.Parameter(torch.ones(1) / 3)
        else:  # 'concat'
            self.conv = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, u_smooth, u_boundary, u_high):
        """
        三个分支输出: [batch, 1, h, w]
        """
        if self.fusion_type == 'attention':
            # 拼接三个输出
            concat = torch.cat([u_smooth, u_boundary, u_high], dim=1)
            weights = self.weight_net(concat)  # [b, 3]
            # 加权求和
            return (weights[:, 0:1].unsqueeze(-1).unsqueeze(-1) * u_smooth +
                    weights[:, 1:2].unsqueeze(-1).unsqueeze(-1) * u_boundary +
                    weights[:, 2:3].unsqueeze(-1).unsqueeze(-1) * u_high)

        elif self.fusion_type == 'learnable':
            # 归一化确保和为1
            total = self.alpha + self.beta + self.gamma
            return (self.alpha / total) * u_smooth + (self.beta / total) * u_boundary + (self.gamma / total) * u_high

        else:  # concat
            concat = torch.cat([u_smooth, u_boundary, u_high], dim=1)
            return self.conv(concat)
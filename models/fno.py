import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w    = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=4, out_channels=1, n_layers=4, padding=9):
        """
        in_channels: input feature dim per point (coords + grid = 4)
        """
        super().__init__()
        self.padding = padding
        self.fc0     = nn.Linear(in_channels, width)
        self.blocks  = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.fc1     = nn.Linear(width, 128)
        self.fc2     = nn.Linear(128, out_channels)

    def forward(self, x):
        """
        x: (B, s1, s2, in_channels)
        returns: (B, s1, s2, out_channels)
        """
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for block in self.blocks:
            x = block(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def get_grid(shape, device):
        B, s1, s2 = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, s1, device=device).reshape(1, s1, 1, 1).repeat(B, 1, s2, 1)
        gridy = torch.linspace(0, 1, s2, device=device).reshape(1, 1, s2, 1).repeat(B, s1, 1, 1)
        return torch.cat([gridx, gridy], dim=-1)
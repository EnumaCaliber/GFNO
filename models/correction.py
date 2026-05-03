import torch
import torch.nn as nn


class LowRankCorrection(nn.Module):
    """
    Approximates the boundary correction term:
        delta_G(x, y; geom) = sum_k phi_k(x; geom) * psi_k(y; geom)

    Applied as:
        correction(x) = phi(x) * mean_y[psi(y)]

    This is an O(N) approximation of the O(N^2) full kernel.

    input_dim: per-point feature dim (coords + sdf = 3)
    rank:      number of basis functions
    hidden:    MLP hidden dim
    """

    def __init__(self, input_dim=3, rank=4, hidden=64):
        super().__init__()
        self.rank = rank

        self.phi_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )

        self.psi_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )

        self.proj = nn.Linear(rank, 1)

    def forward(self, feat, mask=None):
        """
        feat: (B, s1, s2, input_dim)  coords + sdf
        mask: (B, s1, s2) optional, 1 = physical domain
        returns: (B, s1, s2, 1)
        """
        phi = self.phi_net(feat)   # (B, s1, s2, rank)
        psi = self.psi_net(feat)   # (B, s1, s2, rank)

        if mask is not None:
            # only aggregate over physical domain
            m = mask.unsqueeze(-1)                          # (B, s1, s2, 1)
            psi_mean = (psi * m).sum(dim=(1, 2), keepdim=True) / \
                       (m.sum(dim=(1, 2), keepdim=True) + 1e-8)
        else:
            psi_mean = psi.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1, rank)

        combined = phi * psi_mean                           # (B, s1, s2, rank)
        out = self.proj(combined)                           # (B, s1, s2, 1)
        return out
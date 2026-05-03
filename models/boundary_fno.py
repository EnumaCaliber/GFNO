import torch
import torch.nn as nn
from models.fno import FNO2d
from models.correction import LowRankCorrection


class BoundaryFNO(nn.Module):
    """
    G(x, y; geom) = G0(x-y) + delta_G(x, y; geom)

    G0:      learned by standard FNO (translation-invariant)
    delta_G: learned by LowRankCorrection conditioned on SDF geometry
    """

    def __init__(self, modes1, modes2, width, rank=4, n_layers=4, padding=9):
        super().__init__()

        # FNO: takes (coords + grid) = 4 channels
        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            in_channels=4,
            out_channels=1,
            n_layers=n_layers,
            padding=padding,
        )

        # Correction: takes (coords + sdf) = 3 channels
        self.correction = LowRankCorrection(input_dim=3, rank=rank, hidden=64)

    def forward(self, coords, sdf):
        """
        coords: (B, s1, s2, 2)  physical coordinates (X, Y)
        sdf:    (B, s1, s2)     signed distance function

        returns: (B, s1, s2)
        """
        # build grid and append to coords for FNO
        grid    = FNO2d.get_grid(coords.shape[:3], coords.device)  # (B, s1, s2, 2)
        fno_in  = torch.cat([coords, grid], dim=-1)                # (B, s1, s2, 4)
        fno_out = self.fno(fno_in)                                 # (B, s1, s2, 1)

        # build correction input
        sdf_in   = sdf.unsqueeze(-1)                               # (B, s1, s2, 1)
        corr_in  = torch.cat([coords, sdf_in], dim=-1)            # (B, s1, s2, 3)
        mask     = (sdf > 0).float()                               # physical domain mask
        corr_out = self.correction(corr_in, mask=mask)            # (B, s1, s2, 1)

        out = (fno_out + corr_out).squeeze(-1)                    # (B, s1, s2)
        return fno_out.squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
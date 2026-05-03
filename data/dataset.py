import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt


def compute_sdf(coords_x, coords_y):
    """
    coords_x, coords_y: (N, s1, s2) physical coordinates
    Returns normalized SDF: (N, s1, s2)
    Positive inside physical domain, negative outside.
    """
    N, s1, s2 = coords_x.shape
    sdf_all = np.zeros((N, s1, s2), dtype=np.float32)

    for i in range(N):
        # Simple boundary mask: border pixels = outside
        mask = np.ones((s1, s2), dtype=np.float32)
        mask[0, :]  = 0
        mask[-1, :] = 0
        mask[:, 0]  = 0
        mask[:, -1] = 0

        dist_in  = distance_transform_edt(mask)
        dist_out = distance_transform_edt(1 - mask)
        sdf = dist_in - dist_out
        sdf = sdf / (np.abs(sdf).max() + 1e-8)
        sdf_all[i] = sdf

    return sdf_all


class ElasticityDataset(Dataset):
    def __init__(self, coords, sdf, targets):
        """
        coords:  (N, s1, s2, 2)
        sdf:     (N, s1, s2)
        targets: (N, s1, s2)
        """
        self.coords  = coords
        self.sdf     = sdf
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.coords[idx], self.sdf[idx], self.targets[idx]


def load_elasticity(cfg):
    """
    Load Omesh elasticity data and return train/test DataLoaders.
    cfg: dict with keys INPUT_X, INPUT_Y, OUTPUT_Sigma, ntrain, ntest,
         batch_size, r1, r2, s1, s2
    """
    print("Loading data...")
    inputX = torch.tensor(
        np.load(cfg['INPUT_X']), dtype=torch.float).permute(2, 0, 1)  # (N, 65, 41)
    inputY = torch.tensor(
        np.load(cfg['INPUT_Y']), dtype=torch.float).permute(2, 0, 1)
    output = torch.tensor(
        np.load(cfg['OUTPUT_Sigma']), dtype=torch.float).permute(2, 0, 1)

    r1, r2 = cfg['r1'], cfg['r2']
    s1, s2 = cfg['s1'], cfg['s2']

    inputX_s = inputX[:, ::r1, ::r2][:, :s1, :s2]
    inputY_s = inputY[:, ::r1, ::r2][:, :s1, :s2]
    output_s = output[:, ::r1, ::r2][:, :s1, :s2]

    print("Computing SDF...")
    sdf_np = compute_sdf(inputX_s.numpy(), inputY_s.numpy())
    sdf    = torch.tensor(sdf_np, dtype=torch.float)  # (N, s1, s2)

    coords = torch.stack([inputX_s, inputY_s], dim=-1)  # (N, s1, s2, 2)

    ntrain = cfg['ntrain']
    ntest  = cfg['ntest']

    train_ds = ElasticityDataset(coords[:ntrain], sdf[:ntrain], output_s[:ntrain])
    test_ds  = ElasticityDataset(coords[-ntest:], sdf[-ntest:], output_s[-ntest:])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Grid: {s1}x{s2}")
    return train_loader, test_loader
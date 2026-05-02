"""
visualize_data.py - Visualize irregular domain PDE datasets
Usage: python visualize_data.py --domain l_shape --n 4
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def visualize(domain_type='l_shape', n_samples=4, split='train', save_path=None):
    path = f'{domain_type}_{split}.pt'
    data = torch.load(path, weights_only=False, map_location='cpu')

    x        = data['x'].float()         # [N, H, W] coefficient field
    y        = data['y'].float()         # [N, H, W] PDE solution
    boundary = data['boundary'].float()  # [N, n_bc, 3]
    mask     = data['mask'].float()      # [H, W]

    N = len(x)
    n_samples = min(n_samples, N)
    nan_mask = (mask == 0).numpy()       # outside domain -> NaN

    print(f"Dataset: {domain_type} | split={split} | {N} samples")
    print(f"  Coefficient x : {tuple(x.shape[1:])}  [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Solution    y : {tuple(y.shape[1:])}  [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Boundary pts  : {tuple(boundary.shape[1:])}  coord range [{boundary[..., :2].min():.2f}, {boundary[..., :2].max():.2f}]")
    print(f"  Domain mask   : {mask.sum().int()} / {mask.numel()} valid pts ({mask.mean()*100:.1f}%)")

    col_titles = [
        'Coefficient a(x)',
        'True Solution u*(x)',
        'Boundary Points',
        'Coeff. Distribution',
        'Solution Distribution',
        'Boundary Values g(x,y)'
    ]

    fig = plt.figure(figsize=(22, 4.5 * n_samples))
    fig.suptitle(f'Dataset Visualization — {domain_type}  ({split} split, {N} samples)',
                 fontsize=14, y=1.01, fontweight='bold')

    for row, idx in enumerate(np.linspace(0, N - 1, n_samples, dtype=int)):
        xi  = x[idx].numpy().copy()
        yi  = y[idx].numpy().copy()
        bci = boundary[idx].numpy()

        xi[nan_mask] = np.nan
        yi[nan_mask] = np.nan

        for col in range(6):
            ax = fig.add_subplot(n_samples, 6, row * 6 + col + 1)

            if col == 0:
                im = ax.imshow(xi, origin='lower', cmap='YlOrRd')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            elif col == 1:
                vabs = np.nanmax(np.abs(yi))
                im = ax.imshow(yi, origin='lower', cmap='RdBu_r', vmin=-vabs, vmax=vabs)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            elif col == 2:
                ax.imshow(mask.numpy(), origin='lower', cmap='Greys_r', alpha=0.3, vmin=0, vmax=1)
                sc = ax.scatter(
                    (bci[:, 0] + 1) / 2 * (mask.shape[1] - 1),
                    (bci[:, 1] + 1) / 2 * (mask.shape[0] - 1),
                    c=bci[:, 2], cmap='coolwarm', s=20, zorder=3,
                    vmin=bci[:, 2].min(), vmax=bci[:, 2].max()
                )
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

            elif col == 3:
                vals = xi[~np.isnan(xi)].ravel()
                ax.hist(vals, bins=30, color='#E8855A', edgecolor='white', linewidth=0.3)
                ax.set_xlabel('a(x)', fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
                ax.tick_params(labelsize=7)

            elif col == 4:
                vals = yi[~np.isnan(yi)].ravel()
                ax.hist(vals, bins=30, color='#5A9AE8', edgecolor='white', linewidth=0.3)
                ax.set_xlabel('u*(x)', fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
                ax.tick_params(labelsize=7)

            elif col == 5:
                g_vals = bci[:, 2]
                ax.plot(g_vals, color='#2ecc71', lw=1.5)
                ax.fill_between(range(len(g_vals)), g_vals, alpha=0.2, color='#2ecc71')
                ax.axhline(0, color='gray', lw=0.8, ls='--')
                ax.set_xlabel('Boundary point index', fontsize=8)
                ax.set_ylabel('g value', fontsize=8)
                ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col < 3:
                ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'Sample #{idx}', fontsize=8)

    plt.tight_layout()
    out = save_path or f'viz_{domain_type}_{split}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'\nSaved -> {out}')

    # Global statistics plot
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle(f'{domain_type} — Global Statistics ({N} samples)', fontsize=12, fontweight='bold')

    mean_x = x.mean(0).numpy(); mean_x[nan_mask] = np.nan
    im0 = axes[0].imshow(mean_x, origin='lower', cmap='YlOrRd')
    axes[0].set_title('Mean Coefficient  E[a(x)]')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    mean_y = y.mean(0).numpy(); mean_y[nan_mask] = np.nan
    vabs = np.nanmax(np.abs(mean_y))
    im1 = axes[1].imshow(mean_y, origin='lower', cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    axes[1].set_title('Mean Solution  E[u*(x)]')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    std_y = y.std(0).numpy(); std_y[nan_mask] = np.nan
    im2 = axes[2].imshow(std_y, origin='lower', cmap='viridis')
    axes[2].set_title('Solution Std  Std[u*(x)]')
    axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    out2 = out.replace('.png', '_stats.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    print(f'Saved -> {out2}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='l_shape', choices=['l_shape', 'circle_hole', 'square'])
    p.add_argument('--split',  default='test',    choices=['train', 'test'])
    p.add_argument('--n',      type=int, default=4, help='Number of samples to visualize')
    p.add_argument('--out',    default=None)
    args = p.parse_args()
    visualize(args.domain, args.n, args.split, args.out)
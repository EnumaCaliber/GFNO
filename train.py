import torch
import yaml
import os
import sys
from timeit import default_timer

sys.path.append(os.path.dirname(__file__))

from data.dataset import load_elasticity
from models.boundary_fno import BoundaryFNO, count_params
from utils.losses import LpLoss
from utils.metrics import compute_all_metrics

# Adam from Geo-FNO repo
from Adam import Adam


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(cfg_path='configs/elasticity.yaml'):
    cfg = load_config(cfg_path)

    # flatten config for dataset loader
    data_cfg = {**cfg['data'], **{'batch_size': cfg['train']['batch_size']}}
    train_loader, test_loader = load_elasticity(data_cfg)

    # model
    mcfg  = cfg['model']
    model = BoundaryFNO(
        modes1=mcfg['modes1'],
        modes2=mcfg['modes2'],
        width=mcfg['width'],
        rank=mcfg['rank'],
        n_layers=mcfg['n_layers'],
        padding=mcfg['padding'],
    ).cuda()
    print(f"Parameters: {count_params(model):,}")

    # optimizer
    tcfg      = cfg['train']
    optimizer = Adam(model.parameters(), lr=tcfg['learning_rate'], weight_decay=float(tcfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tcfg['step_size'], gamma=tcfg['gamma'])
    criterion = LpLoss(p=2, reduction='mean')

    os.makedirs(cfg['logging']['model_dir'], exist_ok=True)

    for ep in range(tcfg['epochs']):
        model.train()
        t1 = default_timer()
        train_loss = 0.0

        for coords, sdf, y in train_loader:
            coords, sdf, y = coords.cuda(), sdf.cuda(), y.cuda()
            optimizer.zero_grad()

            pred = model(coords, sdf)
            mask = (sdf > 0).float()
            loss = criterion(pred, y, mask=mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # eval
        model.eval()
        test_loss = 0.0
        all_metrics = {'overall_l2': 0, 'boundary_l2': 0, 'interior_l2': 0}

        with torch.no_grad():
            for coords, sdf, y in test_loader:
                coords, sdf, y = coords.cuda(), sdf.cuda(), y.cuda()
                pred = model(coords, sdf)
                mask = (sdf > 0).float()
                test_loss += criterion(pred, y, mask=mask).item()

                m = compute_all_metrics(pred, y, sdf)
                for k in all_metrics:
                    all_metrics[k] += m[k]

        test_loss /= len(test_loader)
        for k in all_metrics:
            all_metrics[k] /= len(test_loader)

        t2 = default_timer()
        print(f"ep {ep:03d} | {t2-t1:.1f}s | train {train_loss:.5f} | "
              f"test {test_loss:.5f} | "
              f"boundary {all_metrics['boundary_l2']:.5f} | "
              f"interior {all_metrics['interior_l2']:.5f}")

        if ep % cfg['logging']['save_every'] == 0:
            path = os.path.join(cfg['logging']['model_dir'], f'boundary_fno_ep{ep}.pt')
            torch.save(model.state_dict(), path)
            print(f"  Saved: {path}")

    print("\nDone.")


if __name__ == '__main__':
    train()
import torch
import numpy as np


def relative_l2(pred, target):
    """Overall relative L2 error"""
    diff = torch.norm(pred - target, dim=-1)
    norm = torch.norm(target, dim=-1)
    return (diff / (norm + 1e-8)).mean().item()


def boundary_interior_error(pred, target, sdf, threshold=0.05):
    """
    Split error into boundary region and interior region.
    pred, target: (B, s1, s2)
    sdf:          (B, s1, s2) normalized SDF
    threshold:    points with |sdf| < threshold are considered boundary region
    """
    boundary_mask = (sdf.abs() < threshold).float()
    interior_mask = (sdf > threshold).float()

    def masked_rel_l2(mask):
        n = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
        diff = ((pred - target).pow(2) * mask).sum(dim=(1, 2))
        norm = (target.pow(2) * mask).sum(dim=(1, 2))
        return (diff / (norm + 1e-8)).sqrt().mean().item()

    return {
        'boundary_l2': masked_rel_l2(boundary_mask),
        'interior_l2': masked_rel_l2(interior_mask),
    }


def compute_all_metrics(pred, target, sdf, threshold=0.05):
    B = pred.shape[0]
    pred_flat   = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)

    overall = relative_l2(pred_flat, target_flat)
    split   = boundary_interior_error(pred, target, sdf, threshold)

    return {
        'overall_l2':   overall,
        'boundary_l2':  split['boundary_l2'],
        'interior_l2':  split['interior_l2'],
    }
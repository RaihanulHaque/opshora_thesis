from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits) * ~self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives = positive_mask.sum(dim=1).clamp_min(1)
    return -((log_prob * positive_mask).sum(dim=1) / positives).mean()


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    distances = torch.cdist(F.normalize(embeddings, dim=1), F.normalize(embeddings, dim=1))
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = distances.masked_fill(~same | eye, float("-inf")).max(dim=1).values
    negative = distances.masked_fill(same, float("inf")).min(dim=1).values
    valid = torch.isfinite(positive) & torch.isfinite(negative)
    if not valid.any():
        return embeddings.sum() * 0.0
    return F.relu(positive[valid] - negative[valid] + margin).mean()


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lambda_radius: float = 0.5,
) -> torch.Tensor:
    selected = mask[:, :, None, None, None].expand_as(prediction)
    if not selected.any():
        selected = torch.ones_like(prediction, dtype=torch.bool)
    topology_loss = F.binary_cross_entropy_with_logits(prediction[:, :, :1][selected[:, :, :1]], target[:, :, :1][selected[:, :, :1]])
    radius_loss = F.smooth_l1_loss(torch.sigmoid(prediction[:, :, 1:2][selected[:, :, 1:2]]), target[:, :, 1:2][selected[:, :, 1:2]])
    return topology_loss + lambda_radius * radius_loss

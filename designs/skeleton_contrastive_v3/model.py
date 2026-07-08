"""Skeleton Contrastive V3: generative motion learning + metric verification.

The architecture intentionally mirrors the advisor-level flow without using an
unstable adversarial discriminator:

Hamilton skeleton sequence -> spatial encoder -> temporal encoder
-> motion latent + structure latent -> embedding projector
-> masked reconstruction loss + contrastive/triplet metric loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SkeletonFrameEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, output_dim: int = 192):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 160, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(160 * 5, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(frames)
        global_feature = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        horizontal_parts = F.adaptive_avg_pool2d(feature_map, (4, 1)).flatten(1)
        return self.projection(torch.cat((global_feature, horizontal_parts), dim=1))


class SkeletonDecoder(nn.Module):
    def __init__(self, temporal_dim: int, output_channels: int = 2):
        super().__init__()
        self.seed = nn.Linear(temporal_dim, 160 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(160, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(128, 80, 4, stride=2, padding=1),
            nn.BatchNorm2d(80),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(80, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, output_channels, 1),
        )

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        batch, time, channels = temporal.shape
        seed = self.seed(temporal.reshape(batch * time, channels)).view(batch * time, 160, 8, 8)
        decoded = self.decoder(seed)
        return decoded.view(batch, time, decoded.shape[1], decoded.shape[2], decoded.shape[3])


class SkeletonContrastiveV3(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        frame_dim = 192
        self.frame_encoder = SkeletonFrameEncoder(in_channels=3, output_dim=frame_dim)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(frame_dim, frame_dim, 3, padding=1, groups=3),
            nn.BatchNorm1d(frame_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, 3, padding=2, dilation=2, groups=3),
            nn.BatchNorm1d(frame_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal_rnn = nn.GRU(
            frame_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        temporal_dim = hidden_dim * 2
        self.motion_attention = nn.Sequential(nn.Linear(temporal_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.structure_pool = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(temporal_dim + hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.decoder = SkeletonDecoder(temporal_dim, output_channels=2)

    @staticmethod
    def temporal_mask(batch: int, time: int, ratio: float, device: torch.device) -> torch.Tensor:
        count = max(1, round(time * ratio))
        mask = torch.zeros(batch, time, dtype=torch.bool, device=device)
        for row in range(batch):
            mask[row, torch.randperm(time, device=device)[:count]] = True
        return mask

    def forward(self, skeleton, topology, reconstruct: bool = False, mask_ratio: float = 0.0):
        batch, time = skeleton.shape[:2]
        full_input = torch.cat((skeleton, topology), dim=2)
        if reconstruct and mask_ratio > 0:
            masked = self.temporal_mask(batch, time, mask_ratio, skeleton.device)
            keep = (~masked).to(full_input.dtype)[:, :, None, None, None]
            full_input = full_input * keep
        else:
            masked = torch.zeros(batch, time, dtype=torch.bool, device=skeleton.device)

        frame_features = self.frame_encoder(full_input.flatten(0, 1)).view(batch, time, -1)
        temporal_features = self.temporal_cnn(frame_features.transpose(1, 2)).transpose(1, 2)
        temporal, _ = self.temporal_rnn(temporal_features)

        motion_weights = torch.softmax(self.motion_attention(temporal).squeeze(-1), dim=1)
        motion_latent = torch.sum(temporal * motion_weights.unsqueeze(-1), dim=1)
        structure_latent = self.structure_pool(frame_features.mean(dim=1))
        embedding = self.fusion(torch.cat((motion_latent, structure_latent), dim=1))
        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "mask": masked,
            "motion_latent": motion_latent,
            "structure_latent": structure_latent,
        }
        if reconstruct:
            output["reconstruction"] = self.decoder(temporal)
        return output


def build_reconstruction_target(skeleton: torch.Tensor, topology: torch.Tensor) -> torch.Tensor:
    """Reconstruct the Hamilton skeleton and temporal motion field."""
    motion = topology[:, :, 1:2] if topology.shape[2] > 1 else topology[:, :, :1]
    return torch.cat((skeleton, motion), dim=2)


def reconstruction_preview_labels() -> list[str]:
    return [
        "input skeleton",
        "target skeleton",
        "reconstructed skeleton",
        "target motion",
        "reconstructed motion",
    ]


def compute_reconstruction_loss(prediction, target, mask, config):
    selected_frames = mask[:, :, None, None, None].to(prediction.dtype)
    skeleton_logits = prediction[:, :, :1]
    motion_logits = prediction[:, :, 1:2]
    skeleton_target = target[:, :, :1]
    motion_target = target[:, :, 1:2]

    positive_weight = 1.0 + config.topology_positive_weight * skeleton_target
    skeleton_bce = F.binary_cross_entropy_with_logits(skeleton_logits, skeleton_target, reduction="none")
    skeleton_bce = (skeleton_bce * positive_weight * selected_frames).sum() / (
        selected_frames.sum() * skeleton_target.shape[-1] * skeleton_target.shape[-2]
    ).clamp_min(1.0)

    skeleton_prob = torch.sigmoid(skeleton_logits) * selected_frames
    selected_target = skeleton_target * selected_frames
    intersection = (skeleton_prob * selected_target).sum(dim=(1, 2, 3, 4))
    dice = 1.0 - (
        (2.0 * intersection + 1.0)
        / (skeleton_prob.sum(dim=(1, 2, 3, 4)) + selected_target.sum(dim=(1, 2, 3, 4)) + 1.0)
    ).mean()

    motion_error = F.smooth_l1_loss(torch.sigmoid(motion_logits), motion_target, reduction="none")
    motion_loss = (motion_error * selected_frames).sum() / (
        selected_frames.sum() * motion_target.shape[-1] * motion_target.shape[-2]
    ).clamp_min(1.0)
    return skeleton_bce + config.lambda_dice * dice + config.lambda_radius * motion_loss


def build_model(config, num_classes: int) -> SkeletonContrastiveV3:
    del num_classes
    return SkeletonContrastiveV3(
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

"""Skeleton + silhouette fusion V6.

This design fuses silhouette context with Hamilton skeleton motion maps to
recover strict Rank-1 retrieval cues that skeleton-only models lost.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PartFrameEncoder(nn.Module):
    def __init__(self, in_channels: int, output_dim: int = 192, parts: int = 6):
        super().__init__()
        self.parts = parts
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
            nn.Conv2d(96, 144, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(144),
            nn.SiLU(inplace=True),
            nn.Conv2d(144, 176, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(176),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(176 * (parts + 1), output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.08),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(frames)
        global_feature = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        part_feature = F.adaptive_avg_pool2d(feature_map, (self.parts, 1)).flatten(1)
        return self.projection(torch.cat((global_feature, part_feature), dim=1))


class SkeletonDecoder(nn.Module):
    def __init__(self, temporal_dim: int, output_channels: int = 2):
        super().__init__()
        self.seed = nn.Linear(temporal_dim, 144 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(144, 96, 4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 40, 4, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.SiLU(inplace=True),
            nn.Conv2d(40, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, output_channels, 1),
        )

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        batch, time, channels = temporal.shape
        seed = self.seed(temporal.reshape(batch * time, channels)).view(batch * time, 144, 8, 8)
        decoded = self.decoder(seed)
        return decoded.view(batch, time, decoded.shape[1], decoded.shape[2], decoded.shape[3])


class FusionV6(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        frame_dim = 192
        self.silhouette_encoder = PartFrameEncoder(1, frame_dim, parts=6)
        self.skeleton_encoder = PartFrameEncoder(3, frame_dim, parts=6)
        self.gate = nn.Sequential(
            nn.Linear(frame_dim * 2, frame_dim),
            nn.LayerNorm(frame_dim),
            nn.SiLU(inplace=True),
            nn.Linear(frame_dim, frame_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(frame_dim)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(frame_dim, frame_dim, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm1d(frame_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, 3, padding=2, dilation=2, groups=4, bias=False),
            nn.BatchNorm1d(frame_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal_rnn = nn.GRU(
            frame_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.12,
        )
        temporal_dim = hidden_dim * 2
        self.attention = nn.Sequential(nn.Linear(temporal_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        pooled_dim = temporal_dim * 3 + frame_dim
        self.embedding = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.decoder = SkeletonDecoder(temporal_dim, output_channels=2)

    @staticmethod
    def temporal_mask(batch: int, time: int, ratio: float, device: torch.device) -> torch.Tensor:
        count = max(1, round(time * ratio))
        mask = torch.zeros(batch, time, dtype=torch.bool, device=device)
        for row in range(batch):
            mask[row, torch.randperm(time, device=device)[:count]] = True
        return mask

    def forward(self, silhouette, topology, reconstruct: bool = False, mask_ratio: float = 0.0):
        batch, time = silhouette.shape[:2]
        if reconstruct and mask_ratio > 0:
            masked = self.temporal_mask(batch, time, mask_ratio, silhouette.device)
            keep = (~masked).to(silhouette.dtype)[:, :, None, None, None]
            silhouette_input = silhouette * keep
            topology_input = topology * keep
        else:
            masked = torch.zeros(batch, time, dtype=torch.bool, device=silhouette.device)
            silhouette_input = silhouette
            topology_input = topology

        sil = self.silhouette_encoder(silhouette_input.flatten(0, 1)).view(batch, time, -1)
        skel = self.skeleton_encoder(topology_input.flatten(0, 1)).view(batch, time, -1)
        gate = self.gate(torch.cat((sil, skel), dim=-1))
        fused = self.fusion_norm(gate * skel + (1.0 - gate) * sil)
        temporal_features = self.temporal_cnn(fused.transpose(1, 2)).transpose(1, 2)
        temporal, _ = self.temporal_rnn(temporal_features)

        weights = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
        attention_pool = torch.sum(temporal * weights.unsqueeze(-1), dim=1)
        mean_pool = temporal.mean(dim=1)
        max_pool = temporal.max(dim=1).values
        structure_pool = fused.mean(dim=1)
        embedding = self.embedding(torch.cat((attention_pool, mean_pool, max_pool, structure_pool), dim=1))
        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "logits": self.classifier(embedding),
            "mask": masked,
        }
        if reconstruct:
            output["reconstruction"] = self.decoder(temporal)
        return output


def build_reconstruction_target(silhouette: torch.Tensor, topology: torch.Tensor) -> torch.Tensor:
    del silhouette
    skeleton = topology[:, :, :1]
    motion = topology[:, :, 2:3] if topology.shape[2] > 2 else topology[:, :, :1]
    return torch.cat((skeleton, motion), dim=2)


def reconstruction_preview_labels() -> list[str]:
    return [
        "input silhouette",
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


def build_model(config, num_classes: int) -> FusionV6:
    return FusionV6(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

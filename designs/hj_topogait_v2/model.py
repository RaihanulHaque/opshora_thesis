"""HJ-TopoGait v2: part-preserving encoding and sparse-aware reconstruction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PartFrameEncoder(nn.Module):
    """Retain coarse horizontal body-part information instead of global pooling only."""

    def __init__(self, in_channels: int, output_dim: int = 160, parts: int = 4):
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
            nn.Conv2d(96, 128, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * (parts + 1), output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(frames)
        global_feature = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        part_feature = F.adaptive_avg_pool2d(feature_map, (self.parts, 1)).flatten(1)
        return self.projection(torch.cat((global_feature, part_feature), dim=1))


class SharpTopologyDecoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.seed = nn.Linear(input_dim, 128 * 8 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 2, 1),
        )

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        batch, time, channels = temporal.shape
        seed = self.seed(temporal.reshape(batch * time, channels)).view(batch * time, 128, 8, 6)
        decoded = self.decoder(seed)[:, :, :, 2:46]
        return decoded.view(batch, time, 2, 64, 44)


class HJTopoGaitV2(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        frame_dim = 160
        self.silhouette_encoder = PartFrameEncoder(1, frame_dim)
        self.topology_encoder = PartFrameEncoder(3, frame_dim)
        self.gate = nn.Sequential(nn.Linear(frame_dim * 2, frame_dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(frame_dim)
        self.temporal = nn.GRU(
            frame_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        temporal_dim = hidden_dim * 2
        layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=4,
            dim_feedforward=temporal_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_refiner = nn.TransformerEncoder(layer, num_layers=1)
        self.attention = nn.Sequential(nn.Linear(temporal_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.embedding = nn.Sequential(
            nn.Linear(temporal_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.decoder = SharpTopologyDecoder(temporal_dim)

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
            silhouette_input, topology_input = silhouette * keep, topology * keep
        else:
            masked = torch.zeros(batch, time, dtype=torch.bool, device=silhouette.device)
            silhouette_input, topology_input = silhouette, topology

        sil = self.silhouette_encoder(silhouette_input.flatten(0, 1)).view(batch, time, -1)
        topo = self.topology_encoder(topology_input.flatten(0, 1)).view(batch, time, -1)
        gate = self.gate(torch.cat((sil, topo), dim=-1))
        fused = self.fusion_norm(gate * topo + (1.0 - gate) * sil)
        temporal, _ = self.temporal(fused)
        temporal = self.temporal_refiner(temporal)
        weights = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
        pooled = torch.sum(temporal * weights.unsqueeze(-1), dim=1)
        embedding = self.embedding(pooled)
        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "logits": self.classifier(embedding),
            "mask": masked,
        }
        if reconstruct:
            output["reconstruction"] = self.decoder(temporal)
        return output


def compute_reconstruction_loss(prediction, target, mask, config):
    """Weighted BCE + soft Dice, with radius learned mainly along topology."""
    selected_frames = mask[:, :, None, None, None]
    topology_logits = prediction[:, :, :1]
    radius_logits = prediction[:, :, 1:2]
    topology_target = target[:, :, :1]
    radius_target = target[:, :, 1:2]

    frame_weight = selected_frames.to(prediction.dtype)
    positive_weight = 1.0 + config.topology_positive_weight * topology_target
    bce = F.binary_cross_entropy_with_logits(topology_logits, topology_target, reduction="none")
    bce = (bce * positive_weight * frame_weight).sum() / (frame_weight.sum() * topology_target.shape[-1] * topology_target.shape[-2]).clamp_min(1.0)

    probability = torch.sigmoid(topology_logits) * frame_weight
    target_selected = topology_target * frame_weight
    intersection = (probability * target_selected).sum(dim=(1, 2, 3, 4))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probability.sum(dim=(1, 2, 3, 4)) + target_selected.sum(dim=(1, 2, 3, 4)) + 1.0)).mean()

    skeleton_weight = (topology_target > 0.05).to(prediction.dtype) * frame_weight
    radius_error = F.smooth_l1_loss(torch.sigmoid(radius_logits), radius_target, reduction="none")
    radius = (radius_error * skeleton_weight).sum() / skeleton_weight.sum().clamp_min(1.0)
    return bce + config.lambda_dice * dice + config.lambda_radius * radius


def build_model(config, num_classes: int) -> HJTopoGaitV2:
    return HJTopoGaitV2(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

"""Skeleton Rank-1 V4: retrieval-focused skeleton gait model.

This design keeps the V3 generative + contrastive framework but adds a
stronger temporal encoder and an auxiliary normalized identity head to push
Rank-1 retrieval higher. Test-time behavior still uses embeddings/distances.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=max(1, out_channels // 32), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            SqueezeExcitation(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RankFrameEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, output_dim: int = 256, parts: int = 6):
        super().__init__()
        self.parts = parts
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 48),
            ConvBlock(48, 80, stride=2),
            ConvBlock(80, 128, stride=2),
            ConvBlock(128, 192, stride=2),
            ConvBlock(192, 224),
        )
        self.projection = nn.Sequential(
            nn.Linear(224 * (parts + 1), output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(frames)
        global_feature = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        part_feature = F.adaptive_avg_pool2d(feature_map, (self.parts, 1)).flatten(1)
        return self.projection(torch.cat((global_feature, part_feature), dim=1))


class MultiBranchTemporalCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch3 = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.branch5 = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.branch_dilated = nn.Conv1d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x T x C
        transposed = x.transpose(1, 2)
        mixed = self.mix(torch.cat((self.branch3(transposed), self.branch5(transposed), self.branch_dilated(transposed)), dim=1))
        return self.activation((mixed + transposed).transpose(1, 2))


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


class NormalizedClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 24.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(F.normalize(embedding, dim=1), F.normalize(self.weight, dim=1))


class SkeletonRank1V4(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        frame_dim = 256
        self.frame_encoder = RankFrameEncoder(in_channels=3, output_dim=frame_dim, parts=6)
        self.temporal_cnn = MultiBranchTemporalCNN(frame_dim)
        self.temporal_rnn = nn.GRU(
            frame_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.18,
        )
        temporal_dim = hidden_dim * 2
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=8,
            dim_feedforward=temporal_dim * 2,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_refiner = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.motion_attention = nn.Sequential(nn.Linear(temporal_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.structure_pool = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        pooled_dim = temporal_dim * 3 + hidden_dim
        self.embedding = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.classifier = NormalizedClassifier(embedding_dim, num_classes)
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
        temporal_features = self.temporal_cnn(frame_features)
        temporal, _ = self.temporal_rnn(temporal_features)
        temporal = self.temporal_refiner(temporal)

        attention_weights = torch.softmax(self.motion_attention(temporal).squeeze(-1), dim=1)
        attention_pool = torch.sum(temporal * attention_weights.unsqueeze(-1), dim=1)
        mean_pool = temporal.mean(dim=1)
        max_pool = temporal.max(dim=1).values
        structure_latent = self.structure_pool(frame_features.mean(dim=1))
        embedding = self.embedding(torch.cat((attention_pool, mean_pool, max_pool, structure_latent), dim=1))

        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "logits": self.classifier(embedding),
            "mask": masked,
            "motion_latent": attention_pool,
            "structure_latent": structure_latent,
        }
        if reconstruct:
            output["reconstruction"] = self.decoder(temporal)
        return output


def build_reconstruction_target(skeleton: torch.Tensor, topology: torch.Tensor) -> torch.Tensor:
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


def build_model(config, num_classes: int) -> SkeletonRank1V4:
    return SkeletonRank1V4(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

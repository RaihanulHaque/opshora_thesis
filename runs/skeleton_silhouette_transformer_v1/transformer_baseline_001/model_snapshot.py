"""Skeleton + silhouette early-fusion Transformer baseline.

Comparison baseline for `skeleton_silhouette_fusion_v6`. This design tests
plain self-attention temporal modeling instead of V6's GRU + attention/mean/
max multi-pool:

  early channel concatenation -> plain CNN frame encoder -> learned position
  embedding + CLS token -> standard Transformer encoder -> CLS-token
  embedding head.

It shares V6's exact dataset config, cache directory, and loss weights so
the only variable between runs is the model architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FusionFrameEncoder(nn.Module):
    """Plain CNN encoder over the concatenated silhouette+topology frame."""

    def __init__(self, in_channels: int = 4, output_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(frames))


class SkeletonDecoder(nn.Module):
    def __init__(self, temporal_dim: int, output_channels: int = 2):
        super().__init__()
        self.seed = nn.Linear(temporal_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 40, 4, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 1),
        )

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        batch, time, channels = temporal.shape
        seed = self.seed(temporal.reshape(batch * time, channels)).view(batch * time, 128, 8, 8)
        decoded = self.decoder(seed)
        return decoded.view(batch, time, decoded.shape[1], decoded.shape[2], decoded.shape[3])


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        embedding_dim: int,
        projection_dim: int,
        sequence_length: int = 30,
    ):
        super().__init__()
        del hidden_dim  # width is fixed by frame_dim (the transformer d_model)
        frame_dim = 128
        self.encoder = FusionFrameEncoder(in_channels=4, output_dim=frame_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, frame_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length + 1, frame_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=frame_dim,
            nhead=4,
            dim_feedforward=frame_dim * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.embedding = nn.Sequential(
            nn.Linear(frame_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.decoder = SkeletonDecoder(frame_dim, output_channels=2)

    @staticmethod
    def temporal_mask(batch: int, time: int, ratio: float, device: torch.device) -> torch.Tensor:
        count = max(1, round(time * ratio))
        mask = torch.zeros(batch, time, dtype=torch.bool, device=device)
        for row in range(batch):
            mask[row, torch.randperm(time, device=device)[:count]] = True
        return mask

    def forward(self, silhouette, topology, reconstruct: bool = False, mask_ratio: float = 0.0):
        batch, time = silhouette.shape[:2]
        fused_input = torch.cat((silhouette, topology), dim=2)
        if reconstruct and mask_ratio > 0:
            masked = self.temporal_mask(batch, time, mask_ratio, silhouette.device)
            keep = (~masked).to(fused_input.dtype)[:, :, None, None, None]
            fused_input = fused_input * keep
        else:
            masked = torch.zeros(batch, time, dtype=torch.bool, device=silhouette.device)

        frame_features = self.encoder(fused_input.flatten(0, 1)).view(batch, time, -1)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls_tokens, frame_features), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.temporal_transformer(tokens)
        cls_output = encoded[:, 0]
        frame_output = encoded[:, 1:]
        embedding = self.embedding(cls_output)

        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "logits": self.classifier(embedding),
            "mask": masked,
        }
        if reconstruct:
            output["reconstruction"] = self.decoder(frame_output)
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


def build_model(config, num_classes: int) -> TransformerBaseline:
    return TransformerBaseline(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
        sequence_length=config.sequence_length,
    )

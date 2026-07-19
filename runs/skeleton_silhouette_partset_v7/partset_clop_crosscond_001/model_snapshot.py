"""Skeleton + silhouette part-set fusion V7.

V7 keeps V6's training recipe (SupCon + batch-hard triplet + small CE +
masked skeleton reconstruction), dataset cache, and retrieval protocol
completely untouched, and changes only the architecture — so it stays
directly comparable to V6 and the Tier A baselines in
`runs/MODEL_COMPARISON.md`. Three changes, each aimed at a specific V6
weakness:

1. Spatial gated fusion. V6 fused the two streams as per-frame *vectors*,
   after all spatial structure was pooled away. V7 fuses them at
   feature-map level with a per-pixel, per-channel learned gate, so the
   network can trust the skeleton in thin limb regions and the silhouette
   on the body contour within the same frame.
2. Part-set temporal branch. V6 collapsed each frame to one 192-D vector
   *before* any temporal modeling, discarding where on the body a shape
   or motion difference happens. V7 keeps a 16x16 fused feature map,
   pools it into 8 horizontal body-part strips, aggregates each part over
   time with set-max plus a local temporal-conv branch (GaitSet/GaitGL
   style), and embeds each part with its own linear head.
3. The global Bi-GRU branch survives from V6 (it feeds the skeleton
   decoder and captures long-range dynamics) but is now one of two
   pooling paths into the embedding instead of the only one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StreamStem(nn.Module):
    """Light per-stream encoder up to the fusion point (stride 2)."""

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.layers(frames)


class SeparateFC(nn.Module):
    """One independent linear head per horizontal body part."""

    def __init__(self, parts: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(parts, in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bpc,pcd->bpd", part_features, self.weight)


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


class FusionV7(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        stem_dim = 64
        map_dim = 192
        parts = 8
        part_dim = 64
        self.parts = parts
        self.silhouette_stem = StreamStem(1, stem_dim)
        self.skeleton_stem = StreamStem(3, stem_dim)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(stem_dim * 2, stem_dim, 1),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.BatchNorm2d(stem_dim)
        self.trunk = nn.Sequential(
            nn.Conv2d(stem_dim, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 160, 3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, map_dim, 3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(map_dim),
            nn.SiLU(inplace=True),
        )
        self.part_motion = nn.Sequential(
            nn.Conv1d(map_dim, map_dim, 3, padding=1, groups=8, bias=False),
            nn.BatchNorm1d(map_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(map_dim, map_dim, 3, padding=2, dilation=2, groups=8, bias=False),
            nn.BatchNorm1d(map_dim),
            nn.SiLU(inplace=True),
        )
        self.part_dropout = nn.Dropout(0.10)
        self.part_fc = SeparateFC(parts, map_dim, part_dim)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(map_dim, map_dim, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm1d(map_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal_rnn = nn.GRU(
            map_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.12,
        )
        temporal_dim = hidden_dim * 2
        self.attention = nn.Sequential(nn.Linear(temporal_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        pooled_dim = parts * part_dim + temporal_dim * 2
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

        sil_map = self.silhouette_stem(silhouette_input.flatten(0, 1))
        skel_map = self.skeleton_stem(topology_input.flatten(0, 1))
        gate = self.spatial_gate(torch.cat((sil_map, skel_map), dim=1))
        fused = self.fusion_norm(gate * skel_map + (1.0 - gate) * sil_map)
        feature_map = self.trunk(fused)

        strip_avg = F.adaptive_avg_pool2d(feature_map, (self.parts, 1))
        strip_max = F.adaptive_max_pool2d(feature_map, (self.parts, 1))
        part_frames = (strip_avg + strip_max).squeeze(-1)
        channels = part_frames.shape[1]
        part_frames = part_frames.view(batch, time, channels, self.parts)
        set_pool = part_frames.max(dim=1).values.transpose(1, 2)
        motion_input = part_frames.permute(0, 3, 2, 1).reshape(batch * self.parts, channels, time)
        motion_pool = self.part_motion(motion_input).max(dim=2).values.view(batch, self.parts, channels)
        part_features = set_pool + motion_pool
        part_embeddings = self.part_fc(self.part_dropout(part_features)).flatten(1)

        frame_vectors = F.adaptive_avg_pool2d(feature_map, 1).flatten(1).view(batch, time, channels)
        temporal_features = self.temporal_cnn(frame_vectors.transpose(1, 2)).transpose(1, 2)
        temporal, _ = self.temporal_rnn(temporal_features)
        weights = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
        attention_pool = torch.sum(temporal * weights.unsqueeze(-1), dim=1)
        mean_pool = temporal.mean(dim=1)

        embedding = self.embedding(torch.cat((part_embeddings, attention_pool, mean_pool), dim=1))
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


def build_model(config, num_classes: int) -> FusionV7:
    return FusionV7(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

"""Replication of "Context-Sensitive Temporal Feature Learning for Gait
Recognition" (Huang, Zhu, Wang, Wang, Yang, He, Liu, Feng; ICCV 2021, CSTL).

The paper's two contributions:
  1. Multi-Scale Temporal Extraction + Adaptive Temporal Aggregation (ATA):
     part-pooled per-frame features are expanded into frame-level,
     short-term (two serial 1D convs) and long-term (attention-weighted
     temporal pooling) representations; a cross-scale relation-modeling FC
     then predicts a soft gate that adaptively re-weights the three scales
     before they are averaged into a single sequence-level feature `T`
     (their Sec. 3.2-3.3, Eq. 1-5).
  2. Salient Spatial Feature Learning (SSFL): for each of K horizontal body
     parts, a per-frame saliency score picks out the single best-appearing
     frame; the selected per-part features are concatenated into one
     "recombinant frame" `S` that is robust to occlusion/misalignment
     (their Sec. 3.4, Eq. 6-10).
  Final embedding = FC(concat(T, S)); loss = CE + triplet in the paper.

Adaptation notes (documented, not silently approximated):
  - Paper is silhouette-only (CASIA-B/OU-MVLP). We feed only the
    `silhouette` channel into the 4-layer CNN encoder; `topology` is used
    solely as the framework's mandatory reconstruction target (never fed to
    the encoder), matching the paper's own input modality.
  - The cross-scale relation-modeling gate (Eq. 3, `W_T`) is defined by the
    paper per-channel-per-part (shape B x N x 3 x C x K). We compute it at
    a coarser per-frame-per-scale granularity (global-pooled context ->
    3-way sigmoid gate, shape B x N x 3), broadcast back over channels and
    parts. This keeps the relation-modeling FC small while preserving the
    paper's core idea -- adaptively re-weighting frame/short-term/long-term
    scales using cross-scale contextual information -- rather than gating
    every (channel, part) pair independently.
  - K = 8 horizontal parts (matches this repo's other part-based designs
    and is within the paper's own K in {16, 32} exploration in spirit,
    scaled down for our 64x64 -- vs. their 64x44 -- input and much smaller
    training population).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

NUM_PARTS = 8


class FrameEncoder(nn.Module):
    """4-layer 2D CNN over silhouette frames (paper Sec. 3.1: "G -> CNN -> F")."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.layers(frames)


class MultiScaleTemporalExtraction(nn.Module):
    """Produces frame-level, short-term and long-term part-pooled features."""

    def __init__(self, channels: int, num_parts: int = NUM_PARTS):
        super().__init__()
        self.num_parts = num_parts
        self.short_conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.short_conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.long_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
        )

    def part_pool(self, feature: torch.Tensor) -> torch.Tensor:
        # feature: (B*T) x C x H x W -> (B*T) x C x num_parts
        strips = feature.chunk(self.num_parts, dim=2)
        pooled = [strip.amax(dim=(2, 3)) + strip.mean(dim=(2, 3)) for strip in strips]
        return torch.stack(pooled, dim=-1)

    def forward(self, feature_map: torch.Tensor, batch: int, time: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.part_pool(feature_map)  # (B*T) x C x K
        channels = pooled.shape[1]
        frame_level = pooled.view(batch, time, channels, self.num_parts)

        short_input = pooled.view(batch, time, channels, self.num_parts).permute(0, 3, 2, 1)
        short_input = short_input.reshape(batch * self.num_parts, channels, time)
        short = F.relu(self.short_conv1(short_input))
        short = short_input + F.relu(self.short_conv2(short))
        short_term = short.view(batch, self.num_parts, channels, time).permute(0, 3, 2, 1)

        importance = torch.sigmoid(self.long_mlp(frame_level.transpose(2, 3)))  # B x T x K x C
        importance = importance.transpose(2, 3)  # B x T x C x K
        weighted = (frame_level * importance).sum(dim=1)
        normalizer = importance.sum(dim=1).clamp_min(1e-6)
        long_term = (weighted / normalizer).unsqueeze(1).expand(-1, time, -1, -1)

        return frame_level, short_term, long_term


class AdaptiveTemporalAggregation(nn.Module):
    """Cross-scale relation modeling -> soft gate -> weighted sequence feature."""

    def __init__(self, channels: int, num_parts: int = NUM_PARTS):
        super().__init__()
        context_dim = 3
        self.relation = nn.Sequential(
            nn.Linear(context_dim, context_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim * 4, context_dim),
        )
        del channels, num_parts

    def forward(self, frame_level, short_term, long_term):
        cascaded_short = frame_level + short_term
        cascaded_long = frame_level + short_term + long_term

        context = torch.stack(
            [feat.mean(dim=(2, 3)) for feat in (frame_level, cascaded_short, cascaded_long)], dim=-1
        )  # B x T x 3
        gate = torch.sigmoid(self.relation(context))  # B x T x 3
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # B x T x 3 x 1 x 1

        aggregated = (
            frame_level * gate[:, :, 0] + cascaded_short * gate[:, :, 1] + cascaded_long * gate[:, :, 2]
        )
        sequence_feature = aggregated.mean(dim=1)  # B x C x K, sequence-level via temporal averaging
        return sequence_feature


class SalientSpatialFeatureLearning(nn.Module):
    """Per-part saliency scoring -> soft-weighted recombinant frame.

    The paper picks the single most-salient frame per part via a hard
    argmax (their Eq. 9) and supervises the saliency scores through a
    *separate* auxiliary cross-entropy loss on a soft-weighted feature
    (their Eq. 6-7). Our shared training recipe (fixed across every design
    in this repo, for comparability) does not add per-design auxiliary
    losses, so a hard argmax here would leave `saliency` with zero gradient
    from the primary loss and it would never move from random init. We
    therefore use the paper's own soft, normalized saliency weighting
    (Eq. 6) directly as the aggregation weights -- differentiable, and
    functionally the same "recombine the most discriminative parts across
    frames" idea, without the disconnected hard-selection step.
    """

    def __init__(self, channels: int, num_parts: int = NUM_PARTS):
        super().__init__()
        self.saliency = nn.Sequential(
            nn.Linear(channels * 3, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1),
        )
        self.num_parts = num_parts

    def forward(self, frame_level, short_term, long_term):
        joint = torch.cat([frame_level, short_term, long_term], dim=2)  # B x T x 3C x K
        joint = joint.permute(0, 1, 3, 2)  # B x T x K x 3C
        scores = torch.sigmoid(self.saliency(joint).squeeze(-1))  # B x T x K
        weights = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)  # normalized over T (Eq. 6)

        recombinant = (frame_level * weights.unsqueeze(2)).sum(dim=1)  # B x C x K
        return recombinant.flatten(1)  # B x C*K


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


class CSTL(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        channels = 128
        self.encoder = FrameEncoder(in_channels=1)
        self.mste = MultiScaleTemporalExtraction(channels)
        self.ata = AdaptiveTemporalAggregation(channels)
        self.ssfl = SalientSpatialFeatureLearning(channels)
        fused_dim = channels * NUM_PARTS * 2  # T (C x K) concatenated with S (C*K)
        self.embedding = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.decoder = SkeletonDecoder(channels, output_channels=2)

    @staticmethod
    def temporal_mask(batch: int, time: int, ratio: float, device: torch.device) -> torch.Tensor:
        count = max(1, round(time * ratio))
        mask = torch.zeros(batch, time, dtype=torch.bool, device=device)
        for row in range(batch):
            mask[row, torch.randperm(time, device=device)[:count]] = True
        return mask

    def forward(self, silhouette, topology, reconstruct: bool = False, mask_ratio: float = 0.0):
        del topology
        batch, time = silhouette.shape[:2]
        if reconstruct and mask_ratio > 0:
            masked = self.temporal_mask(batch, time, mask_ratio, silhouette.device)
            keep = (~masked).to(silhouette.dtype)[:, :, None, None, None]
            frames = silhouette * keep
        else:
            masked = torch.zeros(batch, time, dtype=torch.bool, device=silhouette.device)
            frames = silhouette

        feature_map = self.encoder(frames.flatten(0, 1))  # (B*T) x C x H' x W'
        frame_level, short_term, long_term = self.mste(feature_map, batch, time)
        sequence_feature = self.ata(frame_level, short_term, long_term)  # B x C x K
        recombinant = self.ssfl(frame_level, short_term, long_term)  # B x C*K

        fused = torch.cat([sequence_feature.flatten(1), recombinant], dim=1)
        embedding = self.embedding(fused)

        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "logits": self.classifier(embedding),
            "mask": masked,
        }
        if reconstruct:
            per_frame_code = frame_level.mean(dim=-1)  # B x T x C
            output["reconstruction"] = self.decoder(per_frame_code)
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


def build_model(config, num_classes: int) -> CSTL:
    return CSTL(
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

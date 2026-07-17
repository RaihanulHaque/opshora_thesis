"""Replication of the silhouette branch of "Gait Recognition in the Wild with
Dense 3D Representations and A Benchmark" (Zheng, Liu, Liu, He, Yan, Mei;
CVPR 2022, SMPLGait).

The full SMPLGait framework has two branches: a Silhouette Learning Network
(SLN, a GaitSet-style CNN) and a 3D Spatial-Transformation Network (3D-STN)
that consumes per-frame SMPL body-mesh parameters (pose/shape/viewpoint) to
align the silhouette features across viewpoints. It is trained on their
Gait3D dataset, which ships SMPL fits recovered from video via a 3D human
mesh recovery model (ROMP).

Adaptation notes (documented, not silently approximated):
  - **Neither CASIA-B nor CLoP-Gait provides SMPL parameters** (no 3D mesh
    recovery was run on either dataset), so the 3D-STN branch cannot be
    replicated here -- there is no 3D input to feed it. We therefore
    implement exactly the ablation the original paper itself reports as
    **"SMPLGait w/o 3D"** (their Table 2), i.e. the SLN branch alone with
    the 3D-STN branch and the matrix-multiplication feature alignment
    removed. This is not an ad hoc simplification: it is a configuration
    the authors defined, trained, and reported numbers for, specifically to
    isolate the silhouette branch's own contribution.
  - SLN: a 6-layer 2D CNN (paper: "similar to the backbone of GaitSet"),
    with Set Pooling (max over time, at the raw feature-map resolution,
    before spatial pooling -- GaitSet's defining operation) followed by
    Horizontal Pyramid Pooling (HPP) at stripe scales {1, 2, 4, 8} (GaitSet/
    GLN-style; the paper's Fig. 2 shows "SP -> HPP -> Feature Aggregation"
    without stating exact stripe scales, so we use the standard GaitSet/GLN
    convention also used by `3DLocal`'s and `CSTL`'s reference papers).
  - Input is silhouette-only, matching both the paper's SLN and the
    SMPLGait-w/o-3D ablation; `topology` is used only as the framework's
    mandatory reconstruction target, never fed to the encoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

STRIPE_SCALES = (1, 2, 4, 8)


class SilhouetteLearningNetwork(nn.Module):
    """GaitSet-style 6-layer CNN backbone (paper Sec. 3.2, "SLN")."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.layers(frames)


class HorizontalPyramidPooling(nn.Module):
    """Multi-scale horizontal stripe pooling + per-stripe separate FC."""

    def __init__(self, channels: int, fc_dim: int, scales: tuple[int, ...] = STRIPE_SCALES):
        super().__init__()
        self.scales = scales
        self.num_stripes = sum(scales)
        self.weight = nn.Parameter(torch.empty(self.num_stripes, channels * 2, fc_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # feature_map: B x C x H x W (already set-pooled over time)
        batch, channels, height, _ = feature_map.shape
        stripes = []
        for scale in self.scales:
            for chunk in feature_map.chunk(scale, dim=2):
                avg = chunk.mean(dim=(2, 3))
                mx = chunk.amax(dim=(2, 3))
                stripes.append(torch.cat((avg, mx), dim=1))
        stacked = torch.stack(stripes, dim=1)  # B x num_stripes x 2C
        projected = torch.einsum("bsc,scd->bsd", stacked, self.weight)  # B x num_stripes x fc_dim
        del height, batch
        return projected.flatten(1)


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


class SMPLGaitWithout3D(nn.Module):
    """The "SMPLGait w/o 3D" ablation: SLN + Set Pooling + HPP, no 3D-STN."""

    def __init__(self, num_classes: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        channels = 128
        stripe_fc_dim = 48
        self.sln = SilhouetteLearningNetwork(in_channels=1)
        self.hpp = HorizontalPyramidPooling(channels, stripe_fc_dim)
        hpp_dim = self.hpp.num_stripes * stripe_fc_dim
        self.embedding = nn.Sequential(
            nn.Linear(hpp_dim, embedding_dim),
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

        feature_map = self.sln(frames.flatten(0, 1))
        _, channels, height, width = feature_map.shape
        feature_map = feature_map.view(batch, time, channels, height, width)

        set_pooled = feature_map.amax(dim=1)  # Set Pooling: max over time (GaitSet-style)
        hpp_feature = self.hpp(set_pooled)
        embedding = self.embedding(hpp_feature)

        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "logits": self.classifier(embedding),
            "mask": masked,
        }
        if reconstruct:
            per_frame_code = feature_map.mean(dim=(3, 4))  # B x T x C
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


def build_model(config, num_classes: int) -> SMPLGaitWithout3D:
    return SMPLGaitWithout3D(
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

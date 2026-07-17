"""Replication of "3D Local Convolutional Neural Networks for Gait Recognition"
(Huang, Xue, Shen, Tian, Li, Huang, Hua; ICCV 2021).

The paper's core contribution is the **3D local block**: for each of six
predefined body parts (head, left-arm, right-arm, torso, left-leg, right-leg),
a small localization network predicts an adaptive, per-frame 3D sampling
region (center + scale), which is used to extract a spatially/temporally
localized feature via Gaussian spatial filters and trilinear temporal
filters (their Eq. 7-10). Local features are fused with the global CNN
backbone by channel-concatenation + 1x1x1 conv after every backbone block.

Adaptation notes (documented, not silently approximated):
  - Paper input is silhouette-only (CASIA-B/OU-MVLP). We feed only the
    `silhouette` channel from the shared 4-channel cache into the backbone,
    exactly matching the paper's input modality. The `topology` channel is
    used only as an auxiliary reconstruction *target* for the framework's
    mandatory masked-reconstruction step (see module docstring convention
    established by the other `skeleton_silhouette_*` baselines) -- it is
    never fed into the encoder, so it plays no role in the paper's own
    formulation.
  - We implement the **Gaussian-only sampling variant** of their local
    operation (Sec. 3.2.2): a per-frame, per-part 2D Gaussian gate over the
    full backbone feature map, with learned center offset and log-scale
    predicted by a small temporal-aware localization head (their Eq. 3-8).
    We skip the explicit trilinear resampling onto a smaller M x N x L grid
    (their Eq. 9-10) and instead apply the Gaussian gate multiplicatively at
    the backbone's native resolution, which keeps tensor shapes trivially
    compatible with their declared channel-concatenation fusion step. The
    paper's own ablation (Table 4) shows the Gaussian-only variant performs
    within 0.1-1.3 points of their full Gaussian+Trilinear+Mixture sampling
    (Mean 96.2/93.3/82.4 vs 97.5/94.3/83.7 rank-1 on CASIA-B NM/BG/CL), so
    this is a faithful, paper-endorsed simplification rather than an
    arbitrary one.
  - Backbone follows their Sec. 3.3 / Table 1 setting "f": three
    convolutional blocks, 3D-local operation inserted after every block
    (their best-performing configuration).
  - Part priors (normalized center, in [0, 1] over height x width) are
    standard body proportions: head (0.10, 0.5), torso (0.35, 0.5),
    left/right arm (0.35, 0.25 / 0.75), left/right leg (0.75, 0.35 / 0.65).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

PART_PRIORS: dict[str, tuple[float, float]] = {
    "head": (0.10, 0.50),
    "torso": (0.35, 0.50),
    "left_arm": (0.35, 0.25),
    "right_arm": (0.35, 0.75),
    "left_leg": (0.75, 0.35),
    "right_leg": (0.75, 0.65),
}


class LocalizationHead(nn.Module):
    """Predicts per-frame (dy, dx, log_sigma) offsets for one body part.

    Mirrors the paper's localization module (Sec. 3.2.1): a temporal-aware
    head that consumes the global feature map and emits real-valued offsets
    and an isotropic Gaussian scale, gradients from which do not flow back
    into a separate "global-only" path (there is no such path here, since
    our global backbone already carries the gradient through the gate).
    """

    def __init__(self, channels: int, prior: tuple[float, float]):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))
        self.context = nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1)
        self.head = nn.Linear(channels // 2, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pooled: B x T x C (spatially global-average-pooled backbone features)
        context = F.relu(self.context(pooled.transpose(1, 2))).transpose(1, 2)
        raw = self.head(context)  # B x T x 3
        offset = torch.tanh(raw[..., :2]) * 0.15
        center = self.prior.to(pooled.device) + offset  # B x T x 2, in [0, 1]
        sigma = F.softplus(raw[..., 2]) + 0.08  # B x T
        return center, sigma


class LocalPart(nn.Module):
    """One of the six 3D-local part paths within a single backbone block."""

    def __init__(self, channels: int, prior: tuple[float, float]):
        super().__init__()
        self.localization = LocalizationHead(channels, prior)
        self.extract = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_map: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        # feature_map: B x T x C x H x W, pooled: B x T x C
        batch, time, channels, height, width = feature_map.shape
        center, sigma = self.localization(pooled)  # B x T x 2, B x T
        ys = torch.linspace(0.0, 1.0, height, device=feature_map.device).view(1, 1, height, 1)
        xs = torch.linspace(0.0, 1.0, width, device=feature_map.device).view(1, 1, 1, width)
        cy = center[..., 0].view(batch, time, 1, 1)
        cx = center[..., 1].view(batch, time, 1, 1)
        sig = sigma.view(batch, time, 1, 1).clamp_min(1e-3)
        gate = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig**2))  # B x T x H x W
        gated = feature_map * gate.unsqueeze(2)
        return self.extract(gated.flatten(0, 1)).view(batch, time, channels // 2, height, width)


class LocalCNNBlock(nn.Module):
    """One backbone conv block fused with its six 3D-local part paths."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else None
        self.parts = nn.ModuleDict(
            {name: LocalPart(out_channels, prior) for name, prior in PART_PRIORS.items()}
        )
        fused_channels = out_channels + len(PART_PRIORS) * (out_channels // 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: B x T x C_in x H x W
        batch, time = frames.shape[:2]
        global_feature = self.conv(frames.flatten(0, 1))
        if self.pool is not None:
            global_feature = self.pool(global_feature)
        _, out_channels, height, width = global_feature.shape
        global_feature = global_feature.view(batch, time, out_channels, height, width)
        pooled = global_feature.mean(dim=(3, 4))  # B x T x C, for localization context

        local_outputs = [part(global_feature, pooled) for part in self.parts.values()]
        fused = torch.cat([global_feature, *local_outputs], dim=2)
        fused = self.fusion(fused.flatten(0, 1)).view(batch, time, out_channels, height, width)
        return fused


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


class ThreeDLocalCNN(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, projection_dim: int):
        super().__init__()
        self.block1 = LocalCNNBlock(1, 32, pool=True)
        self.block2 = LocalCNNBlock(32, 64, pool=True)
        self.block3 = LocalCNNBlock(64, 128, pool=False)
        pooled_dim = 128 * 2  # avg + max spatial pooling, then max over time
        self.embedding = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim),
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
        self.decoder = SkeletonDecoder(128, output_channels=2)

    @staticmethod
    def temporal_mask(batch: int, time: int, ratio: float, device: torch.device) -> torch.Tensor:
        count = max(1, round(time * ratio))
        mask = torch.zeros(batch, time, dtype=torch.bool, device=device)
        for row in range(batch):
            mask[row, torch.randperm(time, device=device)[:count]] = True
        return mask

    def forward(self, silhouette, topology, reconstruct: bool = False, mask_ratio: float = 0.0):
        del topology  # paper is silhouette-only; topology is decoder-target only
        batch, time = silhouette.shape[:2]
        if reconstruct and mask_ratio > 0:
            masked = self.temporal_mask(batch, time, mask_ratio, silhouette.device)
            keep = (~masked).to(silhouette.dtype)[:, :, None, None, None]
            frames = silhouette * keep
        else:
            masked = torch.zeros(batch, time, dtype=torch.bool, device=silhouette.device)
            frames = silhouette

        feature = self.block1(frames)
        feature = self.block2(feature)
        feature = self.block3(feature)  # B x T x 128 x H x W

        avg_pool = feature.mean(dim=(3, 4))
        max_pool = feature.amax(dim=(3, 4))
        per_frame = torch.cat((avg_pool, max_pool), dim=2)  # B x T x 256
        pooled = per_frame.max(dim=1).values  # temporal max pooling (GaitPart-style)
        embedding = self.embedding(pooled)

        output = {
            "embedding": embedding,
            "projection": F.normalize(self.projector(embedding), dim=1),
            "logits": self.classifier(embedding),
            "mask": masked,
        }
        if reconstruct:
            per_frame_code = feature.mean(dim=(3, 4))  # B x T x 128
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


def build_model(config, num_classes: int) -> ThreeDLocalCNN:
    return ThreeDLocalCNN(
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

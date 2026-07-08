from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(96, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(self.network(frames).flatten(1))


class TopologyDecoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.seed = nn.Linear(input_dim, 128 * 4 * 3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        batch, time, channels = sequence_features.shape
        seeds = self.seed(sequence_features.reshape(batch * time, channels)).view(batch * time, 128, 4, 3)
        decoded = self.decoder(seeds)[:, :, :, 2:46]
        return decoded.view(batch, time, 2, 64, 44)


class HJTopoGait(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        projection_dim: int = 128,
    ):
        super().__init__()
        feature_dim = 128
        self.silhouette_encoder = FrameEncoder(1, feature_dim)
        self.topology_encoder = FrameEncoder(3, feature_dim)
        self.gate = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.Sigmoid())
        self.temporal = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        temporal_dim = hidden_dim * 2
        self.attention = nn.Linear(temporal_dim, 1)
        self.embedding = nn.Sequential(
            nn.Linear(temporal_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.decoder = TopologyDecoder(temporal_dim)

    @staticmethod
    def temporal_mask(batch: int, time: int, ratio: float, device: torch.device) -> torch.Tensor:
        count = max(1, round(time * ratio))
        mask = torch.zeros(batch, time, dtype=torch.bool, device=device)
        for row in range(batch):
            mask[row, torch.randperm(time, device=device)[:count]] = True
        return mask

    def forward(
        self,
        silhouette: torch.Tensor,
        topology: torch.Tensor,
        reconstruct: bool = False,
        mask_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        batch, time = silhouette.shape[:2]
        if reconstruct and mask_ratio > 0:
            masked = self.temporal_mask(batch, time, mask_ratio, silhouette.device)
            keep = (~masked).to(silhouette.dtype)[:, :, None, None, None]
            silhouette_input = silhouette * keep
            topology_input = topology * keep
        else:
            masked = torch.zeros(batch, time, dtype=torch.bool, device=silhouette.device)
            silhouette_input, topology_input = silhouette, topology

        sil = self.silhouette_encoder(silhouette_input.flatten(0, 1)).view(batch, time, -1)
        topo = self.topology_encoder(topology_input.flatten(0, 1)).view(batch, time, -1)
        gate = self.gate(torch.cat((sil, topo), dim=-1))
        fused = gate * topo + (1.0 - gate) * sil
        temporal, _ = self.temporal(fused)
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

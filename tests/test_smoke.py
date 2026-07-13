import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from gait.config import ExperimentConfig
from gait.losses import batch_hard_triplet_loss, masked_reconstruction_loss, supervised_contrastive_loss
from gait.model import HJTopoGait
from gait.preprocessing import (
    RawSequence,
    hamilton_jacobi_topology,
    iter_archive_sequences,
    process_sequence,
    process_skeleton_sequence,
)
from gait.train import save_reconstruction_preview


def test_preprocessing_and_model_shapes(tmp_path: Path) -> None:
    frames = []
    for offset in range(35):
        image = np.zeros((80, 60), np.uint8)
        cv2.ellipse(image, (30 + offset % 3, 40), (10, 25), 0, 0, 360, 255, -1)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        frames.append(encoded.tobytes())
    config = ExperimentConfig(height=64, width=44, sequence_length=30)
    maps = process_sequence(RawSequence("001", "fn00", "single", "001/fn00", frames), config)
    assert maps.shape == (30, 4, 64, 44)
    topology, radius, flux = hamilton_jacobi_topology(maps[0, 0] > 0)
    assert topology.max() > 0 and radius.max() > 0 and flux.shape == topology.shape

    model = HJTopoGait(num_classes=2, hidden_dim=32, embedding_dim=64, projection_dim=32)
    silhouette = torch.from_numpy(maps[None, :, :1].astype(np.float32) / 255)
    topo = torch.from_numpy(maps[None, :, 1:].astype(np.float32) / 255)
    model.eval()
    with torch.no_grad():
        output = model(silhouette, topo, reconstruct=True, mask_ratio=0.4)
    assert output["embedding"].shape == (1, 64)
    assert output["reconstruction"].shape == (1, 30, 2, 64, 44)


def test_nested_zip_reader(tmp_path: Path) -> None:
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        for frame in range(3):
            image = np.zeros((20, 10), np.uint8)
            image[3:18, 3:8] = 255
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            inner.writestr(f"001/fn00/{frame:03d}.png", encoded.tobytes())
    outer_path = tmp_path / "silhouette-C.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr("silhouettes/001.zip", inner_buffer.getvalue())
    sequences = list(iter_archive_sequences(outer_path))
    assert len(sequences) == 1
    assert sequences[0].subject == "001"
    assert sequences[0].condition == "fn00"


def test_losses_are_finite() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    features = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    assert torch.isfinite(supervised_contrastive_loss(features, labels))
    assert torch.isfinite(batch_hard_triplet_loss(features, labels))
    prediction = torch.randn(4, 3, 2, 8, 8)
    target = torch.rand_like(prediction)
    mask = torch.ones(4, 3, dtype=torch.bool)
    assert torch.isfinite(masked_reconstruction_loss(prediction, target, mask))


def test_skeleton_v3_shapes() -> None:
    from designs.skeleton_contrastive_v3.model import (
        build_model,
        build_reconstruction_target,
        compute_reconstruction_loss,
    )

    frames = []
    for offset in range(36):
        image = np.zeros((64, 64), np.uint8)
        cv2.line(image, (32, 8), (32 + offset % 4, 34), 255, 2)
        cv2.line(image, (32, 34), (20 + offset % 5, 56), 255, 2)
        cv2.line(image, (32, 34), (44 - offset % 5, 56), 255, 2)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        frames.append(encoded.tobytes())

    config = ExperimentConfig(
        dataset_format="skeleton_hamilton",
        height=64,
        width=64,
        sequence_length=30,
        hidden_dim=32,
        embedding_dim=64,
        projection_dim=32,
        lambda_ce=0.0,
    )
    maps = process_skeleton_sequence(RawSequence("001", "nm-01", "000", "001/nm-01/000", frames), config)
    assert maps.shape == (30, 3, 64, 64)
    skeleton = torch.from_numpy(maps[None, :, :1].astype(np.float32) / 255)
    topology = torch.from_numpy(maps[None, :, 1:].astype(np.float32) / 255)
    model = build_model(config, num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(skeleton, topology, reconstruct=True, mask_ratio=0.4)
    target = build_reconstruction_target(skeleton, topology)
    loss = compute_reconstruction_loss(output["reconstruction"], target, output["mask"], config)
    assert output["embedding"].shape == (1, 64)
    assert output["projection"].shape == (1, 32)
    assert output["reconstruction"].shape == (1, 30, 2, 64, 64)
    assert torch.isfinite(loss)


def test_fusion_baseline_designs_shapes() -> None:
    import importlib

    config = ExperimentConfig(
        dataset_format="skeleton_silhouette_fusion",
        height=64,
        width=64,
        sequence_length=30,
        hidden_dim=32,
        embedding_dim=64,
        projection_dim=32,
        lambda_ce=1.0,
    )
    silhouette = torch.rand(2, 30, 1, 64, 64)
    topology = torch.rand(2, 30, 3, 64, 64)

    for design in (
        "skeleton_silhouette_lstm_v1",
        "skeleton_silhouette_tcn_v1",
        "skeleton_silhouette_transformer_v1",
    ):
        module = importlib.import_module(f"designs.{design}.model")
        model = module.build_model(config, num_classes=5)
        model.eval()
        with torch.no_grad():
            output = model(silhouette, topology, reconstruct=True, mask_ratio=0.3)
        assert output["embedding"].shape == (2, 64)
        assert output["projection"].shape == (2, 32)
        assert output["logits"].shape == (2, 5)
        assert output["reconstruction"].shape == (2, 30, 2, 64, 64)
        target = module.build_reconstruction_target(silhouette, topology)
        loss = module.compute_reconstruction_loss(output["reconstruction"], target, output["mask"], config)
        assert torch.isfinite(loss)


def test_reconstruction_preview(tmp_path: Path) -> None:
    silhouette = torch.rand(1, 3, 1, 64, 44)
    topology = torch.rand(1, 3, 3, 64, 44)
    prediction = torch.randn(1, 3, 2, 64, 44)
    mask = torch.tensor([[False, True, False]])
    output = tmp_path / "reconstruction.png"
    save_reconstruction_preview(silhouette, topology, prediction, mask, output)
    assert output.exists() and output.stat().st_size > 0

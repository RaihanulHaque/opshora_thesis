from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

import cv2
import numpy as np
from scipy import ndimage

from .config import ExperimentConfig


def topology_preserving_thinning(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen two-subiteration thinning for a binary 2-D mask."""
    image = mask.astype(np.uint8).copy()
    while True:
        changed = False
        for first_pass in (True, False):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )
            if first_pass:
                condition_a = (p2 * p4 * p6) == 0
                condition_b = (p4 * p6 * p8) == 0
            else:
                condition_a = (p2 * p4 * p8) == 0
                condition_b = (p2 * p6 * p8) == 0
            remove = (
                (image == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & condition_a
                & condition_b
            )
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image.astype(bool)


@dataclass(slots=True)
class RawSequence:
    subject: str
    condition: str
    view: str
    key: str
    frames: list[bytes]


def _numeric_sort_key(path: str) -> tuple:
    return tuple(int(token) if token.isdigit() else token for token in re.split(r"(\d+)", path))


def _sequence_metadata(image_path: str) -> tuple[str, str, str, str]:
    parts = [part for part in PurePosixPath(image_path).parts if part not in {".", "__MACOSX"}]
    numeric_subjects = [part for part in parts[:-1] if re.fullmatch(r"\d{3}", part)]
    if not numeric_subjects:
        raise ValueError(f"Cannot find a three-digit subject in {image_path!r}")
    subject = numeric_subjects[0]
    subject_index = parts.index(subject)
    tail = parts[subject_index + 1 : -1]
    condition = tail[0] if tail else "unknown"
    view = tail[1] if len(tail) > 1 and re.fullmatch(r"\d{3}", tail[1]) else "single"
    key = "/".join(parts[:-1])
    return subject, condition, view, key


def _looks_like_frame(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))


def _sequences_from_zip_bytes(data: bytes) -> Iterator[RawSequence]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        yield from _sequences_from_zip_archive(archive)


def _sequences_from_zip_archive(archive: zipfile.ZipFile) -> Iterator[RawSequence]:
    groups: dict[str, list[str]] = defaultdict(list)
    metadata: dict[str, tuple[str, str, str]] = {}
    for name in archive.namelist():
        if not _looks_like_frame(name):
            continue
        subject, condition, view, key = _sequence_metadata(name)
        groups[key].append(name)
        metadata[key] = (subject, condition, view)
    for key in sorted(groups):
        names = sorted(groups[key], key=_numeric_sort_key)
        subject, condition, view = metadata[key]
        yield RawSequence(
            subject=subject,
            condition=condition,
            view=view,
            key=key,
            frames=[archive.read(name) for name in names],
        )


def _zip_debug_summary(archive: zipfile.ZipFile, limit: int = 8) -> str:
    names = archive.namelist()
    frame_names = [name for name in names if _looks_like_frame(name)]
    nested = [name for name in names if name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))]
    sample = names[:limit]
    return (
        f"entries={len(names)}, image_entries={len(frame_names)}, "
        f"nested_archives={len(nested)}, sample_entries={sample}"
    )


def _iter_nested_archive_bytes(name: str, data: bytes) -> Iterator[RawSequence]:
    lower = name.lower()
    if lower.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(data)):
        yield from _sequences_from_zip_bytes(data)
        return
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as archive:
            members = [member for member in archive.getmembers() if member.isfile() and _looks_like_frame(member.name)]
            groups: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
            metadata: dict[str, tuple[str, str, str]] = {}
            for member in members:
                subject, condition, view, key = _sequence_metadata(member.name)
                groups[key].append(member)
                metadata[key] = (subject, condition, view)
            for key in sorted(groups):
                selected = sorted(groups[key], key=lambda member: _numeric_sort_key(member.name))
                subject, condition, view = metadata[key]
                frames: list[bytes] = []
                for member in selected:
                    handle = archive.extractfile(member)
                    if handle is not None:
                        frames.append(handle.read())
                yield RawSequence(subject=subject, condition=condition, view=view, key=key, frames=frames)
        return
    # Some Modal uploads lose extensions. Try ZIP last by magic bytes.
    if data[:4] == b"PK\x03\x04":
        yield from _sequences_from_zip_bytes(data)
        return


def _sequences_from_directory(source: Path) -> Iterator[RawSequence]:
    groups: dict[str, list[Path]] = defaultdict(list)
    metadata: dict[str, tuple[str, str, str]] = {}
    for path in source.rglob("*"):
        if not path.is_file() or not _looks_like_frame(path.name):
            continue
        relative = path.relative_to(source).as_posix()
        subject, condition, view, key = _sequence_metadata(relative)
        groups[key].append(path)
        metadata[key] = (subject, condition, view)
    for key in sorted(groups):
        paths = sorted(groups[key], key=lambda item: _numeric_sort_key(item.as_posix()))
        subject, condition, view = metadata[key]
        yield RawSequence(
            subject=subject,
            condition=condition,
            view=view,
            key=key,
            frames=[path.read_bytes() for path in paths],
        )


def _sequences_from_tar(source: Path) -> Iterator[RawSequence]:
    mode = "r:gz" if source.name.endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(source, mode) as archive:
        members = [member for member in archive.getmembers() if member.isfile() and _looks_like_frame(member.name)]
        groups: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
        metadata: dict[str, tuple[str, str, str]] = {}
        for member in members:
            subject, condition, view, key = _sequence_metadata(member.name)
            groups[key].append(member)
            metadata[key] = (subject, condition, view)
        for key in sorted(groups):
            selected = sorted(groups[key], key=lambda member: _numeric_sort_key(member.name))
            subject, condition, view = metadata[key]
            frames: list[bytes] = []
            for member in selected:
                handle = archive.extractfile(member)
                if handle is not None:
                    frames.append(handle.read())
            yield RawSequence(subject=subject, condition=condition, view=view, key=key, frames=frames)


def _resolve_dataset_source(source: str | Path) -> Path:
    source = Path(source)
    if source.exists():
        return source
    if source.is_absolute() and source.parts[:2] == ("/", "data"):
        roots = [Path("/data")]
    else:
        roots = [Path.cwd(), Path.cwd() / "datasets", Path("/data")]
    names = [source.name]
    if source.name.lower().endswith(".zip"):
        names.append(source.stem)
    else:
        names.append(source.name + ".zip")
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
        for pattern in ("*Hamilton*Skeleton*.zip", "*hamilton*skeleton*.zip", "*silh*.zip", "*silhouette*.zip"):
            matches = sorted(root.rglob(pattern)) if root.exists() else []
            if matches:
                return matches[0]
    return source


def iter_archive_sequences(source: str | Path) -> Iterator[RawSequence]:
    """Read a directory, a direct ZIP, or an outer ZIP containing subject ZIPs."""
    source = _resolve_dataset_source(source)
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Dataset archive not found: {source}")
    if source.is_dir():
        has_images = any(path.is_file() and _looks_like_frame(path.name) for path in source.rglob("*"))
        yielded = False
        if has_images:
            for sequence in _sequences_from_directory(source):
                yielded = True
                yield sequence
        nested_archives = sorted(
            [
                path
                for path in source.rglob("*")
                if path.is_file() and path.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
            ],
            key=lambda item: _numeric_sort_key(item.as_posix()),
        )
        if nested_archives:
            for nested in nested_archives:
                for sequence in iter_archive_sequences(nested):
                    yielded = True
                    yield sequence
        if yielded:
            return
        raise RuntimeError(f"No image sequences or nested archives found in {source}")
    if source.name.endswith((".tar", ".tar.gz", ".tgz")):
        yield from _sequences_from_tar(source)
        return
    with zipfile.ZipFile(source) as outer:
        yielded = False
        nested = sorted(
            (
                name
                for name in outer.namelist()
                if name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
            ),
            key=_numeric_sort_key,
        )
        if nested:
            for name in nested:
                for sequence in _iter_nested_archive_bytes(name, outer.read(name)):
                    yielded = True
                    yield sequence
        else:
            for sequence in _sequences_from_zip_archive(outer):
                yielded = True
                yield sequence
        if not yielded:
            print(f"Warning: no sequences found while reading {source}. ZIP summary: {_zip_debug_summary(outer)}", flush=True)


def _sequence_lookup_key(sequence: RawSequence) -> tuple[str, str, str]:
    return sequence.subject, sequence.condition.lower(), sequence.view


def build_sequence_lookup(source: str | Path) -> dict[tuple[str, str, str], RawSequence]:
    lookup: dict[tuple[str, str, str], RawSequence] = {}
    for sequence in iter_archive_sequences(source):
        lookup[_sequence_lookup_key(sequence)] = sequence
    return lookup


def decode_mask(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("OpenCV could not decode an image frame")
    return image > 0


def clean_silhouette_mask(mask: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Morphological close + keep-largest-components + fill holes.

    Same-shape output, not yet cropped/resized -- split out of clean_and_align
    so callers that want a different crop/margin policy (e.g. segment.py's
    natural-margin storage crop) can reuse this cleanup step without
    duplicating it.
    """
    mask_u8 = mask.astype(np.uint8)
    kernel_size = max(1, config.morphology_kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask_u8, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep = np.flatnonzero(areas >= max(2, largest * config.min_component_ratio)) + 1
    mask_u8 = np.isin(labels, keep)
    return ndimage.binary_fill_holes(mask_u8)


@dataclass(frozen=True, slots=True)
class CropTransform:
    """Bbox + scale + placement derived from one mask, reusable to co-align
    a second image that shares the first mask's native pixel grid (e.g. a
    skeleton frame rendered on the same canvas as its paired silhouette)."""

    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    new_h: int
    new_w: int
    y0: int
    x0: int


def compute_crop_transform(mask: np.ndarray, height: int, width: int, margin: int = 2) -> CropTransform | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y_start, y_stop = int(ys.min()), int(ys.max()) + 1
    x_start, x_stop = int(xs.min()), int(xs.max()) + 1
    available_h = height - 2 * margin
    available_w = width - 2 * margin
    scale = min(available_h / (y_stop - y_start), available_w / (x_stop - x_start))
    new_h = max(1, round((y_stop - y_start) * scale))
    new_w = max(1, round((x_stop - x_start) * scale))
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    return CropTransform(y_start, y_stop, x_start, x_stop, new_h, new_w, y0, x0)


def apply_crop_transform(
    mask: np.ndarray,
    transform: CropTransform,
    height: int,
    width: int,
    resize_interpolation: int = cv2.INTER_NEAREST,
    any_coverage: bool = False,
) -> np.ndarray:
    """Render `mask` (same native pixel grid as whatever mask produced `transform`) onto a height x width canvas.

    any_coverage=True marks an output pixel as set if the source region has
    ANY nonzero coverage (rather than requiring >50%), which is what thin,
    1px-wide structures like a skeleton need to survive a large downscale --
    exact-point NEAREST sampling or a majority-vote threshold both tend to
    drop most of a 1px line's pixels entirely.
    """
    cropped = mask[transform.y_start : transform.y_stop, transform.x_start : transform.x_stop].astype(np.uint8)
    canvas = np.zeros((height, width), dtype=np.uint8)
    if any_coverage:
        soft = cv2.resize(cropped * 255, (transform.new_w, transform.new_h), interpolation=cv2.INTER_AREA)
        resized = (soft > 0).astype(np.uint8)
    elif resize_interpolation == cv2.INTER_NEAREST:
        resized = cv2.resize(cropped, (transform.new_w, transform.new_h), interpolation=cv2.INTER_NEAREST)
    else:
        soft = cv2.resize(cropped * 255, (transform.new_w, transform.new_h), interpolation=resize_interpolation)
        resized = (soft > 127).astype(np.uint8)
    canvas[transform.y0 : transform.y0 + transform.new_h, transform.x0 : transform.x0 + transform.new_w] = resized
    return canvas


def crop_to_canvas(
    mask: np.ndarray,
    height: int,
    width: int,
    margin: int = 2,
    resize_interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Crop to mask's bounding box (with `margin` px breathing room) and rescale to fill height x width.

    resize_interpolation defaults to INTER_NEAREST (exact behavior CASIA-B/V6
    preprocessing has always used -- its native silhouettes are small enough
    that this crop-then-resize is an upscale, where NEAREST stays block-clean).
    Callers cropping from higher-resolution source (e.g. full HD video, where
    this step is a real downscale and NEAREST aliases into jagged edges) should
    pass cv2.INTER_AREA instead, which area-averages then re-binarizes.
    """
    transform = compute_crop_transform(mask, height, width, margin)
    if transform is None:
        return np.zeros((height, width), dtype=np.uint8)
    return apply_crop_transform(mask, transform, height, width, resize_interpolation=resize_interpolation)


def clean_and_align(
    mask: np.ndarray,
    config: ExperimentConfig,
    resize_interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Clean up a raw mask and crop/rescale it to fill config.height x width.

    This is what V6 training applies to CASIA-B's raw silhouettes at
    cache-build time -- see crop_to_canvas()'s docstring for the
    resize_interpolation tradeoff.
    """
    cleaned = clean_silhouette_mask(mask, config)
    return crop_to_canvas(cleaned, config.height, config.width, margin=2, resize_interpolation=resize_interpolation)


def hamilton_jacobi_topology(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return topology heatmap, skeletal radius, and flux strength.

    This is a discrete HJ-inspired implementation: the Euclidean distance field
    supplies the normalized gradient and outward-flux shock strength, while
    topology-preserving thinning supplies the connected medial set. It is a
    practical prototype, not a reproduction of Siddiqi et al.'s complete
    flux-ordered thinning algorithm.
    """
    foreground = mask.astype(bool)
    if foreground.sum() < 4:
        zeros = np.zeros_like(mask, dtype=np.float32)
        return zeros, zeros, zeros

    distance = ndimage.distance_transform_edt(foreground).astype(np.float32)
    grad_y, grad_x = np.gradient(distance)
    norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
    field_x = grad_x / norm
    field_y = grad_y / norm
    divergence = np.gradient(field_x, axis=1) + np.gradient(field_y, axis=0)
    flux_strength = np.maximum(-ndimage.uniform_filter(divergence, size=3), 0.0)

    medial = topology_preserving_thinning(foreground)
    topology = ndimage.gaussian_filter(medial.astype(np.float32), sigma=0.65)
    if topology.max() > 0:
        topology /= topology.max()
    radius = distance / max(float(distance.max()), 1.0)
    radius *= topology
    flux_strength *= medial
    if flux_strength.max() > 0:
        flux_strength /= flux_strength.max()
    return topology.astype(np.float32), radius.astype(np.float32), flux_strength.astype(np.float32)


def uniform_sample(items: list[bytes], length: int) -> list[bytes]:
    if not items:
        raise ValueError("Cannot sample an empty sequence")
    indices = np.linspace(0, len(items) - 1, length).round().astype(int)
    return [items[index] for index in indices]


def process_sequence(sequence: RawSequence, config: ExperimentConfig) -> np.ndarray:
    output = np.zeros((config.sequence_length, 4, config.height, config.width), dtype=np.uint8)
    for index, encoded in enumerate(uniform_sample(sequence.frames, config.sequence_length)):
        silhouette = clean_and_align(decode_mask(encoded), config)
        topology, radius, flux = hamilton_jacobi_topology(silhouette)
        output[index, 0] = silhouette * 255
        output[index, 1] = np.round(topology * 255).astype(np.uint8)
        output[index, 2] = np.round(radius * 255).astype(np.uint8)
        output[index, 3] = np.round(flux * 255).astype(np.uint8)
    return output


def process_skeleton_sequence(sequence: RawSequence, config: ExperimentConfig) -> np.ndarray:
    """Prepare Hamilton skeleton maps for sequence verification.

    Output channels:
    0. binary Hamilton skeleton image,
    1. blurred structural field around the skeleton,
    2. temporal motion-difference field against the previous sampled frame.
    """
    output = np.zeros((config.sequence_length, 3, config.height, config.width), dtype=np.uint8)
    previous: np.ndarray | None = None
    for index, encoded in enumerate(uniform_sample(sequence.frames, config.sequence_length)):
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("OpenCV could not decode a skeleton frame")
        skeleton = (image > 0).astype(np.uint8)
        if skeleton.shape != (config.height, config.width):
            skeleton = cv2.resize(skeleton, (config.width, config.height), interpolation=cv2.INTER_NEAREST)
        structure = cv2.GaussianBlur(skeleton.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
        if structure.max() > 0:
            structure /= structure.max()
        motion = np.zeros_like(structure)
        if previous is not None:
            motion = np.abs(skeleton.astype(np.float32) - previous.astype(np.float32))
            motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=0.9, sigmaY=0.9)
            if motion.max() > 0:
                motion /= motion.max()
        previous = skeleton
        output[index, 0] = skeleton * 255
        output[index, 1] = np.round(structure * 255).astype(np.uint8)
        output[index, 2] = np.round(motion * 255).astype(np.uint8)
    return output


def process_skeleton_silhouette_sequence(
    skeleton_sequence: RawSequence,
    silhouette_sequence: RawSequence,
    config: ExperimentConfig,
) -> np.ndarray:
    """Prepare fused CASIA-B silhouette + Hamilton skeleton maps.

    Output channels:
    0. cleaned/aligned silhouette,
    1. binary Hamilton skeleton,
    2. blurred structural field around the skeleton,
    3. temporal skeleton motion-difference field.

    Skeleton alignment has two supported cases:
    - Already pre-aligned to (config.height, config.width) externally (CASIA-B's
      CASIA_B_Hamilton_Skeleton convention) -- used as-is, no resize at all.
    - Shares the same native pixel grid as the paired silhouette (CLoP-Gait's own
      pipeline: tools/dataset_pipeline/skeleton.py computes the skeleton directly
      from the same stored silhouette PNG) -- co-aligned using the SAME crop
      transform as the silhouette, so the two channels stay spatially registered.
      A naive direct resize here would both misalign the two channels (each
      cropped/scaled independently) and, for a 1px-wide skeleton line being
      downscaled a lot (e.g. CLoP-Gait's 320x240 storage -> 64x64 training),
      destroy nearly all of it via point-sampled NEAREST. Instead this crops
      with the silhouette's transform, downsamples via any-coverage (keeps a
      pixel if the source region has ANY skeleton coverage, not just >50%),
      then re-thins so the result is still a genuine 1px skeleton.
    """
    output = np.zeros((config.sequence_length, 4, config.height, config.width), dtype=np.uint8)
    skeleton_frames = uniform_sample(skeleton_sequence.frames, config.sequence_length)
    silhouette_frames = uniform_sample(silhouette_sequence.frames, config.sequence_length)
    previous_skeleton: np.ndarray | None = None
    for index, (skeleton_encoded, silhouette_encoded) in enumerate(zip(skeleton_frames, silhouette_frames)):
        raw_silhouette_mask = decode_mask(silhouette_encoded)
        cleaned_silhouette_mask = clean_silhouette_mask(raw_silhouette_mask, config)
        transform = compute_crop_transform(cleaned_silhouette_mask, config.height, config.width, margin=2)
        if transform is None:
            silhouette = np.zeros((config.height, config.width), dtype=np.uint8)
        else:
            silhouette = apply_crop_transform(cleaned_silhouette_mask, transform, config.height, config.width)

        skeleton_image = cv2.imdecode(np.frombuffer(skeleton_encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if skeleton_image is None:
            raise ValueError("OpenCV could not decode a skeleton frame")
        skeleton_mask = skeleton_image > 0
        if skeleton_mask.shape == (config.height, config.width):
            skeleton = skeleton_mask.astype(np.uint8)
        elif skeleton_mask.shape == raw_silhouette_mask.shape and transform is not None:
            covered = apply_crop_transform(
                skeleton_mask.astype(np.uint8), transform, config.height, config.width, any_coverage=True
            )
            skeleton = topology_preserving_thinning(covered.astype(bool)).astype(np.uint8)
        else:
            skeleton = cv2.resize(
                skeleton_mask.astype(np.uint8), (config.width, config.height), interpolation=cv2.INTER_NEAREST
            )
        structure = cv2.GaussianBlur(skeleton.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
        if structure.max() > 0:
            structure /= structure.max()
        motion = np.zeros_like(structure)
        if previous_skeleton is not None:
            motion = np.abs(skeleton.astype(np.float32) - previous_skeleton.astype(np.float32))
            motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=0.9, sigmaY=0.9)
            if motion.max() > 0:
                motion /= motion.max()
        previous_skeleton = skeleton

        output[index, 0] = silhouette * 255
        output[index, 1] = skeleton * 255
        output[index, 2] = np.round(structure * 255).astype(np.uint8)
        output[index, 3] = np.round(motion * 255).astype(np.uint8)
    return output


def prepare_dataset(config: ExperimentConfig, force: bool = False) -> dict[str, int | str]:
    config.ensure_output_dirs()
    cache_dir = Path(config.cache_dir)
    samples_dir = cache_dir / "samples"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "COMPLETE.json"
    if complete_path.exists() and manifest_path.exists() and not force:
        return json.loads(complete_path.read_text())

    samples_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    subject_set: set[str] = set()
    silhouette_lookup: dict[tuple[str, str, str], RawSequence] = {}
    if config.dataset_format == "skeleton_silhouette_fusion":
        if not config.silhouette_dataset_path:
            raise ValueError("skeleton_silhouette_fusion requires silhouette_dataset_path")
        print(f"Indexing paired silhouette dataset: {config.silhouette_dataset_path}", flush=True)
        silhouette_lookup = build_sequence_lookup(config.silhouette_dataset_path)
        print(f"Indexed {len(silhouette_lookup)} silhouette sequences for fusion", flush=True)
    missing_pairs = 0
    for sample_index, sequence in enumerate(iter_archive_sequences(config.dataset_path)):
        if config.dataset_format == "skeleton_hamilton":
            maps = process_skeleton_sequence(sequence, config)
        elif config.dataset_format == "skeleton_silhouette_fusion":
            paired = silhouette_lookup.get(_sequence_lookup_key(sequence))
            if paired is None:
                missing_pairs += 1
                continue
            maps = process_skeleton_silhouette_sequence(sequence, paired, config)
        elif config.dataset_format == "silhouette_hj":
            maps = process_sequence(sequence, config)
        else:
            raise ValueError(f"Unsupported dataset_format: {config.dataset_format!r}")
        filename = f"{sample_index:06d}_{sequence.subject}_{sequence.condition}_{sequence.view}.npz"
        np.savez_compressed(samples_dir / filename, maps=maps)
        records.append(
            {
                "path": f"samples/{filename}",
                "subject": sequence.subject,
                "condition": sequence.condition,
                "view": sequence.view,
                "source_key": sequence.key,
                "frames": len(sequence.frames),
            }
        )
        subject_set.add(sequence.subject)
        if (sample_index + 1) % 100 == 0:
            print(f"Prepared {sample_index + 1} sequences", flush=True)

    if not records:
        raise RuntimeError(f"No image sequences found in {config.dataset_path}")
    manifest_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    summary: dict[str, int | str] = {
        "dataset": config.dataset_name,
        "sequences": len(records),
        "subjects": len(subject_set),
        "height": config.height,
        "width": config.width,
        "sequence_length": config.sequence_length,
        "representation": (
            "silhouette+hamilton_skeleton+structure_blur+temporal_motion"
            if config.dataset_format == "skeleton_silhouette_fusion"
            else (
                "hamilton_skeleton+structure_blur+temporal_motion"
                if config.dataset_format == "skeleton_hamilton"
                else "silhouette+discrete_hj_topology+radius+flux"
            )
        ),
        "missing_pairs": missing_pairs,
    }
    complete_path.write_text(json.dumps(summary, indent=2))
    return summary

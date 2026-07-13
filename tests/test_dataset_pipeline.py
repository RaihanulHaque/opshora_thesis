import numpy as np

from tools.dataset_pipeline.segment import select_tracked_mask


def _blob(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    mask[(ys - center[0]) ** 2 + (xs - center[1]) ** 2 <= radius**2] = 1
    return mask


def test_first_frame_picks_largest_mask() -> None:
    small = _blob((64, 64), (10, 10), radius=3)
    large = _blob((64, 64), (40, 40), radius=8)
    chosen, centroid = select_tracked_mask([small, large], previous_centroid=None)
    assert np.array_equal(chosen, large)
    assert centroid is not None
    assert abs(centroid[0] - 40) < 1 and abs(centroid[1] - 40) < 1


def test_later_frame_picks_closest_centroid_not_largest() -> None:
    tracked_person = _blob((64, 64), (12, 12), radius=3)
    background_person = _blob((64, 64), (50, 50), radius=10)
    chosen, centroid = select_tracked_mask(
        [background_person, tracked_person], previous_centroid=(11.0, 11.0)
    )
    assert np.array_equal(chosen, tracked_person)
    assert abs(centroid[0] - 12) < 1 and abs(centroid[1] - 12) < 1


def test_no_masks_returns_none_and_keeps_previous_centroid() -> None:
    chosen, centroid = select_tracked_mask([], previous_centroid=(5.0, 5.0))
    assert chosen is None
    assert centroid == (5.0, 5.0)


def test_empty_mask_candidates_fall_back_to_previous_centroid() -> None:
    empty = np.zeros((64, 64), dtype=np.uint8)
    chosen, centroid = select_tracked_mask([empty], previous_centroid=(5.0, 5.0))
    assert chosen is None
    assert centroid == (5.0, 5.0)

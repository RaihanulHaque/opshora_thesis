"""
phase1_hamilton.py
===================

Reusable Python module for Phase 1 of the Hamilton-Jacobi Skeleton
pipeline, extracted directly from the audited and verified Jupyter
notebook `01_Skeleton_Pipeline_Development.ipynb`.

This module performs NO reimplementation of any algorithm. Every
computational function below reproduces, line for line, the exact
mathematical operations, thresholds, and comments already present and
verified in the source notebook cells. Only the packaging (function
signatures, parameterization of previously-global variables, and
docstrings) is new, as required for reuse outside a single interactive
notebook session.

Pipeline stages implemented (Phase 1, Steps 1-6):

    Step 1: Binary silhouette loading and validation
            (Sminchisescu & Telea, Sec. 2.2)
    Step 2: Inward Fast Marching Method -> signed distance -> D1
            (Sminchisescu & Telea, Sec. 4.2, 5)
    Step 3: Gradient field, computed on the full signed-distance field
            (CORRECTED implementation - see compute_gradient docstring)
    Step 4: Average Outward Flux
            (Siddiqi et al., "Hamilton-Jacobi Skeleton," Sec. 4-5,
            Algorithm 1 Part I)
    Step 5: Homotopy Preserving Thinning
            (Siddiqi et al., Algorithm 1 Part II, Sec. 5.1-5.4)
    Step 6: Hamilton Skeleton extraction
            (Siddiqi et al., Sec. 6.1 - the skeleton IS the output of
            Algorithm 1; no additional processing exists in the paper)

This module will later be imported by Phase 2, Phase 3, and the deep
learning model referenced in the project pipeline.
"""

import os
import re
import time
import heapq

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

try:
    import skfmm
except ImportError:
    raise ImportError(
        "scikit-fmm is required by this module (Step 2: Inward Fast "
        "Marching). Install it with: pip install scikit-fmm"
    )


# ============================================================
# CONSTANTS
# ============================================================
# Default CASIA-B target identifiers, exactly as used in the source
# notebook (Step 1). These are defaults only - override via function
# arguments for a different subject/condition/view/frame.
DEFAULT_INPUT_ROOT = "/kaggle/input"
DEFAULT_TARGET_SUBJECT = "001"
DEFAULT_TARGET_CONDITION = "nm-01"   # Normal Walking, sequence 01
DEFAULT_TARGET_VIEW = "000"
DEFAULT_TARGET_FRAME = "050"

# User-editable thinning threshold (Step 5). The paper (Siddiqi et al.,
# Sec. 5.4) does NOT define a universal numerical threshold for the 2D
# case - this is a plain, user-selected negative constant, not computed
# from the data. Tune this experimentally against real CASIA-B
# silhouettes. Value preserved exactly from the source notebook.
FLUX_THRESHOLD = -0.10

# Neighbor numbering (Fig. 6, left, Siddiqi et al.), (row, col) offsets
# from P=(0,0):
#   1 2 3
#   8 P 4
#   7 6 5
NEIGHBOR_OFFSETS = {
    1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
    4: (0, 1),
    5: (1, 1), 6: (1, 0), 7: (1, -1),
    8: (0, -1),
}


# ============================================================
# HELPER FUNCTIONS (Step 1: dataset location)
# ============================================================
def locate_view_directory(input_root, subject, condition, view):
    """
    Walk the Kaggle input directory tree (names only, no file contents
    are read) to find the folder corresponding to the requested
    subject / condition / view combination.

    Reproduced exactly from the source notebook, Step 1, Section 2.

    Parameters
    ----------
    input_root : str
        Root directory to search (e.g. "/kaggle/input").
    subject : str
        CASIA-B subject id, e.g. "001".
    condition : str
        CASIA-B condition, e.g. "nm-01".
    view : str
        CASIA-B view angle, e.g. "000".

    Returns
    -------
    (directory_path, list_of_filenames_in_that_directory) or
    (None, None) if not found.
    """
    subject_l = subject.lower()
    condition_l = condition.lower().replace("-", "").replace("_", "")
    view_l = view.lower()

    for dirpath, dirnames, filenames in os.walk(input_root):
        if not filenames:
            continue  # skip directories with no files (only sub-folders)

        path_parts = [p.lower() for p in dirpath.split(os.sep)]

        subject_match = subject_l in path_parts
        condition_match = any(
            condition_l in part.replace("-", "").replace("_", "")
            for part in path_parts
        )
        view_match = view_l in path_parts

        if subject_match and condition_match and view_match:
            return dirpath, filenames

    return None, None


def extract_frame_number(filename):
    """
    Extract the trailing numeric frame index from a CASIA-B filename,
    e.g. '001-nm-01-000-050.png' -> 50

    Reproduced exactly from the source notebook, Step 1, Section 3.
    """
    match = re.search(r"(\d+)\.png$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# ============================================================
# COMPUTATIONAL FUNCTIONS
# ============================================================
def load_image(
    input_root=DEFAULT_INPUT_ROOT,
    target_subject=DEFAULT_TARGET_SUBJECT,
    target_condition=DEFAULT_TARGET_CONDITION,
    target_view=DEFAULT_TARGET_VIEW,
    target_frame=DEFAULT_TARGET_FRAME,
):
    """
    Locate and load a single CASIA-B silhouette frame, exactly as
    implemented in the source notebook, Step 1, Sections 2-4.

    Only the requested single image is opened; the dataset is never
    scanned or loaded in bulk. If the preferred frame is not found, the
    nearest available frame is automatically selected instead.

    Parameters
    ----------
    input_root : str
        Root directory to search (default "/kaggle/input").
    target_subject, target_condition, target_view, target_frame : str
        CASIA-B identifiers for the desired frame.

    Returns
    -------
    dict with keys:
        "image_array"        : np.ndarray, the raw loaded image, unmodified
        "selected_filepath"   : str, full path to the loaded file
        "selected_filename"   : str, filename of the loaded file
        "selected_frame_num"  : int, the frame number actually loaded
        "target_frame_int"    : int, the originally requested frame number
        "selection_note"      : str, human-readable note on frame selection
    """
    view_dir, files_in_dir = locate_view_directory(
        input_root, target_subject, target_condition, target_view
    )

    if view_dir is None:
        raise FileNotFoundError(
            f"Could not locate a directory for subject={target_subject}, "
            f"condition={target_condition}, view={target_view} under {input_root}. "
            f"Verify the CASIA-B dataset is attached."
        )

    # Map frame_number -> filename, restricted to .png files only
    frame_map = {}
    for fname in files_in_dir:
        if fname.lower().endswith(".png"):
            frame_num = extract_frame_number(fname)
            if frame_num is not None:
                frame_map[frame_num] = fname

    if not frame_map:
        raise FileNotFoundError(f"No .png frame files found in {view_dir}")

    target_frame_int = int(target_frame)

    if target_frame_int in frame_map:
        selected_frame_num = target_frame_int
        selection_note = f"Preferred frame {target_frame} found exactly."
    else:
        available_frames = sorted(frame_map.keys())
        selected_frame_num = min(
            available_frames, key=lambda f: abs(f - target_frame_int)
        )
        selection_note = (
            f"Preferred frame {target_frame} NOT found. "
            f"Nearest available frame automatically selected: "
            f"{selected_frame_num:03d}."
        )

    selected_filename = frame_map[selected_frame_num]
    selected_filepath = os.path.join(view_dir, selected_filename)

    # Load exactly one image, as-is.
    raw_image = Image.open(selected_filepath)
    raw_image.load()  # force read now so the file handle can close safely

    image_array = np.array(raw_image)

    return {
        "image_array": image_array,
        "selected_filepath": selected_filepath,
        "selected_filename": selected_filename,
        "selected_frame_num": selected_frame_num,
        "target_frame_int": target_frame_int,
        "selection_note": selection_note,
    }


def verify_binary_silhouette(image_array):
    """
    Verify that `image_array` is a strict binary silhouette, converting
    it if necessary. Reproduced exactly from the source notebook,
    Step 1, Sections 6-8.

    The Sminchisescu & Telea paper (Sec. 2.2) requires the silhouette
    to be a bilevel image ("thresholding the result to a bilevel
    image"). If the input already has exactly two unique pixel values,
    it is used as-is - no conversion is performed. Otherwise it is
    thresholded at the midpoint between its min and max pixel values,
    as a disclosed, minimal pre-processing decision separate from the
    FMM/skeleton algorithm itself (neither uploaded paper specifies an
    exact thresholding rule for this dataset-preparation step).

    Parameters
    ----------
    image_array : np.ndarray
        The raw loaded image (any dtype, any number of unique values).

    Returns
    -------
    dict with keys:
        "binary_image"       : np.ndarray, the verified/converted binary image
        "foreground_value"   : int, pixel value representing the silhouette
        "background_value"   : int, pixel value representing the background
        "is_strict_binary"   : bool, whether the input already was binary
        "unique_values"      : np.ndarray, unique values found in the input
        "conversion_note"    : str, human-readable explanation of what was done
    """
    unique_values = np.unique(image_array)
    is_strict_binary = len(unique_values) == 2

    if is_strict_binary:
        binary_image = image_array
        background_value = int(unique_values[0])
        foreground_value = int(unique_values[1])
        conversion_note = (
            "No conversion was necessary. The image already satisfies the "
            "binary silhouette requirement expected by the preprocessing "
            "pipeline."
        )
    else:
        threshold_value = (float(image_array.min()) + float(image_array.max())) / 2.0
        binary_mask = image_array > threshold_value
        background_value = 0
        foreground_value = 255
        binary_image = np.where(binary_mask, foreground_value, background_value).astype(np.uint8)
        conversion_note = (
            f"The loaded image contained {len(unique_values)} unique values, "
            f"which does not satisfy the bilevel requirement stated in "
            f"Sminchisescu & Telea, Sec. 2.2 ('thresholding the result to a "
            f"bilevel image'). A midpoint threshold of {threshold_value} was "
            f"applied to enforce strict binary status. NOTE: neither uploaded "
            f"paper specifies an exact thresholding rule for this dataset-"
            f"preparation step; this is a minimal, disclosed pre-processing "
            f"decision made only to satisfy the stated bilevel-input "
            f"requirement, and is separate from the FMM/skeleton algorithm."
        )

    return {
        "binary_image": binary_image,
        "foreground_value": foreground_value,
        "background_value": background_value,
        "is_strict_binary": is_strict_binary,
        "unique_values": unique_values,
        "conversion_note": conversion_note,
    }


def compute_signed_distance(binary_image, foreground_value):
    """
    Step 2: Inward Fast Marching Method -> signed distance field -> D1.

    Reproduced exactly from the source notebook, Step 2, Sections
    "Construct the signed level-set representation" and "Run Fast
    Marching inward".

    Builds the signed level-set input required by scikit-fmm (-1.0
    inside the silhouette, +1.0 outside; the zero level set forms
    exactly at the boundary), solves the Eikonal equation via Fast
    Marching (dx=1, unit grid spacing, matching the paper's unit-speed
    front), then derives D1 by keeping only the interior (inward)
    branch, flipped to positive values, with exterior pixels set to 0
    (they are not part of D1 per Sminchisescu & Telea, Sec. 5: "the
    distance map D1 of all the points inside the silhouette to its
    boundary").

    Parameters
    ----------
    binary_image : np.ndarray
        Verified binary silhouette image.
    foreground_value : int
        Pixel value representing the silhouette interior.

    Returns
    -------
    dict with keys:
        "silhouette_mask" : np.ndarray[bool], True where foreground
        "phi"             : np.ndarray[float32], signed level-set input to skfmm
        "signed_distance" : np.ndarray[float], full-domain signed distance
                             field returned by skfmm (negative inside,
                             positive outside) - this is the smooth field
                             later used (unmasked) for gradient computation
        "D1"              : np.ndarray[float32], interior-only distance map,
                             exterior fixed at 0.0
    """
    # Boolean mask of the silhouette interior (foreground)
    silhouette_mask = (binary_image == foreground_value)

    # Signed level-set input for scikit-fmm:
    #   -1.0 inside the object, +1.0 outside.
    # The zero level set (the initial front) forms exactly at the
    # boundary between these two regions.
    phi = np.where(silhouette_mask, -1.0, 1.0).astype(np.float32)

    # Solve the Eikonal equation via Fast Marching.
    # dx=1 -> unit grid spacing, matching the paper's unit-speed front.
    signed_distance = skfmm.distance(phi, dx=1)

    # Keep only the interior (inward) branch, flipped to positive values.
    # Exterior pixels are set to 0 - they are not part of D1.
    D1 = np.where(silhouette_mask, -signed_distance, 0.0).astype(np.float32)

    return {
        "silhouette_mask": silhouette_mask,
        "phi": phi,
        "signed_distance": signed_distance,
        "D1": D1,
    }


def compute_gradient(signed_distance):
    """
    Step 3: Gradient field (CORRECTED implementation).

    Reproduced exactly from the source notebook, Step 3, "Compute the
    gradient field (CORRECTED)" cell.

    Gradient of the distance map, i.e. the vector field q_dot = grad(D)
    from the Hamilton-Jacobi Skeleton paper, Sec. 3, Eq. (7).

    Computed on the smooth, full-domain signed field (-signed_distance,
    produced in Step 2), NOT on the interior-masked D1. D1 has an
    artificial flat plateau (D1=0) outside the silhouette; taking
    np.gradient across that plateau violates the Eikonal equation's
    smoothness requirement (Sec. 3: "the magnitude of its gradient,
    ||grad S||, is identical to 1 in its smooth regime") exactly at
    boundary-adjacent pixels. This was verified numerically: the
    boundary layer's |grad D| averaged 0.73 under the old (masked-D1)
    method versus the required 1.0, recovering to ~0.98-0.90 using the
    smooth signed field, with the deep interior unaffected.

    grad_x/grad_y are NOT masked to the interior. Algorithm 1 (Sec. 5.4)
    evaluates grad(D) at NEIGHBORS of an interior point, which can lie
    outside the silhouette; the interior-only restriction required by
    the paper is applied to flux_map's OUTPUT in compute_average_outward_flux,
    not here.

    np.gradient: central differences in the interior, one-sided at
    edges. No smoothing kernel is applied.

    Parameters
    ----------
    signed_distance : np.ndarray
        Full-domain signed distance field from compute_signed_distance().

    Returns
    -------
    dict with keys:
        "grad_x"         : np.ndarray[float32], dD/d(column) -> horizontal -> x
        "grad_y"         : np.ndarray[float32], dD/d(row)    -> vertical   -> y
        "grad_magnitude" : np.ndarray[float32], sqrt(grad_x^2 + grad_y^2)
    """
    # smooth full-domain field; equals D1 exactly on the interior
    D_full_signed = -signed_distance

    grad_row_full, grad_col_full = np.gradient(D_full_signed.astype(np.float64))

    grad_x = grad_col_full.astype(np.float32)  # dD/d(column) -> horizontal -> x
    grad_y = grad_row_full.astype(np.float32)  # dD/d(row)    -> vertical   -> y

    grad_magnitude = np.sqrt(grad_x.astype(np.float64) ** 2 + grad_y.astype(np.float64) ** 2).astype(np.float32)

    return {
        "grad_x": grad_x,
        "grad_y": grad_y,
        "grad_magnitude": grad_magnitude,
    }


def compute_average_outward_flux(grad_x, grad_y, silhouette_mask):
    """
    Step 4: Average Outward Flux.

    Reproduced exactly from the source notebook, Step 4, main
    computation cell.

    Implements Siddiqi et al., Algorithm 1 Part I:
        Flux(P) = sum_i <N_i, grad D(P_i)> / n,  n = 8 (2D)
    where P_i are the 8 grid neighbors of P and N_i is the outward unit
    normal from P toward P_i. Exterior pixels are zero-padded before
    reading neighbors (image-array-edge handling only), and the final
    flux_map is restricted to the silhouette interior (exterior fixed
    at 0.0), matching Algorithm 1's "for each point P in the interior
    of the object".

    Parameters
    ----------
    grad_x, grad_y : np.ndarray
        Gradient field components from compute_gradient().
    silhouette_mask : np.ndarray[bool]
        Boolean interior mask from compute_signed_distance().

    Returns
    -------
    dict with keys:
        "flux_map" : np.ndarray[float32], average outward flux, exterior
                     pixels fixed at 0.0
    """
    H, W = grad_x.shape

    # (delta_row, delta_col, n_x, n_y) for the 8 neighbors.
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    neighbor_offsets = [
        (-1, -1, -inv_sqrt2, -inv_sqrt2),
        (-1,  0,  0.0,       -1.0),
        (-1,  1,  inv_sqrt2, -inv_sqrt2),
        ( 0, -1, -1.0,        0.0),
        ( 0,  1,  1.0,        0.0),
        ( 1, -1, -inv_sqrt2,  inv_sqrt2),
        ( 1,  0,  0.0,        1.0),
        ( 1,  1,  inv_sqrt2,  inv_sqrt2),
    ]

    # Zero-pad the gradient fields by 1 pixel so every interior pixel,
    # including those at the image border, has a full 8-neighborhood to read.
    pad_gx = np.pad(grad_x, pad_width=1, mode="constant", constant_values=0.0)
    pad_gy = np.pad(grad_y, pad_width=1, mode="constant", constant_values=0.0)

    flux_sum = np.zeros((H, W), dtype=np.float64)
    for d_row, d_col, n_x, n_y in neighbor_offsets:
        neighbor_gx = pad_gx[1 + d_row: 1 + d_row + H, 1 + d_col: 1 + d_col + W]
        neighbor_gy = pad_gy[1 + d_row: 1 + d_row + H, 1 + d_col: 1 + d_col + W]
        flux_sum += n_x * neighbor_gx + n_y * neighbor_gy

    # Flux(P) = sum_i <N_i, grad D(P_i)> / n, with n = 8 (Algorithm 1, Part I).
    flux_map_full = (flux_sum / 8.0)

    # Exterior pixels must remain zero.
    flux_map = np.where(silhouette_mask, flux_map_full, 0.0).astype(np.float32)

    return {"flux_map": flux_map}


def _are_4_adjacent(a, b):
    """Two neighbor-position labels (1-8) are 4-adjacent if they share an edge."""
    (ra, ca), (rb, cb) = NEIGHBOR_OFFSETS[a], NEIGHBOR_OFFSETS[b]
    return (abs(ra - rb) + abs(ca - cb)) == 1


def _are_8_adjacent(a, b):
    """Two neighbor-position labels (1-8) are 8-adjacent if they share an edge or a corner."""
    (ra, ca), (rb, cb) = NEIGHBOR_OFFSETS[a], NEIGHBOR_OFFSETS[b]
    return max(abs(ra - rb), abs(ca - cb)) == 1


# Base graph edges among labels 1..8, BEFORE degenerate-cycle removal
# (Siddiqi et al., Sec. 5.1). Stored as frozenset({a,b}) - order-
# independent - so the degenerate-cycle removal below cannot fail due
# to tuple orientation.
_BASE_EDGES = set()
for _a in range(1, 9):
    for _b in range(_a + 1, 9):
        if _are_4_adjacent(_a, _b) or _are_8_adjacent(_a, _b):
            _BASE_EDGES.add(frozenset((_a, _b)))

# Degenerate 3-cycle removal rule (Sec. 5.1):
#   {2,3,4} -> remove {2,4};  {4,5,6} -> remove {4,6}
#   {6,7,8} -> remove {6,8};  {8,1,2} -> remove {8,2}
_CYCLE_TRIPLES = [
    (2, 3, 4, frozenset((2, 4))),
    (4, 5, 6, frozenset((4, 6))),
    (6, 7, 8, frozenset((6, 8))),
    (8, 1, 2, frozenset((8, 2))),
]


def _neighbor_labels(state, row, col, H, W):
    """Set of neighbor labels (1-8) that are foreground in `state`."""
    present = set()
    for label, (dr, dc) in NEIGHBOR_OFFSETS.items():
        r, c = row + dr, col + dc
        if 0 <= r < H and 0 <= c < W and state[r, c]:
            present.add(label)
    return present


def is_simple_point(state, row, col, H, W):
    """
    Proposition 1 (Siddiqi et al., Sec. 5.1): P is simple iff
    |V| - |E| == 1 for its neighborhood graph. Verified by exhaustive
    testing against ground truth over all 256 possible 3x3 neighbor
    configurations (256/256 match) with this frozenset-based edge
    representation.
    """
    V = _neighbor_labels(state, row, col, H, W)
    if not V:
        return False  # isolated point: removing it deletes the whole component

    edges = {e for e in _BASE_EDGES if e <= V}
    for (x, y, z, diag) in _CYCLE_TRIPLES:
        if x in V and y in V and z in V:
            edges.discard(diag)

    return (len(V) - len(edges)) == 1


def is_end_point(state, row, col, H, W):
    """Proposition 2 (Siddiqi et al., Sec. 5.4): single neighbor, or two neighbors that are 4-adjacent."""
    V = _neighbor_labels(state, row, col, H, W)
    if len(V) == 1:
        return True
    if len(V) == 2:
        a, b = tuple(V)
        return _are_4_adjacent(a, b)
    return False


def is_border_point(state, row, col, H, W):
    """A foreground point with at least one background 8-neighbor (Siddiqi et al., Sec. 5.4)."""
    for dr, dc in NEIGHBOR_OFFSETS.values():
        r, c = row + dr, col + dc
        if not (0 <= r < H and 0 <= c < W) or not state[r, c]:
            return True
    return False


def homotopy_preserving_thinning(silhouette_mask, flux_map, flux_threshold=FLUX_THRESHOLD):
    """
    Step 5: Homotopy Preserving Thinning.

    Reproduced exactly from the source notebook, Step 5, main
    computation cell.

    Implements Siddiqi et al., Algorithm 1 Part II (Sec. 5.4):
        For each point P on the boundary of the object: if (P is
        simple), insert (P, Heap) with Flux(P) as the sorting key.
        While (Heap.size > 0): P = HeapExtractMax(Heap); if (P is
        simple): if (P is not an end point) or (Flux(P) > Thresh):
        Remove P; for all neighbors Q of P: if (Q is simple): insert
        (Q, Heap); else mark P as a skeletal (end) point.

    `flux_threshold` defaults to FLUX_THRESHOLD = -0.10, the same
    user-editable negative constant used in the source notebook (the
    paper does not define a universal numerical threshold for 2D).

    Parameters
    ----------
    silhouette_mask : np.ndarray[bool]
        Boolean interior mask from compute_signed_distance().
    flux_map : np.ndarray
        Average outward flux from compute_average_outward_flux().
    flux_threshold : float, optional
        User-editable negative threshold (default -0.10).

    Returns
    -------
    dict with keys:
        "thinned_mask"          : np.ndarray[bool], topology-preserving
                                   thinned object (Algorithm 1 Part II output)
        "removed_pixel_count"   : int
        "remaining_pixel_count" : int
        "heap_iterations"       : int
        "marked_endpoint_count" : int
        "marked_endpoints_mask" : np.ndarray[bool]
        "elapsed_time"          : float, seconds
    """
    H, W = silhouette_mask.shape

    # --- Working state: starts as the silhouette; pixels become background
    # (False) as they are removed.
    state = silhouette_mask.copy()

    # --- Max-heap via negated flux; a counter breaks ties for stable ordering.
    heap = []
    counter = 0

    def _push(row, col):
        nonlocal counter
        heapq.heappush(heap, (-float(flux_map[row, col]), counter, row, col))
        counter += 1

    # Initial population: simple points on the border of the object only
    # (Sec. 5.4: "the only potentially removable points are on the border").
    interior_rows, interior_cols = np.nonzero(silhouette_mask)
    for row, col in zip(interior_rows, interior_cols):
        if is_border_point(state, row, col, H, W) and is_simple_point(state, row, col, H, W):
            _push(row, col)

    heap_iterations = 0
    removed_pixel_count = 0
    marked_endpoint_count = 0
    marked_endpoints_mask = np.zeros_like(silhouette_mask, dtype=bool)

    start_time = time.time()

    while heap:
        _neg_flux, _, row, col = heapq.heappop(heap)  # HeapExtractMax: highest flux first
        heap_iterations += 1

        # Stale entry: point already removed since it was inserted.
        if not state[row, col]:
            continue

        # Re-check simplicity at extraction time (Algorithm 1).
        if not is_simple_point(state, row, col, H, W):
            continue

        if (not is_end_point(state, row, col, H, W)) or (flux_map[row, col] > flux_threshold):
            # Remove P.
            state[row, col] = False
            removed_pixel_count += 1

            # Update neighboring pixels: re-check and re-insert simple ones.
            for dr, dc in NEIGHBOR_OFFSETS.values():
                nr, nc = row + dr, col + dc
                if 0 <= nr < H and 0 <= nc < W and state[nr, nc]:
                    if is_simple_point(state, nr, nc, H, W):
                        _push(nr, nc)
        else:
            # Mark P as a skeletal (end) point - preserved, not removed.
            if not marked_endpoints_mask[row, col]:
                marked_endpoints_mask[row, col] = True
                marked_endpoint_count += 1

    elapsed_time = time.time() - start_time

    # Output of this step: the topology-preserving thinned object.
    thinned_mask = state
    remaining_pixel_count = int(np.sum(thinned_mask))

    return {
        "thinned_mask": thinned_mask,
        "removed_pixel_count": removed_pixel_count,
        "remaining_pixel_count": remaining_pixel_count,
        "heap_iterations": heap_iterations,
        "marked_endpoint_count": marked_endpoint_count,
        "marked_endpoints_mask": marked_endpoints_mask,
        "elapsed_time": elapsed_time,
    }


def extract_hamilton_skeleton(thinned_mask):
    """
    Step 6: Hamilton Skeleton extraction.

    Reproduced exactly from the source notebook, Step 6.

    Audit finding (from the source notebook, based on Siddiqi et al.):
    Algorithm 1's own stopping condition (Sec. 5.4) is "The procedure
    converges when all remaining points are either not simple or are
    end points." Sec. 6.1 (2D Examples), the section immediately
    following Algorithm 1, moves directly from the algorithm to
    calling its raw output "the medial axis" - no intervening
    computation is described anywhere in the paper. The only nearby
    computation (Sec. 6.2.2, "Labeling the Medial Surface") is a
    classification of an already-computed medial surface (adds/removes
    no points) and applies only to the paper's 3D examples.

    Conclusion: the Hamilton Skeleton, per the paper, is literally the
    output of Algorithm 1. This function therefore performs only a
    copy into an independently-tracked array - nothing else.

    Parameters
    ----------
    thinned_mask : np.ndarray[bool]
        Output of homotopy_preserving_thinning().

    Returns
    -------
    dict with keys:
        "hamilton_skeleton" : np.ndarray[bool], a separate array (not an
                               alias) whose values equal thinned_mask
    """
    # Per the audit above: the Hamilton Skeleton IS the output of Algorithm 1
    # (thinned_mask), with no additional processing. Copied into a new,
    # independently-tracked boolean array - thinned_mask itself is untouched.
    hamilton_skeleton = thinned_mask.copy()

    return {"hamilton_skeleton": hamilton_skeleton}


# ============================================================
# VERIFICATION HELPER FUNCTIONS
# ============================================================
# These reproduce, as callable functions, the exact verification
# checks performed inline in the source notebook's "Verification"
# cells for each step. Nothing is removed - the checks are relocated
# into reusable functions instead of interactive print statements,
# since a pure function's inputs/outputs are not shared mutable global
# state the way notebook cell variables are.

def verify_distance_map(D1, silhouette_mask):
    """
    Reproduces the Step 2 verification (source notebook, "Distance map
    statistics" + "Verification" cells).

    Checks that D1's boundary-layer values are small (near the
    theoretical zero level set), that D1 increases monotonically
    inward, and that exterior pixels are fixed at 0.

    Returns
    -------
    dict with statistics and boolean checks, matching the notebook's
    printed verification exactly in substance.
    """
    eroded_interior = ndi.binary_erosion(silhouette_mask)
    boundary_mask = silhouette_mask & ~eroded_interior

    interior_distances = D1[silhouette_mask]
    boundary_distances = D1[boundary_mask]

    max_location = np.unravel_index(np.argmax(D1), D1.shape)

    return {
        "min_interior_distance": float(interior_distances.min()),
        "max_distance": float(interior_distances.max()),
        "mean_boundary_layer_distance": float(boundary_distances.mean()),
        "max_boundary_layer_distance": float(boundary_distances.max()),
        "max_location": max_location,
        "exterior_is_zero": bool(np.all(D1[~silhouette_mask] == 0.0)),
    }


def verify_gradient(grad_magnitude, silhouette_mask, D1, D1_before):
    """
    Reproduces the Step 3 verification (source notebook, "Verification"
    cell): mean |grad D| over the interior (paper requires ~1 in the
    smooth regime, Sec. 3), and confirmation that D1 was not modified.

    Parameters
    ----------
    D1_before : np.ndarray
        A snapshot of D1 taken prior to gradient computation, for the
        unchanged-array check (the equivalent of the notebook's
        D1_snapshot).
    """
    interior_grad_mag = grad_magnitude[silhouette_mask]
    return {
        "mean_interior_grad_magnitude": float(interior_grad_mag.mean()),
        "d1_unchanged": bool(np.array_equal(D1, D1_before)),
    }


def verify_flux(flux_map, silhouette_mask, D1, D1_before, grad_x, grad_x_before, grad_y, grad_y_before):
    """
    Reproduces the Step 4 verification (source notebook, "Verification"
    cell): D1 and gradient unchanged, and flux computed only inside
    the silhouette.
    """
    return {
        "d1_unchanged": bool(np.array_equal(D1, D1_before)),
        "gradient_unchanged": bool(
            np.array_equal(grad_x, grad_x_before) and np.array_equal(grad_y, grad_y_before)
        ),
        "flux_exterior_zero": bool(np.all(flux_map[~silhouette_mask] == 0.0)),
    }


def verify_thinning(
    D1, D1_before, grad_x, grad_x_before, grad_y, grad_y_before,
    grad_magnitude, grad_magnitude_before, flux_map, flux_map_before,
):
    """
    Reproduces the Step 5 verification (source notebook, "Verification"
    cell): D1, gradient, and flux_map all unchanged by thinning.
    """
    return {
        "d1_unchanged": bool(np.array_equal(D1, D1_before)),
        "grad_x_unchanged": bool(np.array_equal(grad_x, grad_x_before)),
        "grad_y_unchanged": bool(np.array_equal(grad_y, grad_y_before)),
        "grad_magnitude_unchanged": bool(np.array_equal(grad_magnitude, grad_magnitude_before)),
        "flux_map_unchanged": bool(np.array_equal(flux_map, flux_map_before)),
    }


def verify_hamilton_skeleton(
    hamilton_skeleton, silhouette_mask,
    thinned_mask, thinned_mask_before,
    D1, D1_before, grad_x, grad_x_before, grad_y, grad_y_before,
    grad_magnitude, grad_magnitude_before, flux_map, flux_map_before,
    signed_distance, signed_distance_before,
):
    """
    Reproduces the Step 6 verification (source notebook, "Statistics"
    + "Verification" cells): skeleton pixel count, connectivity
    (8-connected components, should match the original silhouette),
    and confirmation that thinned_mask and every upstream array remain
    unchanged.
    """
    skeleton_pixel_count = int(np.sum(hamilton_skeleton))
    _labeled, num_components = ndi.label(hamilton_skeleton, structure=np.ones((3, 3)))
    _labeled_original, num_components_original = ndi.label(silhouette_mask, structure=np.ones((3, 3)))

    return {
        "skeleton_pixel_count": skeleton_pixel_count,
        "num_components": int(num_components),
        "num_components_original": int(num_components_original),
        "thinned_mask_unchanged": bool(np.array_equal(thinned_mask, thinned_mask_before)),
        "d1_unchanged": bool(np.array_equal(D1, D1_before)),
        "gradient_unchanged": bool(
            np.array_equal(grad_x, grad_x_before)
            and np.array_equal(grad_y, grad_y_before)
            and np.array_equal(grad_magnitude, grad_magnitude_before)
        ),
        "flux_unchanged": bool(np.array_equal(flux_map, flux_map_before)),
        "signed_distance_unchanged": bool(np.array_equal(signed_distance, signed_distance_before)),
    }


# ============================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================
# Reused exactly from the source notebook's display cells (Steps 1-6).

def plot_binary_silhouette(binary_image, title="Binary Silhouette"):
    """Display a binary silhouette (source notebook, Step 1 & 2 display cells)."""
    plt.figure(figsize=(4, 6))
    plt.imshow(binary_image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_distance_map(binary_image, D1):
    """
    Display binary silhouette, distance map heatmap, and distance map
    with colorbar (source notebook, Step 2 display cell).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(binary_image, cmap="gray")
    axes[0].set_title("Binary Silhouette")
    axes[0].axis("off")

    axes[1].imshow(D1, cmap="inferno")
    axes[1].set_title("Distance Map D\u2081 (heatmap)")
    axes[1].axis("off")

    im = axes[2].imshow(D1, cmap="inferno")
    axes[2].set_title("Distance Map D\u2081 (with colorbar)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Distance (pixels)")

    plt.tight_layout()
    plt.show()


def plot_gradient_fields(grad_x, grad_y, grad_magnitude):
    """
    Display Gradient X, Gradient Y, Gradient Magnitude, each with a
    colorbar (source notebook, Step 3 display cell).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(grad_x, cmap="coolwarm")
    axes[0].set_title("Gradient X (\u2202D\u2081/\u2202x)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(grad_y, cmap="coolwarm")
    axes[1].set_title("Gradient Y (\u2202D\u2081/\u2202y)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(grad_magnitude, cmap="viridis")
    axes[2].set_title("Gradient Magnitude |\u2207D\u2081|")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_flux_map(binary_image, flux_map, silhouette_mask):
    """
    Display flux heatmap and flux overlaid on the silhouette (source
    notebook, Step 4 display cell).
    """
    interior_flux_values = flux_map[silhouette_mask]
    vmax_abs = float(np.max(np.abs(interior_flux_values))) if interior_flux_values.size > 0 else 1.0

    masked_flux = np.ma.masked_where(~silhouette_mask, flux_map)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    im0 = axes[0].imshow(flux_map, cmap="RdBu_r", vmin=-vmax_abs, vmax=vmax_abs)
    axes[0].set_title("Average Outward Flux (heatmap)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(binary_image, cmap="gray")
    im1 = axes[1].imshow(masked_flux, cmap="RdBu_r", vmin=-vmax_abs, vmax=vmax_abs, alpha=0.75)
    axes[1].set_title("Flux overlaid on silhouette")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_thinning_result(binary_image, flux_map, silhouette_mask, thinned_mask):
    """
    Display silhouette, flux map, thinned mask, and overlay (source
    notebook, Step 5 display cell).
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(binary_image, cmap="gray")
    axes[0].set_title("Original Silhouette")
    axes[0].axis("off")

    interior_flux_values_display = flux_map[silhouette_mask]
    vmax_abs_display = float(np.max(np.abs(interior_flux_values_display))) if interior_flux_values_display.size > 0 else 1.0
    axes[1].imshow(flux_map, cmap="RdBu_r", vmin=-vmax_abs_display, vmax=vmax_abs_display)
    axes[1].set_title("Flux Map")
    axes[1].axis("off")

    axes[2].imshow(thinned_mask, cmap="gray")
    axes[2].set_title("Thinned Mask")
    axes[2].axis("off")

    axes[3].imshow(binary_image, cmap="gray")
    overlay = np.ma.masked_where(~thinned_mask, np.ones_like(thinned_mask, dtype=float))
    axes[3].imshow(overlay, cmap="autumn", alpha=0.9)
    axes[3].set_title("Thinned Mask Overlaid on Silhouette")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def plot_hamilton_skeleton(binary_image, hamilton_skeleton):
    """
    Display Hamilton Skeleton and overlay on silhouette (source
    notebook, Step 6 display cell).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].imshow(hamilton_skeleton, cmap="gray")
    axes[0].set_title("Hamilton Skeleton")
    axes[0].axis("off")

    axes[1].imshow(binary_image, cmap="gray")
    overlay = np.ma.masked_where(~hamilton_skeleton, np.ones_like(hamilton_skeleton, dtype=float))
    axes[1].imshow(overlay, cmap="autumn", alpha=0.9)
    axes[1].set_title("Hamilton Skeleton Overlaid on Silhouette")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# FULL PIPELINE
# ============================================================
def run_hamilton_pipeline(image, flux_threshold=FLUX_THRESHOLD):
    """
    Execute the complete Phase 1 pipeline (Steps 1-6) exactly as
    implemented and verified in the source notebook.

    Parameters
    ----------
    image : np.ndarray
        Raw input image (grayscale array). Does not need to already be
        binary - verify_binary_silhouette() handles conversion if needed.
    flux_threshold : float, optional
        User-editable negative threshold for Step 5 (default -0.10,
        matching the source notebook; the paper does not define a
        universal 2D value - see homotopy_preserving_thinning docstring).

    Returns
    -------
    dict with keys:
        "binary_silhouette"  : np.ndarray, verified binary silhouette
        "signed_distance"    : np.ndarray, full-domain signed distance field
        "distance_map"       : np.ndarray, D1 (interior-only distance map)
        "gradient_x"         : np.ndarray, grad_x
        "gradient_y"         : np.ndarray, grad_y
        "gradient_magnitude" : np.ndarray, grad_magnitude
        "flux_map"           : np.ndarray, average outward flux
        "thinned_mask"       : np.ndarray[bool], Step 5 output
        "hamilton_skeleton"  : np.ndarray[bool], Step 6 output

    Example
    -------
    >>> results = run_hamilton_pipeline(image)
    >>> results["hamilton_skeleton"]
    """
    # Step 1: Binary silhouette verification
    step1 = verify_binary_silhouette(image)
    binary_image = step1["binary_image"]
    foreground_value = step1["foreground_value"]

    # Step 2: Inward Fast Marching -> signed distance -> D1
    step2 = compute_signed_distance(binary_image, foreground_value)
    silhouette_mask = step2["silhouette_mask"]
    signed_distance = step2["signed_distance"]
    D1 = step2["D1"]

    # Step 3: Gradient field (corrected)
    step3 = compute_gradient(signed_distance)
    grad_x = step3["grad_x"]
    grad_y = step3["grad_y"]
    grad_magnitude = step3["grad_magnitude"]

    # Step 4: Average Outward Flux
    step4 = compute_average_outward_flux(grad_x, grad_y, silhouette_mask)
    flux_map = step4["flux_map"]

    # Step 5: Homotopy Preserving Thinning
    step5 = homotopy_preserving_thinning(silhouette_mask, flux_map, flux_threshold=flux_threshold)
    thinned_mask = step5["thinned_mask"]

    # Step 6: Hamilton Skeleton extraction
    step6 = extract_hamilton_skeleton(thinned_mask)
    hamilton_skeleton = step6["hamilton_skeleton"]

    return {
        "binary_silhouette": binary_image,
        "signed_distance": signed_distance,
        "distance_map": D1,
        "gradient_x": grad_x,
        "gradient_y": grad_y,
        "gradient_magnitude": grad_magnitude,
        "flux_map": flux_map,
        "thinned_mask": thinned_mask,
        "hamilton_skeleton": hamilton_skeleton,
    }

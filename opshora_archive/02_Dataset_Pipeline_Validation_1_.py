# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [markdown] Cell 1
# # Notebook 2 - Phase 2: Dataset Pipeline Validation
# ## (Representative Frame Validation)
# 
# This notebook validates the Phase 1 Hamilton-Jacobi Skeleton pipeline across the CASIA-B dataset by processing exactly one representative frame per sequence.
# 
# **This notebook does NOT reimplement any algorithm.** Every computation is delegated to `run_hamilton_pipeline()`, imported from the `phase1_hamilton` module produced by Notebook 1.
# 
# **Scope of this notebook:**
# - Step 2A: Dataset indexing (locate one representative/middle frame per sequence)
# - Step 2B: Dataset verification (coverage tables, missing-data checks)
# - Step 2C: Run the Phase 1 pipeline on every representative frame, saving outputs
# - Step 2D: Validation report (success/failure stats, integrity checks, sample visualizations)
# 
# **Explicitly out of scope (not implemented here or anywhere in this notebook):** contrastive learning, generative learning, feature extraction, classification, training, testing, evaluation, pruning, outward Fast Marching, or any deep learning model. Notebook 1 and `phase1_hamilton.py` are not modified.

# %% [markdown] Cell 2
# ## 0. Import the Phase 1 module
# 
# Per the task requirement, this notebook begins with `from phase1_hamilton import *`. All Hamilton-Jacobi Skeleton computation in this notebook goes through the single function `run_hamilton_pipeline()` from that module - nothing is duplicated from Notebook 1.

# %% [code] Cell 3
# The Phase 1 module (produced by Notebook 1) must be importable from
# the current working directory or the Python path.
from phase1_hamilton import *

# Additional libraries needed ONLY for dataset indexing, bookkeeping,
# I/O, and reporting in THIS notebook - none of these perform any part
# of the Hamilton-Jacobi Skeleton computation itself.
import os
import time
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

print("phase1_hamilton imported successfully.")
print("run_hamilton_pipeline available:", callable(run_hamilton_pipeline))

# %% [markdown] Cell 4
# ## Step 2A: Dataset Indexing
# 
# **Assumed CASIA-B directory structure** (per task spec):
# 
# ```
# Subject/
#     nm-01/
#         000/
#         090/
#         180/
#     ...
#     nm-06/
#     cl-01/
#     cl-02/
# ```
# 
# **Scope:**
# - Subjects: `001`-`040`
# - Conditions: `nm-01`..`nm-06`, `cl-01`, `cl-02`
# - Angles: `000`, `090`, `180`
# 
# For every `(subject, condition, angle)` sequence, exactly **one** representative frame is selected: the exact middle frame.
# 
# **Indexing convention, stated explicitly to avoid future confusion: Python uses zero-based indexing.**
# 
# ```
# frame_list      = sorted PNG filenames for this sequence   # frame_list[0] is the first frame
# middle_index    = len(frame_list) // 2                     # 0-based index INTO frame_list
# selected_frame  = frame_list[middle_index]                 # the actual file selected
# display_frame_number = middle_index + 1                    # 1-based, for HUMAN-READABLE reporting only
# ```
# 
# `middle_index` is the only value ever used to index into `frame_list`. `display_frame_number` is a separate, purely cosmetic 1-based label stored alongside it so that anyone reading the index table sees an unambiguous, human-friendly frame count - it is never used for indexing.
# 
# Result is stored in a pandas DataFrame with columns: `Subject`, `Condition`, `Angle`, `Total Frames`, `Selected Frame Index` (0-based), `Display Frame Number` (1-based, display only), `Selected Filename`, `Full Path`.

# %% [code] Cell 5
# ============================================================
# CONFIGURATION
# ============================================================
SUBJECTS = [f"{i:03d}" for i in range(1, 41)]          # "001".."040"
CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06", "cl-01", "cl-02"]
ANGLES = ["000", "090", "180"]

OUTPUT_ROOT = "Output"  # preserves the CASIA-B hierarchy under this root


def resolve_dataset_root(search_base="/kaggle/input", sample_subject="001"):
    """
    Auto-locate the dataset root: the directory whose immediate children
    are the subject folders (e.g. '001', '002', ...). Searches under
    `search_base` for a directory that directly contains a subfolder
    named `sample_subject`. Only directory names are inspected - no
    image files are opened during this search.

    Returns the located root path, or None if not found.
    """
    for dirpath, dirnames, filenames in os.walk(search_base):
        if sample_subject in dirnames:
            return dirpath
    return None


DATASET_ROOT = resolve_dataset_root()
if DATASET_ROOT is None:
    # Fallback default - adjust to your actual dataset location if
    # auto-detection does not find it (e.g. a different Kaggle dataset
    # slug, or a locally mounted path).
    DATASET_ROOT = "/kaggle/input/casia-b"
    print(f"WARNING: could not auto-detect the dataset root. "
          f"Falling back to default: {DATASET_ROOT}")
else:
    print(f"Dataset root auto-detected: {DATASET_ROOT}")

# %% [code] Cell 6
def index_sequence(dataset_root, subject, condition, angle):
    """
    Locate the frame directory for (subject, condition, angle), list all
    PNG frames, and select the exact middle frame as the representative
    frame for this sequence.

    IMPORTANT - indexing convention (stated explicitly to avoid future
    confusion): Python uses ZERO-BASED indexing. `frame_list[0]` is the
    first frame in the sorted list.

        middle_index          = len(frame_list) // 2   # 0-based list index
        display_frame_number  = middle_index + 1        # 1-based, HUMANS ONLY

    `middle_index` is the only value used to actually select the file
    (`frame_list[middle_index]`). `display_frame_number` is a separate,
    purely cosmetic 1-based label for reporting - it is never used to
    index into `frame_list`.

    Parameters
    ----------
    dataset_root : str
        Root directory containing subject folders.
    subject, condition, angle : str
        CASIA-B identifiers, e.g. "001", "nm-01", "000".

    Returns
    -------
    dict (one row for the index dataframe), or None if the sequence
    directory does not exist or contains no PNG frames.
    """
    seq_dir = os.path.join(dataset_root, subject, condition, angle)
    if not os.path.isdir(seq_dir):
        return None

    frame_list = sorted(f for f in os.listdir(seq_dir) if f.lower().endswith(".png"))
    total_frames = len(frame_list)
    if total_frames == 0:
        return None

    # Zero-based index into frame_list (Python convention).
    middle_index = len(frame_list) // 2
    # Human-readable 1-based frame number, for display/reporting ONLY -
    # never used to index into frame_list.
    display_frame_number = middle_index + 1

    selected_filename = frame_list[middle_index]
    full_path = os.path.join(seq_dir, selected_filename)

    return {
        "Subject": subject,
        "Condition": condition,
        "Angle": angle,
        "Total Frames": total_frames,
        "Selected Frame Index": middle_index,           # 0-based (Python convention)
        "Display Frame Number": display_frame_number,   # 1-based, human-readable only
        "Selected Filename": selected_filename,
        "Full Path": full_path,
    }


# ============================================================
# Build the full index
# ============================================================
index_rows = []
missing_sequences = []

for subject in SUBJECTS:
    for condition in CONDITIONS:
        for angle in ANGLES:
            row = index_sequence(DATASET_ROOT, subject, condition, angle)
            if row is None:
                missing_sequences.append((subject, condition, angle))
            else:
                index_rows.append(row)

df_index = pd.DataFrame(
    index_rows,
    columns=["Subject", "Condition", "Angle", "Total Frames",
             "Selected Frame Index", "Display Frame Number",
             "Selected Filename", "Full Path"],
)

print(f"Indexed {len(df_index)} sequences.")
print(f"Missing sequences: {len(missing_sequences)}")
df_index.head()

# %% [markdown] Cell 7
# ## Step 2B: Dataset Verification
# 
# Summary tables and coverage checks over the index built in Step 2A. Nothing here touches the Hamilton-Jacobi Skeleton pipeline - this is purely dataset bookkeeping.

# %% [code] Cell 8
# ---- Table 1: overall counts ----
num_subjects_found = df_index["Subject"].nunique()
num_sequences = len(df_index)
num_representative_frames = len(df_index)  # exactly one representative frame per sequence

table_1 = pd.DataFrame([{
    "Number of subjects found": num_subjects_found,
    "Number of sequences": num_sequences,
    "Number of representative frames": num_representative_frames,
}])

print("========== TABLE 1: OVERALL COUNTS ==========")
table_1

# %% [code] Cell 9
# ---- Table 2: Subject -> Representative Frames ----
table_2 = (
    df_index.groupby("Subject")
    .size()
    .reset_index(name="Representative Frames")
    .sort_values("Subject")
    .reset_index(drop=True)
)

print("========== TABLE 2: SUBJECT -> REPRESENTATIVE FRAMES ==========")
table_2

# %% [code] Cell 10
# ---- Table 3: Condition -> Representative Frames ----
table_3 = (
    df_index.groupby("Condition")
    .size()
    .reset_index(name="Representative Frames")
    .reset_index(drop=True)
)

print("========== TABLE 3: CONDITION -> REPRESENTATIVE FRAMES ==========")
table_3

# %% [code] Cell 11
# ---- Table 4: Angle -> Representative Frames ----
table_4 = (
    df_index.groupby("Angle")
    .size()
    .reset_index(name="Representative Frames")
    .reset_index(drop=True)
)

print("========== TABLE 4: ANGLE -> REPRESENTATIVE FRAMES ==========")
table_4

# %% [code] Cell 12
# ---- Coverage verification: no missing subject / condition / angle ----
expected_total = len(SUBJECTS) * len(CONDITIONS) * len(ANGLES)

subjects_found = set(df_index["Subject"].unique())
conditions_found = set(df_index["Condition"].unique())
angles_found = set(df_index["Angle"].unique())

subjects_missing_entirely = sorted(set(SUBJECTS) - subjects_found)
conditions_missing_entirely = sorted(set(CONDITIONS) - conditions_found)
angles_missing_entirely = sorted(set(ANGLES) - angles_found)

print("========== COVERAGE VERIFICATION ==========")
print(f"Expected sequences (subjects x conditions x angles): {expected_total}")
print(f"Sequences actually indexed                          : {num_sequences}")
print(f"Sequences missing (directory absent or empty)        : {len(missing_sequences)}")
print()
print(f"No missing subject   : {len(subjects_missing_entirely) == 0}")
if subjects_missing_entirely:
    print(f"   Subjects with ZERO sequences found: {subjects_missing_entirely}")

print(f"No missing condition : {len(conditions_missing_entirely) == 0}")
if conditions_missing_entirely:
    print(f"   Conditions with ZERO sequences found: {conditions_missing_entirely}")

print(f"No missing angle     : {len(angles_missing_entirely) == 0}")
if angles_missing_entirely:
    print(f"   Angles with ZERO sequences found: {angles_missing_entirely}")

if missing_sequences:
    print()
    print(f"First 10 missing (subject, condition, angle) combinations (of {len(missing_sequences)}):")
    for combo in missing_sequences[:10]:
        print("  ", combo)

# %% [markdown] Cell 13
# ## Step 2C: Hamilton Skeleton Validation
# 
# For every representative frame indexed in Step 2A, this loop:
# 1. Loads the image.
# 2. Calls **only** `run_hamilton_pipeline(image)` from `phase1_hamilton` - no algorithm is reimplemented here.
# 3. Saves the binary silhouette and Hamilton Skeleton, preserving the CASIA-B hierarchy under `Output/<subject>/<condition>/<angle>/`, with output filenames that **preserve the original frame filename**:
# 
# ```
# Output/<subject>/<condition>/<angle>/<original_filename>_binary.png
# Output/<subject>/<condition>/<angle>/<original_filename>_skeleton.png
# ```
# 
# e.g. for selected frame `001-nm-01-000-038.png`, the outputs are `001-nm-01-000-038_binary.png` and `001-nm-01-000-038_skeleton.png`.
# 
# 4. Records per-image integrity checks (size match, value-set checks) **on the raw pipeline outputs**, before any visualization-only scaling is applied for the saved PNG.
# 
# **Note on the saved skeleton PNG:** `hamilton_skeleton` from the pipeline is a boolean array (`{False, True}`, i.e. `{0, 1}` when cast to an integer type). For the *saved PNG file* only, it is scaled to `{0, 255}` so it is visible in a normal image viewer (a `{0,1}`-valued PNG would render as almost solid black). The `{0,1}` integrity check in Step 2D is performed on the **raw returned array**, not on the saved, scaled PNG - this is stated explicitly to avoid confusion between the two.

# %% [code] Cell 14
def save_binary_and_skeleton(output_root, subject, condition, angle, original_filename, binary_silhouette, hamilton_skeleton):
    """
    Save the binary silhouette and Hamilton Skeleton for one sequence,
    preserving the CASIA-B hierarchy under `output_root`, with output
    filenames that PRESERVE the original frame filename:

        output_root/<subject>/<condition>/<angle>/<original_filename>_binary.png
        output_root/<subject>/<condition>/<angle>/<original_filename>_skeleton.png

    Parameters
    ----------
    original_filename : str
        The selected frame's filename WITHOUT its extension
        (e.g. "001-nm-01-000-038" from "001-nm-01-000-038.png").

    The binary silhouette is saved exactly as returned by the Phase 1
    pipeline. The Hamilton Skeleton (a boolean {0,1} array) is scaled to
    {0,255} ONLY for this saved PNG, purely so it is visible in a normal
    image viewer - see the markdown note above.

    Returns
    -------
    (binary_path, skeleton_path) : the full paths written.
    """
    out_dir = os.path.join(output_root, subject, condition, angle)
    os.makedirs(out_dir, exist_ok=True)

    binary_path = os.path.join(out_dir, f"{original_filename}_binary.png")
    skeleton_path = os.path.join(out_dir, f"{original_filename}_skeleton.png")

    Image.fromarray(binary_silhouette.astype(np.uint8)).save(binary_path)

    skeleton_uint8_for_viewing = (hamilton_skeleton.astype(np.uint8) * 255)
    Image.fromarray(skeleton_uint8_for_viewing).save(skeleton_path)

    return binary_path, skeleton_path

# %% [code] Cell 15
# ============================================================
# Main validation loop - calls ONLY run_hamilton_pipeline()
# ============================================================
processing_log = []

_progress_interval = 50

for idx, row in df_index.iterrows():
    subject, condition, angle = row["Subject"], row["Condition"], row["Angle"]
    full_path = row["Full Path"]
    selected_filename = row["Selected Filename"]
    # Original filename WITHOUT its extension, used to name the outputs
    # so that outputs become <original_filename>_binary.png and
    # <original_filename>_skeleton.png (preserving traceability to the
    # exact source frame, instead of generic binary.png/skeleton.png).
    original_filename = os.path.splitext(selected_filename)[0]

    start_time = time.time()
    try:
        raw_image = Image.open(full_path)
        raw_image.load()
        image_array = np.array(raw_image)

        # The ONLY call into the Phase 1 algorithm - nothing is
        # reimplemented in this notebook.
        results = run_hamilton_pipeline(image_array)

        binary_silhouette = results["binary_silhouette"]
        hamilton_skeleton = results["hamilton_skeleton"]

        # --- Integrity checks on the RAW pipeline outputs ---
        size_match = (
            image_array.shape == binary_silhouette.shape == hamilton_skeleton.shape
        )
        binary_unique = set(np.unique(binary_silhouette).tolist())
        binary_values_ok = binary_unique.issubset({0, 255})
        skeleton_unique = set(np.unique(hamilton_skeleton.astype(np.uint8)).tolist())
        skeleton_values_ok = skeleton_unique.issubset({0, 1})

        binary_path, skeleton_path = save_binary_and_skeleton(
            OUTPUT_ROOT, subject, condition, angle, original_filename,
            binary_silhouette, hamilton_skeleton,
        )

        elapsed = time.time() - start_time

        processing_log.append({
            "Subject": subject, "Condition": condition, "Angle": angle,
            "Full Path": full_path,
            "Original Filename": original_filename,
            "Status": "Success", "Error": None,
            "Processing Time (s)": elapsed,
            "Input Shape": image_array.shape,
            "Binary Shape": binary_silhouette.shape,
            "Skeleton Shape": hamilton_skeleton.shape,
            "Size Match": size_match,
            "Binary Dtype": str(binary_silhouette.dtype),
            "Skeleton Dtype": str(hamilton_skeleton.dtype),
            "Binary Values OK {0,255}": binary_values_ok,
            "Skeleton Values OK {0,1}": skeleton_values_ok,
            "Binary Unique Values": sorted(binary_unique),
            "Binary Path": binary_path,
            "Skeleton Path": skeleton_path,
        })

    except Exception as e:
        elapsed = time.time() - start_time
        processing_log.append({
            "Subject": subject, "Condition": condition, "Angle": angle,
            "Full Path": full_path,
            "Original Filename": original_filename,
            "Status": "Failed", "Error": str(e),
            "Processing Time (s)": elapsed,
            "Input Shape": None, "Binary Shape": None, "Skeleton Shape": None,
            "Size Match": None, "Binary Dtype": None, "Skeleton Dtype": None,
            "Binary Values OK {0,255}": None, "Skeleton Values OK {0,1}": None,
            "Binary Unique Values": None,
            "Binary Path": None, "Skeleton Path": None,
        })

    if (idx + 1) % _progress_interval == 0 or (idx + 1) == len(df_index):
        print(f"Processed {idx + 1}/{len(df_index)} sequences...")

df_results = pd.DataFrame(processing_log)
print()
print(f"Loop complete. {len(df_results)} sequences attempted.")

# %% [markdown] Cell 16
# ## Step 2D: Validation Report
# 
# Aggregate statistics, coverage tables, random qualitative samples, and data-integrity verification, all derived from `df_results` (Step 2C) - no further calls into the Phase 1 pipeline are made here.

# %% [code] Cell 17
# ---- Core counts ----
total_processed = len(df_results)
successful = int((df_results["Status"] == "Success").sum())
failed = int((df_results["Status"] == "Failed").sum())

total_processing_time = float(df_results["Processing Time (s)"].sum())
avg_processing_time = (
    float(df_results.loc[df_results["Status"] == "Success", "Processing Time (s)"].mean())
    if successful > 0 else float("nan")
)

print("========== VALIDATION REPORT: CORE STATISTICS ==========")
print(f"Total images processed        : {total_processed}")
print(f"Successful images              : {successful}")
print(f"Failed images                   : {failed}")
print(f"Total processing time (s)      : {total_processing_time:.4f}")
print(f"Average processing time/image  : {avg_processing_time:.4f} s")

# %% [code] Cell 18
# ---- Verification tables: success/failure breakdown ----
print("========== VERIFICATION TABLE: STATUS BY CONDITION ==========")
status_by_condition = (
    df_results.groupby(["Condition", "Status"]).size().unstack(fill_value=0)
)
print(status_by_condition)
print()

print("========== VERIFICATION TABLE: STATUS BY ANGLE ==========")
status_by_angle = (
    df_results.groupby(["Angle", "Status"]).size().unstack(fill_value=0)
)
print(status_by_angle)

if failed > 0:
    print()
    print(f"========== FAILED SEQUENCES ({failed}) ==========")
    print(df_results.loc[df_results["Status"] == "Failed",
                          ["Subject", "Condition", "Angle", "Error"]])

# %% [code] Cell 19
# ---- Random qualitative samples: original / binary / skeleton ----
successful_rows = df_results[df_results["Status"] == "Success"]

_sample_size = min(5, len(successful_rows))
if _sample_size > 0:
    sample_rows = successful_rows.sample(n=_sample_size, random_state=42)

    for _, row in sample_rows.iterrows():
        original = np.array(Image.open(row["Full Path"]))
        binary_img = np.array(Image.open(row["Binary Path"]))
        skeleton_img = np.array(Image.open(row["Skeleton Path"]))

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(binary_img, cmap="gray")
        axes[1].set_title("Binary Silhouette")
        axes[1].axis("off")

        axes[2].imshow(skeleton_img, cmap="gray")
        axes[2].set_title("Hamilton Skeleton")
        axes[2].axis("off")

        fig.suptitle(f"Subject {row['Subject']}  |  {row['Condition']}  |  angle {row['Angle']}")
        plt.tight_layout()
        plt.show()
else:
    print("No successful sequences available to display.")

# %% [code] Cell 20
# ---- Data-integrity verification (aggregated over ALL successful images) ----
size_match_ok = int(successful_rows["Size Match"].sum())
binary_values_ok_count = int(successful_rows["Binary Values OK {0,255}"].sum())
skeleton_values_ok_count = int(successful_rows["Skeleton Values OK {0,1}"].sum())

binary_dtypes_seen = sorted(successful_rows["Binary Dtype"].unique().tolist())
skeleton_dtypes_seen = sorted(successful_rows["Skeleton Dtype"].unique().tolist())

print("========== DATA INTEGRITY VERIFICATION ==========")
print(f"Input size == Output size (binary & skeleton) : {size_match_ok}/{successful} "
      f"({'ALL OK' if size_match_ok == successful else 'SOME MISMATCHES'})")
print(f"Binary Dtype(s) observed                       : {binary_dtypes_seen}")
print(f"Skeleton Dtype(s) observed                      : {skeleton_dtypes_seen}")
print(f"Binary image contains only {{0,255}}            : {binary_values_ok_count}/{successful} "
      f"({'ALL OK' if binary_values_ok_count == successful else 'SOME VIOLATIONS'})")
print(f"Hamilton Skeleton contains only {{0,1}} (raw array): {skeleton_values_ok_count}/{successful} "
      f"({'ALL OK' if skeleton_values_ok_count == successful else 'SOME VIOLATIONS'})")

if binary_values_ok_count != successful:
    print()
    print("Sequences whose binary silhouette used unique values other than a subset of {0,255}:")
    print(successful_rows.loc[~successful_rows["Binary Values OK {0,255}"],
                               ["Subject", "Condition", "Angle", "Binary Unique Values"]])

# %% [markdown] Cell 21
# ### Final Validation Report

# %% [code] Cell 22
print("=" * 60)
print("PHASE 2 - FINAL VALIDATION REPORT")
print("=" * 60)
print(f"Dataset root                  : {DATASET_ROOT}")
print(f"Output root                   : {os.path.abspath(OUTPUT_ROOT)}")
print()
print(f"Subjects covered              : {num_subjects_found} / {len(SUBJECTS)}")
print(f"Conditions covered            : {len(conditions_found)} / {len(CONDITIONS)}")
print(f"Angles covered                : {len(angles_found)} / {len(ANGLES)}")
print(f"Sequences indexed             : {num_sequences} / {expected_total}")
print(f"Sequences missing             : {len(missing_sequences)}")
print()
print(f"Images processed              : {total_processed}")
print(f"Successful                    : {successful}")
print(f"Failed                        : {failed}")
print(f"Total processing time (s)     : {total_processing_time:.4f}")
print(f"Average time / image (s)      : {avg_processing_time:.4f}")
print()
print(f"Size match (input==output)    : {size_match_ok}/{successful}")
print(f"Binary values in {{0,255}}      : {binary_values_ok_count}/{successful}")
print(f"Skeleton values in {{0,1}}      : {skeleton_values_ok_count}/{successful}")
print()
overall_pass = (
    failed == 0
    and size_match_ok == successful
    and binary_values_ok_count == successful
    and skeleton_values_ok_count == successful
)
print(f"OVERALL VALIDATION STATUS     : {'PASS' if overall_pass else 'REVIEW NEEDED'}")
print("=" * 60)
print()
print("No contrastive learning, generative learning, feature extraction, "
      "classification, training, testing, evaluation, pruning, outward "
      "Fast Marching, or deep learning model implemented in this notebook.")
print("Phase 2 (Dataset Pipeline Validation) complete.")

# %% [markdown] Cell 23
# ## =========================================================
# ## MANUAL VISUAL VERIFICATION OF REPRESENTATIVE SKELETONS
# ## =========================================================
# 
# **Purpose:** this section is ONLY for human visual inspection before moving to model development. It is **not** part of training or preprocessing.
# 
# - No previous code in this notebook is modified.
# - The dataframe `df_index`, already built in Step 2A, is reused exactly as-is - the dataset is **not** rescanned or regenerated here.
# - Random sampling is **not** used. A fixed list of 10 representative sequences is inspected, given explicitly below.
# - Nothing is saved to disk in this section - all figures are inline, disposable visualizations only.
# - If a requested sample is not present in `df_index` (e.g. that sequence was missing during Step 2A indexing), a warning is printed and the loop continues with the remaining samples.
# 
# **Fixed samples requested for inspection:**
# 
# | # | Subject | Condition | Angle |
# |---|---|---|---|
# | 1 | 001 | nm-01 | 000 |
# | 2 | 001 | nm-01 | 090 |
# | 3 | 001 | nm-01 | 180 |
# | 4 | 010 | nm-03 | 090 |
# | 5 | 015 | cl-02 | 090 |
# | 6 | 020 | nm-06 | 180 |
# | 7 | 025 | cl-01 | 000 |
# | 8 | 030 | nm-04 | 180 |
# | 9 | 035 | cl-02 | 090 |
# | 10 | 040 | nm-06 | 180 |
# 
# For each sample that is found: load the original representative frame, call **only** `run_hamilton_pipeline()` (nothing reimplemented), and display **Original Frame / Binary Silhouette / Hamilton Skeleton** side-by-side, labeled with subject, condition, angle, and filename.

# %% [code] Cell 24
# ============================================================
# Fixed list of representative sequences for manual inspection.
# Random sampling is explicitly NOT used here - this list is fixed and
# given directly, so the same sequences are inspected every time this
# notebook is run, regardless of dataset ordering or random seed.
# ============================================================
MANUAL_INSPECTION_SAMPLES = [
    ("001", "nm-01", "000"),
    ("001", "nm-01", "090"),
    ("001", "nm-01", "180"),
    ("010", "nm-03", "090"),
    ("015", "cl-02", "090"),
    ("020", "nm-06", "180"),
    ("025", "cl-01", "000"),
    ("030", "nm-04", "180"),
    ("035", "cl-02", "090"),
    ("040", "nm-06", "180"),
]

requested_count = len(MANUAL_INSPECTION_SAMPLES)
displayed_count = 0
missing_samples = []

for subject, condition, angle in MANUAL_INSPECTION_SAMPLES:
    # Look up this exact sequence in the index dataframe already built
    # in Step 2A - the dataset is NOT rescanned or regenerated here.
    match = df_index[
        (df_index["Subject"] == subject)
        & (df_index["Condition"] == condition)
        & (df_index["Angle"] == angle)
    ]

    if match.empty:
        print(f"WARNING: requested sample not found in the dataset index - "
              f"Subject {subject} | {condition} | angle {angle}. Skipping.")
        missing_samples.append((subject, condition, angle))
        continue

    row = match.iloc[0]
    full_path = row["Full Path"]
    selected_filename = row["Selected Filename"]

    # Load the original representative frame. Nothing is saved anywhere
    # in this section.
    raw_image = Image.open(full_path)
    raw_image.load()
    image_array = np.array(raw_image)

    # The ONLY call into the Phase 1 algorithm - reused exactly as in
    # Step 2C, nothing reimplemented here.
    results = run_hamilton_pipeline(image_array)
    binary_silhouette = results["binary_silhouette"]
    hamilton_skeleton = results["hamilton_skeleton"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(image_array, cmap="gray")
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(binary_silhouette, cmap="gray")
    axes[1].set_title("Binary Silhouette")
    axes[1].axis("off")

    axes[2].imshow(hamilton_skeleton, cmap="gray")
    axes[2].set_title("Hamilton Skeleton")
    axes[2].axis("off")

    fig.suptitle(f"Subject {subject}  |  {condition}  |  angle {angle}  |  {selected_filename}")
    plt.tight_layout()
    plt.show()

    displayed_count += 1

print()
print("=" * 60)
print("MANUAL VISUAL INSPECTION SUMMARY")
print("=" * 60)
print(f"Requested samples : {requested_count}")
print(f"Displayed samples : {displayed_count}")
print(f"Missing samples   : {len(missing_samples)}")
if missing_samples:
    for s in missing_samples:
        print(f"   Missing: Subject {s[0]} | {s[1]} | angle {s[2]}")
print()
print("Visual inspection completed successfully.")
print("The displayed representative Hamilton Skeletons can be manually "
      "inspected before proceeding to Phase 3.")

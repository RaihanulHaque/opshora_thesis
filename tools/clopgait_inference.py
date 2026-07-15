# -*- coding: utf-8 -*-
"""
Standalone gait-recognition inference script (block-by-block, Jupytext
percent format -- opens as native cells in VSCode/Jupyter/Kaggle).
"""

# %% [markdown] Cell 1
# # CLoP-Gait / V6 Inference
#
# Given a **trained checkpoint** and a **paired skeleton + silhouette
# dataset** (same `subject/condition/view/*.png` directory layout used
# everywhere else in this repo -- e.g. `CLoPGaitHamiltonSkeleton` +
# `CLoPGaitSilhouettes`, or `CASIA_B_Hamilton_Skeleton` + `GaitDatasetB-silh`),
# this script:
#
# 1. Builds/reuses a local `.npz` preprocessing cache (same
#    `gait.preprocessing.prepare_dataset()` used for training -- so results
#    are directly comparable to `runs/MODEL_COMPARISON.md`).
# 2. Loads the model architecture from `designs/<design_name>/model.py` and
#    the checkpoint's weights.
# 3. Runs identification/verification metrics (Rank-1, Rank-5, verification
#    AUC -- `gait.train.evaluate()`, the exact function used during training).
# 4. Shows a handful of probe sequences next to their top-1 gallery match,
#    block by block.
#
# **This is a single, self-contained `.py` file on purpose** (not `.ipynb`):
# it opens as native cells in VSCode/Jupyter/Kaggle via the `# %%` markers,
# without notebook-JSON token overhead.
#
# **To run this elsewhere (Kaggle, another PC):** you need this repo's
# `gait/` and `designs/` folders importable (attach as a Kaggle dataset, git
# clone, or copy alongside this script), plus the checkpoint file and the
# two dataset folders. Edit the CONFIGURATION cell below, then run cells
# top to bottom.

# %% [markdown] Cell 2
# ## 1. Configuration
#
# Edit these paths for wherever this is running. Everything else in this
# script derives from them.

# %% [code] Cell 3
from pathlib import Path

# Repo root containing gait/ and designs/ -- e.g. on Kaggle this might be
# "/kaggle/input/opshora-thesis-code" if you attached the repo as a dataset.
REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()

# Trained model weights: either a plain state_dict (e.g. best_rank1_model.pt)
# or a full training checkpoint dict with a "model" key -- both are handled below.
CHECKPOINT_PATH = Path("runs/fusion_rank1_002/best_rank1_model.pt")

# Paired dataset folders (subject/condition/view/*.png layout).
SKELETON_DATASET_PATH = Path("datasets/CLoPGaitHamiltonSkeleton")
SILHOUETTE_DATASET_PATH = Path("datasets/CLoPGaitSilhouettes")

# Which model architecture to instantiate -- must match the checkpoint's
# training design (designs/<DESIGN_NAME>/model.py must export build_model()).
DESIGN_NAME = "skeleton_silhouette_fusion_v6"

# Local cache for the preprocessed .npz tensors (built once, reused after).
CACHE_DIR = Path("/tmp/clopgait_inference_cache")

# Which subjects go to which split, and how:
#   "domain" -> train = every domain except TEST_DOMAIN_SUFFIX, test = only
#               TEST_DOMAIN_SUFFIX (same subjects in both -- CLoP-Gait's
#               indoor+outdoor-night -> outdoor-day generalization protocol).
#   "subject" -> classic CASIA-B-style: first TRAIN_SUBJECTS subjects train,
#                the rest test (unseen identities).
SPLIT_MODE = "domain"
TEST_DOMAIN_SUFFIX = "od"
TRAIN_SUBJECTS = 100  # only used when SPLIT_MODE == "subject"
EVAL_SPLIT = "test"   # which split to run inference/evaluation over

# Height/width/sequence_length must match what the checkpoint was trained
# with (V6 default: 64x64, 30 frames) -- these are NOT the dataset's raw
# storage resolution, they're the training tensor shape.
HEIGHT = 64
WIDTH = 64
SEQUENCE_LENGTH = 30
EVAL_GALLERY_PER_SUBJECT = 3

# Model hyperparameters -- these change the model's actual architecture
# (layer sizes), so they MUST match the checkpoint's original training
# config exactly (see designs/<design>/config.json for the run that
# produced CHECKPOINT_PATH), unlike num_classes below which is expected to
# differ across datasets. Defaults here match V6's own config.
HIDDEN_DIM = 176
EMBEDDING_DIM = 256
PROJECTION_DIM = 128

NUM_VISUAL_EXAMPLES = 6  # how many probe sequences to visualize in Cell 13

# %% [markdown] Cell 4
# ## 2. Imports
#
# Adds REPO_ROOT to sys.path so gait/ and designs/ import regardless of
# where this script physically lives.

# %% [code] Cell 5
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import importlib

from gait.config import ExperimentConfig
from gait.preprocessing import prepare_dataset
from gait.dataset import GaitSequenceDataset
from gait.train import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown] Cell 6
# ## 3. Build the preprocessing config and cache
#
# `prepare_dataset()` is the exact same function used to build every
# training cache in this repo -- it pairs skeleton + silhouette frames by
# subject/condition/view, applies `clean_and_align()`, and writes `.npz`
# tensors once. Safe to re-run: it skips work if `CACHE_DIR/COMPLETE.json`
# already exists (pass `force=True` below to rebuild).

# %% [code] Cell 7
config = ExperimentConfig(
    design_name=DESIGN_NAME,
    run_name="inference",
    # ExperimentConfig's default experiments_root ("/data/experiments") is a
    # Modal-only path; prepare_dataset() creates it even though inference
    # doesn't need experiment-output artifacts, so point it somewhere local.
    experiments_root=str(CACHE_DIR.parent / "experiments"),
    dataset_format="skeleton_silhouette_fusion",
    dataset_path=str(SKELETON_DATASET_PATH),
    silhouette_dataset_path=str(SILHOUETTE_DATASET_PATH),
    cache_dir=str(CACHE_DIR),
    height=HEIGHT,
    width=WIDTH,
    sequence_length=SEQUENCE_LENGTH,
    split_mode=SPLIT_MODE,
    test_domain_suffix=TEST_DOMAIN_SUFFIX,
    train_subjects=TRAIN_SUBJECTS,
    eval_gallery_per_subject=EVAL_GALLERY_PER_SUBJECT,
    hidden_dim=HIDDEN_DIM,
    embedding_dim=EMBEDDING_DIM,
    projection_dim=PROJECTION_DIM,
)

summary = prepare_dataset(config, force=False)
print(f"Prepared dataset: {summary}")

# %% [markdown] Cell 8
# ## 4. Load the evaluation split

# %% [code] Cell 9
eval_data = GaitSequenceDataset(
    config.cache_dir, EVAL_SPLIT, config.train_subjects, config.split_mode, config.test_domain_suffix
)
eval_loader = DataLoader(eval_data, batch_size=max(1, len(eval_data)), num_workers=0)
print(f"{EVAL_SPLIT} split: {len(eval_data)} sequences, {len(eval_data.subject_to_label)} subjects")

# %% [markdown] Cell 10
# ## 5. Build the model and load checkpoint weights
#
# The classifier head's size depends on how many identities the *original
# training run* saw, which may not match `len(eval_data.subject_to_label)`
# here (e.g. a different dataset, or a subject-count split vs domain
# split). That's fine for inference: `evaluate()` only ever reads the
# model's *embedding* output, never the classifier logits, so a classifier
# shape mismatch is not a problem -- we load with `strict=True` first and
# fall back to dropping only the shape-mismatched keys (typically just the
# classifier) if needed. Note: torch's own strict=False only tolerates
# missing/extra keys, NOT shape mismatches on keys present in both, so a
# plain strict=False retry still raises here -- keys have to be filtered
# manually first.

# %% [code] Cell 11
design_module = importlib.import_module(f"designs.{DESIGN_NAME}.model")
model = design_module.build_model(config, len(eval_data.subject_to_label)).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

try:
    model.load_state_dict(state_dict, strict=True)
    print("Loaded checkpoint weights (strict=True, exact match).")
except RuntimeError as error:
    print(f"strict=True load failed ({error}); retrying after dropping shape-mismatched keys "
          "(embeddings still load fine, only the classifier head differs when num_classes differs).")
    model_shapes = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_shapes and model_shapes[key].shape == value.shape
    }
    dropped = sorted(set(state_dict) - set(compatible))
    print(f"Dropping {len(dropped)} shape-mismatched key(s): {dropped}")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(state_dict)} keys. missing={missing} unexpected={unexpected}")

model.eval()

# %% [markdown] Cell 12
# ## 6. Run evaluation: Rank-1 / Rank-5 / verification AUC
#
# Same metrics, same computation (`gait.train.evaluate`) as
# `runs/MODEL_COMPARISON.md` -- directly comparable to every other run in
# this thesis.

# %% [code] Cell 13
metrics = evaluate(model, eval_loader, device, gallery_per_subject=config.eval_gallery_per_subject)

print("=" * 50)
print(f"{'Metric':<24}{'Value':>10}")
print("-" * 50)
for name in ("rank1", "rank5", "verification_auc", "verification_accuracy", "distance_gap"):
    if name in metrics:
        value = metrics[name]
        formatted = f"{value * 100:.2f}%" if name != "distance_gap" else f"{value:.3f}"
        print(f"{name:<24}{formatted:>10}")
print("=" * 50)

# %% [markdown] Cell 14
# ## 7. Visualize a few probe sequences and their top-1 gallery match
#
# Rebuilds the same gallery/probe split `evaluate()` uses internally (gallery
# = each subject's "normal"-condition sequences, probe = everything else),
# then shows each probe's middle frame (silhouette + topology) next to its
# closest gallery match by cosine similarity.

# %% [code] Cell 15
def _condition_family(condition: str) -> str:
    condition = condition.lower()
    if condition.startswith("nm") or condition.startswith("fn"):
        return "normal"
    if condition.startswith("cl"):
        return "clothing"
    return condition


with torch.no_grad():
    embeddings, subjects, conditions, silhouettes = [], [], [], []
    for batch in eval_loader:
        output = model(batch["silhouette"].to(device), batch["topology"].to(device))
        embeddings.append(F.normalize(output["embedding"], dim=1).cpu())
        subjects.extend(batch["subject"])
        conditions.extend(batch["condition_family"])
        silhouettes.append(batch["silhouette"])
    matrix = torch.cat(embeddings)
    silhouette_stack = torch.cat(silhouettes)

gallery_indices, probe_indices = [], []
for subject in sorted(set(subjects)):
    candidates = [i for i, value in enumerate(subjects) if value == subject]
    normal = [i for i in candidates if conditions[i] == "normal"]
    preferred = normal if normal else candidates
    galleries = preferred[: max(1, min(EVAL_GALLERY_PER_SUBJECT, len(preferred)))]
    gallery_indices.extend(galleries)
    gallery_set = set(galleries)
    probe_indices.extend(i for i in candidates if i not in gallery_set)

similarities = matrix[probe_indices] @ matrix[gallery_indices].T
best_gallery = similarities.argmax(dim=1)

# Spread examples across distinct probe subjects (not just the first
# subject's sequences) for a more representative visualization.
seen_subjects: set[str] = set()
selected_positions: list[int] = []
for position, probe_index in enumerate(probe_indices):
    subject = subjects[probe_index]
    if subject in seen_subjects:
        continue
    seen_subjects.add(subject)
    selected_positions.append(position)
    if len(selected_positions) >= NUM_VISUAL_EXAMPLES:
        break

sample_count = len(selected_positions)
fig, axes = plt.subplots(sample_count, 2, figsize=(8, 3.2 * sample_count))
if sample_count == 1:
    axes = axes[None, :]

for row, probe_position in enumerate(selected_positions):
    probe_index = probe_indices[probe_position]
    gallery_index = gallery_indices[best_gallery[probe_position].item()]
    mid_frame = silhouette_stack.shape[1] // 2

    probe_image = silhouette_stack[probe_index, mid_frame, 0].numpy()
    gallery_image = silhouette_stack[gallery_index, mid_frame, 0].numpy()
    correct = subjects[probe_index] == subjects[gallery_index]

    axes[row, 0].imshow(probe_image, cmap="gray")
    axes[row, 0].set_title(f"Probe: subject {subjects[probe_index]}", fontsize=10)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(gallery_image, cmap="gray")
    marker = "correct" if correct else "WRONG"
    axes[row, 1].set_title(f"Top-1 gallery: subject {subjects[gallery_index]} ({marker})", fontsize=10)
    axes[row, 1].axis("off")

fig.tight_layout()
plt.show()

# %% [markdown] Cell 16
# Done. Re-run Cell 3 with different `CHECKPOINT_PATH`/dataset paths to
# evaluate a different model or dataset without touching anything else.

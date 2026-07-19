# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python gait-recognition thesis prototype. Shared code lives in `gait/`: configuration, dataset loading, preprocessing, losses, model definitions, and training helpers. Model iterations and experiment-specific implementations live under `designs/`, with the current best-performing architecture in `designs/skeleton_silhouette_partset_v7/` (part-set fusion, successor to `designs/skeleton_silhouette_fusion_v6/`) and comparison baselines (LSTM/TCN/Transformer variants, plus three
faithful replications of published architectures for external validity —
`designs/paper_3dlocal_v1/`, `designs/paper_cstl_v1/`,
`designs/paper_smplgait_v1/`, each with a `README.md` documenting exactly
what was replicated vs. adapted) alongside them. Tests are in `tests/`, local analysis and camera utilities and the CLoP-Gait dataset-authoring pipeline (`tools/dataset_pipeline/`: raw video -> YOLO segmentation -> silhouette -> Hamilton-Jacobi skeleton via `opshora_archive/phase1_hamilton.py`'s `run_hamilton_pipeline()` (needs `scikit-fmm`), with a QC report) are in `tools/`, Modal entry points are `modal_app.py`, `submit_modal.py` (single run), and `submit_modal_parallel.py` (spawns multiple designs/runs in parallel containers), and experiment outputs are kept under `runs/` (see `runs/MODEL_COMPARISON.md` for the honest, real-numbers cross-model comparison). Large local data and paper assets are stored in `datasets/` and `papers/`; avoid committing generated caches or heavyweight artifacts unless they are intentional thesis evidence.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the smoke test suite:

```bash
python -m pytest tests/test_smoke.py tests/test_dataset_pipeline.py
```

Note: `pytest` is not installed in this machine's `ML` conda environment. If it's unavailable, load and run test functions directly instead:

```bash
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('m', 'tests/test_smoke.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for name in dir(m):
    if name.startswith('test_'):
        getattr(m, name)()
        print(name, 'OK')
"
```

Deploy the Modal app after code or config changes:

```bash
modal deploy modal_app.py
```

Submit the current V6 experiment:

```bash
python submit_modal.py run --design skeleton_silhouette_fusion_v6 --run fusion_rank1_003
```

Run local post-training analysis with the ML conda Python shown in `README.md` when evaluating checkpoints.

Quantitatively evaluate the masked-reconstruction ("generative") branch of any design that has decoder hooks (Dice/IoU/precision/recall/BCE for the skeleton channel, MSE/MAE/PSNR/SSIM for the motion channel, computed on the held-out test split with the same masking used in training):

```bash
python tools/reconstruction_quality.py --run-dir runs/<design>/<run> \
  --skeleton-dataset <path> --silhouette-dataset <path> --cache-dir <path>
```

`gait/dataset.py`/`gait/config.py` support two extra split modes beyond the
original subject-disjoint split, plus an optional true validation split:
- `split_mode: "domain"` — CLoP-Gait domain-generalization (train
  indoor+outdoor-night, test outdoor-day), via `test_domain_suffix`.
- `split_mode: "condition"` — cross-condition generalization (train one
  condition family, e.g. `nm`, test another, e.g. `cl`), via
  `train_condition_prefixes`/`test_condition_prefixes` and optionally
  `train_domain_suffixes`/`test_domain_suffixes` (CLoP-Gait conditions encode
  a domain suffix; CASIA-B's don't, so leave those empty for CASIA-B).
- `validation_subjects: N` (only with `split_mode: "subject"`) carves the
  last `N` subjects out of `train_subjects` into a true validation split;
  early stopping then monitors validation instead of test (`gait/train.py`
  logs test metrics unprefixed as before and validation metrics under
  `val_*` keys). Defaults to `0`, which reproduces the old test-as-monitor
  behavior exactly — see `SUPERVISOR_QA_NOTES.md` Q4 for the full rationale.

Build/inspect the CLoP-Gait custom dataset (raw videos live in `datasets/CLoP-Gait/`, organized as `INDOOR|OUTDOOR/Outdoor_Day|OUTDOOR/Outdoor_Night`):

```bash
python -m tools.dataset_pipeline.pipeline --subjects 001 --conditions nm-01-id --views 000 --stage all
python -m tools.dataset_pipeline.qc
```

`pipeline.py` is resumable (skips sequences whose `_SEGMENT_COMPLETE.json`/`_SKELETON_COMPLETE.json` marker already exists, unless `--force`) and supports `--sample-fps` for fast sanity checks. `qc.py` is read-only and safe to run at any point mid-batch; it writes `tools/dataset_pipeline/output/qc/qc_report.{json,md}` and a visual sample grid.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, type hints where they clarify data shapes or paths, and descriptive snake_case names for functions, variables, and modules. Keep model classes in PascalCase. Prefer small, testable helpers in `gait/` for shared behavior; keep design-specific variations inside the relevant `designs/<name>/` package. Do not mix unrelated experiment refactors into thesis result updates.

## Testing Guidelines

Tests use `pytest` and currently emphasize smoke coverage for preprocessing, loss functions, model tensor shapes, nested zip reading, and reconstruction previews. Add tests as `tests/test_*.py` and name functions `test_*`. For model changes, verify tensor shapes, finite losses, and minimal synthetic inputs so tests remain fast and independent of full datasets.

## Commit & Pull Request Guidelines

Recent commits use imperative, descriptive messages such as `Add post-training analysis scripts and results...` or `Remove obsolete thesis documents...`. Keep commits focused and mention affected experiments, datasets, or reports when relevant. Pull requests should include a short summary, commands run, linked thesis issue or objective if available, and screenshots or report paths for visual outputs and evaluation changes.

## Security & Configuration Tips

Keep local datasets, Modal credentials, checkpoints, and generated caches out of source control unless explicitly required. Use `experiment.example.json` as a template for shareable configuration, and document any required local paths in README-style docs rather than hard-coding private machine state.

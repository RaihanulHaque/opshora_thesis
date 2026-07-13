# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python gait-recognition thesis prototype. Shared code lives in `gait/`: configuration, dataset loading, preprocessing, losses, model definitions, and training helpers. Model iterations and experiment-specific implementations live under `designs/`, with the current fusion architecture in `designs/skeleton_silhouette_fusion_v6/`. Tests are in `tests/`, local analysis and camera utilities are in `tools/`, Modal entry points are `modal_app.py` and `submit_modal.py`, and experiment outputs are kept under `runs/`. Large local data and paper assets are stored in `datasets/` and `papers/`; avoid committing generated caches or heavyweight artifacts unless they are intentional thesis evidence.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the smoke test suite:

```bash
python -m pytest tests/test_smoke.py
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

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, type hints where they clarify data shapes or paths, and descriptive snake_case names for functions, variables, and modules. Keep model classes in PascalCase. Prefer small, testable helpers in `gait/` for shared behavior; keep design-specific variations inside the relevant `designs/<name>/` package. Do not mix unrelated experiment refactors into thesis result updates.

## Testing Guidelines

Tests use `pytest` and currently emphasize smoke coverage for preprocessing, loss functions, model tensor shapes, nested zip reading, and reconstruction previews. Add tests as `tests/test_*.py` and name functions `test_*`. For model changes, verify tensor shapes, finite losses, and minimal synthetic inputs so tests remain fast and independent of full datasets.

## Commit & Pull Request Guidelines

Recent commits use imperative, descriptive messages such as `Add post-training analysis scripts and results...` or `Remove obsolete thesis documents...`. Keep commits focused and mention affected experiments, datasets, or reports when relevant. Pull requests should include a short summary, commands run, linked thesis issue or objective if available, and screenshots or report paths for visual outputs and evaluation changes.

## Security & Configuration Tips

Keep local datasets, Modal credentials, checkpoints, and generated caches out of source control unless explicitly required. Use `experiment.example.json` as a template for shareable configuration, and document any required local paths in README-style docs rather than hard-coding private machine state.

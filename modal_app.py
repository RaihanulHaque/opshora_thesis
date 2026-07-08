from __future__ import annotations

import modal


APP_NAME = "hj-topogait-training"
VOLUME_NAME = "gait-datasets-store"
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "numpy>=1.26,<3",
        "scipy>=1.12,<2",
        "opencv-python-headless>=4.9,<5",
        "matplotlib>=3.8,<4",
    )
    .add_local_dir("gait", remote_path="/root/gait", copy=True)
    .add_local_dir("designs", remote_path="/root/designs", copy=True)
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    cpu=2.0,
    memory=8192,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=1.0),
    volumes={VOLUME_PATH: volume},
    single_use_containers=True,
)
def preprocess(config_values: dict | None = None, force: bool = False) -> dict:
    from gait.config import ExperimentConfig
    from gait.preprocessing import prepare_dataset

    config = ExperimentConfig.from_dict(config_values or {})
    result = prepare_dataset(config, force=force)
    volume.commit()
    return result


@app.function(
    cpu=2.0,
    memory=8192,
    gpu=["L4", "A10", "T4"],
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=4, initial_delay=2.0, backoff_coefficient=2.0),
    volumes={VOLUME_PATH: volume},
    single_use_containers=True,
)
def train(config_values: dict | None = None) -> dict:
    from gait.config import ExperimentConfig
    from gait.train import train_experiment

    config = ExperimentConfig.from_dict(config_values or {})
    return train_experiment(config, commit_callback=volume.commit)

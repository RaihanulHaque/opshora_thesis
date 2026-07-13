"""Model factory frozen for the first HJ-TopoGait trial.

Create a new sibling design folder instead of editing this file after recording
experimental results.
"""

from gait.model import HJTopoGait


def build_model(config, num_classes: int) -> HJTopoGait:
    return HJTopoGait(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
    )

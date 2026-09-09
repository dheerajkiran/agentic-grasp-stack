"""Perception constants: model choice and detection thresholds.

Kept separate from scene_config.py -- that file is physical-scene constants
(table/robot/camera geometry), this one is model/inference config.
"""

from dataclasses import dataclass

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@dataclass(frozen=True)
class PerceptionConfig:
    box_threshold: float = 0.3
    text_threshold: float = 0.25


DEFAULT_OUTPUT_PATH = "outputs/perception_check.png"

PERCEPTION_CONFIG = PerceptionConfig()

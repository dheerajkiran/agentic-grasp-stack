"""Phase 2 slice 3: the actual Perception agent's detection cycle.

Unlike scripts/run_perception_demo.py (which matches detections against
env.get_object_states() purely to validate accuracy by eye), this module
builds its output ONLY from what the detector + geometry actually produce --
no ground truth, no oracle positions. This is what a real agent is allowed to
know: an image and a known camera/table model, nothing else.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agentic_grasp_stack.config.perception_config import (
    GROUNDING_DINO_MODEL_ID,
    PERCEPTION_CONFIG,
)
from agentic_grasp_stack.config.scene_config import (
    CAMERA_CONFIG,
    CUBE_HALF_EXTENT,
    OBJECT_COLORS,
    CameraConfig,
)
from agentic_grasp_stack.perception.detector import build_prompt, detect
from agentic_grasp_stack.perception.geometry import pixel_to_world

SCHEMA_VERSION = "1.0"


def run_perception_cycle(
    image: np.ndarray,
    surface_z: float,
    object_names: list[str] | None = None,
    box_threshold: float = PERCEPTION_CONFIG.box_threshold,
    text_threshold: float = PERCEPTION_CONFIG.text_threshold,
    camera_config: CameraConfig = CAMERA_CONFIG,
) -> dict:
    """Run detection + 3D pose estimation on `image` and return a JSON-ready
    dict. No ground truth involved: `object_names` is a known vocabulary
    (what classes the detector knows to look for), not per-instance truth.
    `surface_z` is the table's calibrated height (a known scene constant,
    same role it plays in pixel_to_world elsewhere), not object state.
    """
    names = object_names if object_names is not None else list(OBJECT_COLORS)
    prompt = build_prompt(names)

    detections = detect(image, prompt, box_threshold, text_threshold)
    target_height = surface_z + CUBE_HALF_EXTENT

    records = []
    for box, score, label in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        x0, y0, x1, y1 = box
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        x, y, z = pixel_to_world(center, camera_config, target_height)
        records.append(
            {
                "label": label,
                "score": score,
                "position": {"x": x, "y": y, "z": z},
                "box_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": GROUNDING_DINO_MODEL_ID,
        "device": detections["device"],
        "inference_seconds": detections["seconds"],
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "prompt": prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "detections": records,
    }


def write_detections_json(result: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return output_path

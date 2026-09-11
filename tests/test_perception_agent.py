"""Schema/shape test for perception/agent.py's run_perception_cycle -- no
PyBullet, no real Grounding DINO weights (monkeypatches detect() with a
canned result), mirrors test_perception_geometry.py's no-heavy-deps style."""

import json

import numpy as np

from agentic_grasp_stack.perception import agent as agent_module


def _fake_detect(image, prompt, box_threshold, text_threshold):
    return {
        "scores": [0.9],
        "boxes": [[100.0, 100.0, 140.0, 140.0]],
        "labels": ["red cube"],
        "device": "mps",
        "seconds": 0.01,
    }


def test_run_perception_cycle_schema_and_no_ground_truth(monkeypatch):
    monkeypatch.setattr(agent_module, "detect", _fake_detect)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = agent_module.run_perception_cycle(image, surface_z=-0.024)

    assert result["schema_version"] == "1.0"
    assert result["image_width"] == 640 and result["image_height"] == 480
    assert len(result["detections"]) == 1
    det = result["detections"][0]
    assert det["label"] == "red cube"
    assert det["score"] == 0.9
    assert set(det["position"]) == {"x", "y", "z"}
    assert set(det["box_px"]) == {"x0", "y0", "x1", "y1"}
    json.dumps(result)  # must be JSON-serializable


def test_run_perception_cycle_accepts_custom_object_names(monkeypatch):
    monkeypatch.setattr(agent_module, "detect", _fake_detect)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = agent_module.run_perception_cycle(
        image, surface_z=-0.024, object_names=["red_cube"]
    )

    assert result["prompt"] == "red cube."

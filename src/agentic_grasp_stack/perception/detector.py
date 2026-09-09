"""Grounding DINO wrapper: build a text prompt from known object names and run
zero-shot open-vocabulary detection on a rendered scene image.

Model/processor are lazy module-level singletons so repeated detect() calls in
one process don't reload weights. Device placement uses device_map="auto" (per
the documented transformers usage), which falls back to CPU if MPS reports
unavailable -- no hard failure regardless of MPS quirks on this machine.
"""

import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from agentic_grasp_stack.config.perception_config import GROUNDING_DINO_MODEL_ID

_processor = None
_model = None


def _get_processor_and_model():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL_ID, device_map="auto"
        )
    return _processor, _model


def build_prompt(object_names: list[str]) -> str:
    """"red_cube", "blue_cube" -> "red cube. blue cube." (Grounding DINO's expected format)."""
    phrases = [name.replace("_", " ").lower() for name in object_names]
    return ". ".join(phrases) + "."


def detect(
    image: np.ndarray, prompt: str, box_threshold: float, text_threshold: float
) -> dict:
    """Run Grounding DINO on an RGB image (H, W, 3) uint8 array.

    Returns a dict with "scores" (list[float]), "boxes" (list of [x0,y0,x1,y1]
    in pixel coords), "labels" (list[str]), "device" (str), "seconds" (float).
    """
    processor, model = _get_processor_and_model()
    pil_image = Image.fromarray(image)

    start = time.monotonic()
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.monotonic() - start

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(pil_image.height, pil_image.width)],
    )[0]

    return {
        "scores": [float(s) for s in results["scores"]],
        "boxes": [[float(v) for v in box] for box in results["boxes"]],
        "labels": list(results["text_labels"]),
        "device": str(model.device),
        "seconds": elapsed,
    }

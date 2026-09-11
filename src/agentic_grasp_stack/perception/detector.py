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

NMS_IOU_THRESHOLD = 0.5


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    if inter_area == 0.0:
        return 0.0

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _non_max_suppression(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> tuple[list[list[float]], list[float], list[str]]:
    """Collapse overlapping duplicate detections (same object, multiple
    decoder queries) to the single highest-scoring box per cluster. Grounding
    DINO's own post-processing only filters by score, it doesn't suppress
    overlapping boxes -- that's the caller's job, and without it the agent's
    output can contain near-duplicate/garbled-label detections of the same
    physical object (observed in practice, not hypothetical)."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        if all(_box_iou(boxes[i], boxes[j]) < iou_threshold for j in keep):
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]


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

    boxes = [[float(v) for v in box] for box in results["boxes"]]
    scores = [float(s) for s in results["scores"]]
    labels = list(results["text_labels"])
    boxes, scores, labels = _non_max_suppression(boxes, scores, labels)

    return {
        "scores": scores,
        "boxes": boxes,
        "labels": labels,
        "device": str(model.device),
        "seconds": elapsed,
    }

"""Drawing detections onto images."""

import numpy as np
from PIL import Image, ImageDraw

BOX_COLOR = (255, 0, 0)


def draw_detections(image: np.ndarray, detections: dict) -> Image.Image:
    pil_image = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    for box, score, label in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=2)
        draw.text((x0, max(0, y0 - 12)), f"{label} {score:.2f}", fill=BOX_COLOR)
    return pil_image

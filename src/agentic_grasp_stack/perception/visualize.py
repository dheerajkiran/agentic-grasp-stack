"""Drawing detections onto images, and a ground-truth world->pixel forward
projection used only as a textual sanity check (not the 2D box -> 3D pose
un-projection, which is deferred to a later slice)."""

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from agentic_grasp_stack.config.scene_config import CameraConfig

BOX_COLOR = (255, 0, 0)
GT_MARKER_COLOR = (0, 255, 0)


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


def world_to_pixel(
    world_xyz: tuple[float, float, float], camera_config: CameraConfig
) -> tuple[int, int]:
    """Forward-project a known 3D world point to its expected pixel location,
    using the same camera geometry as env.get_rgb_image(). For ground-truth
    sanity-checking only.
    """
    view_matrix = np.array(
        p.computeViewMatrix(
            camera_config.eye_position, camera_config.target_position, camera_config.up_vector
        )
    ).reshape(4, 4, order="F")
    aspect = camera_config.width / camera_config.height
    projection_matrix = np.array(
        p.computeProjectionMatrixFOV(
            camera_config.fov_deg, aspect, camera_config.near_val, camera_config.far_val
        )
    ).reshape(4, 4, order="F")

    world_point = np.array([*world_xyz, 1.0])
    clip = projection_matrix @ (view_matrix @ world_point)
    ndc = clip[:3] / clip[3]

    pixel_x = (ndc[0] + 1.0) / 2.0 * camera_config.width
    pixel_y = (1.0 - ndc[1]) / 2.0 * camera_config.height
    return int(round(pixel_x)), int(round(pixel_y))

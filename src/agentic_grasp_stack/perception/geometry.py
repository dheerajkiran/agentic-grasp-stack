"""Camera-geometry math: mapping between 3D world points and 2D pixels, using
the same pinhole camera model as env.get_rgb_image().

view/projection matrix computation needs no connected PyBullet physics
client -- p.computeViewMatrix/p.computeProjectionMatrixFOV are pure math.
"""

import numpy as np
import pybullet as p

from agentic_grasp_stack.config.scene_config import CameraConfig


def _view_projection_matrix(camera_config: CameraConfig) -> np.ndarray:
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
    return projection_matrix @ view_matrix


def world_to_pixel(
    world_xyz: tuple[float, float, float], camera_config: CameraConfig
) -> tuple[int, int]:
    """Forward-project a known 3D world point to its expected pixel location.
    For ground-truth sanity-checking only.
    """
    view_projection = _view_projection_matrix(camera_config)
    world_point = np.array([*world_xyz, 1.0])
    clip = view_projection @ world_point
    ndc = clip[:3] / clip[3]

    pixel_x = (ndc[0] + 1.0) / 2.0 * camera_config.width
    pixel_y = (1.0 - ndc[1]) / 2.0 * camera_config.height
    return int(round(pixel_x)), int(round(pixel_y))


def pixel_to_world(
    pixel_xy: tuple[float, float],
    camera_config: CameraConfig,
    target_height: float,
) -> tuple[float, float, float]:
    """Back-project a 2D pixel to a 3D world point by intersecting the camera
    ray with the known horizontal plane z = target_height.

    Assumes a pinhole camera (matches env.get_rgb_image()'s renderer) and that
    the pixel corresponds to a point lying on that plane -- valid in this phase
    because every object is a cube of known height sitting on the table, so its
    center height (env.surface_z + CUBE_HALF_EXTENT) is a known constant rather
    than something inferred from the image. Not general monocular depth
    estimation; pure geometry, does not depend on GraspEnv.
    """
    pixel_x, pixel_y = pixel_xy
    ndc_x = 2.0 * pixel_x / camera_config.width - 1.0
    ndc_y = 1.0 - 2.0 * pixel_y / camera_config.height

    view_projection = _view_projection_matrix(camera_config)
    inverse_view_projection = np.linalg.inv(view_projection)

    # Every point along the camera ray through this pixel shares the same NDC
    # x/y regardless of depth, so any ndc_z unprojects to a point on the ray.
    ndc_point = np.array([ndc_x, ndc_y, 0.0, 1.0])
    world_point_h = inverse_view_projection @ ndc_point
    world_point = world_point_h[:3] / world_point_h[3]

    origin = np.array(camera_config.eye_position)
    direction = world_point - origin

    if abs(direction[2]) < 1e-9:
        raise ValueError("Camera ray is parallel to the target plane; cannot intersect.")

    t = (target_height - origin[2]) / direction[2]
    x = origin[0] + t * direction[0]
    y = origin[1] + t * direction[1]
    return (float(x), float(y), float(target_height))

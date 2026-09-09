"""Round-trip self-consistency check for perception/geometry.py -- pure math,
no model weights or PyBullet connection required, so unlike the detection
slice this is cheap to run as an automated regression test."""

import pytest

from agentic_grasp_stack.config.scene_config import CAMERA_CONFIG
from agentic_grasp_stack.perception.geometry import pixel_to_world, world_to_pixel

SAMPLE_WORLD_POINTS = [
    (0.45, -0.15, 0.02),
    (0.45, 0.15, 0.02),
    (0.6, -0.15, 0.02),
    (0.6, 0.15, 0.02),
    (0.5, 0.0, 0.02),
    (0.35, -0.25, 0.02),  # workspace corner
    (0.65, 0.25, 0.02),  # opposite workspace corner
]


@pytest.mark.parametrize("world_xyz", SAMPLE_WORLD_POINTS)
def test_pixel_to_world_round_trips_world_to_pixel(world_xyz):
    pixel = world_to_pixel(world_xyz, CAMERA_CONFIG)
    target_height = world_xyz[2]
    recovered = pixel_to_world(pixel, CAMERA_CONFIG, target_height)

    assert recovered[2] == pytest.approx(target_height)
    xy_error = ((recovered[0] - world_xyz[0]) ** 2 + (recovered[1] - world_xyz[1]) ** 2) ** 0.5
    # world_to_pixel rounds to the nearest integer pixel before we invert, so the
    # round trip isn't bit-exact -- bound pixel-rounding error (~mm), not a loose fudge.
    assert xy_error < 0.005

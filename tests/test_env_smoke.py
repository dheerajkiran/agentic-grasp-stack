"""Headless regression guard for the Phase 1 foundation, so later phases can't
silently break the sim/robot/motion core."""

from agentic_grasp_stack.config.scene_config import CAMERA_CONFIG
from agentic_grasp_stack.sim.env import GraspEnv
from agentic_grasp_stack.sim.motion import run_pick_place


def test_camera_image_shape():
    env = GraspEnv(gui=False)
    env.connect()
    try:
        env.reset(randomize=False)
        image = env.get_rgb_image()
        assert image.shape == (CAMERA_CONFIG.height, CAMERA_CONFIG.width, 3)
    finally:
        env.close()


def test_scripted_pick_place_succeeds():
    env = GraspEnv(gui=False)
    env.connect()
    try:
        env.reset(randomize=False)
        place_xy = (0.55, 0.0)
        run_pick_place(env, "red_cube", place_xy)

        final_position, _ = env.cubes.get_pose("red_cube")
        xy_error = (
            (final_position[0] - place_xy[0]) ** 2 + (final_position[1] - place_xy[1]) ** 2
        ) ** 0.5
        assert xy_error <= 0.05
        assert final_position[2] > env.surface_z
    finally:
        env.close()


def test_reset_randomize_is_seeded_and_non_overlapping():
    env = GraspEnv(gui=False)
    env.connect()
    try:
        states_a = env.reset(randomize=True, seed=42)
        states_b = env.reset(randomize=True, seed=42)
        for name in states_a:
            assert states_a[name]["position"] == states_b[name]["position"]
    finally:
        env.close()

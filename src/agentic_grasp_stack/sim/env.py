"""Gym-like PyBullet environment: connect/reset/close plus ground-truth and
camera observation accessors. This is the surface later phases build on top of
(a future ROS2 node wraps this unchanged; a future Execution agent calls
robot.move_to_pose/set_gripper directly)."""

import random

import numpy as np
import pybullet as p
import pybullet_data

from agentic_grasp_stack.config.scene_config import (
    CAMERA_CONFIG,
    CANONICAL_OBJECT_POSITIONS,
    CUBE_HALF_EXTENT,
    CameraConfig,
    OBJECT_COLORS,
)
from agentic_grasp_stack.sim import scene
from agentic_grasp_stack.sim.objects import CubeSet, sample_free_positions
from agentic_grasp_stack.sim.robot import PandaRobot

SETTLE_STEPS = 10
CALIBRATION_SETTLE_STEPS = 120


class GraspEnv:
    def __init__(self, gui: bool = True):
        self.gui = gui
        self.client_id: int | None = None
        self.robot = PandaRobot()
        self.table_id: int | None = None
        self.surface_z: float = 0.0
        self.cubes: CubeSet | None = None

    def connect(self) -> None:
        mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        scene.load_plane()
        self.table_id, self.surface_z = scene.load_table()
        self.robot.load()
        self.cubes = CubeSet(self.surface_z)
        self.cubes.spawn()

        # scene.load_table()'s AABB-based surface_z doesn't always match where
        # PyBullet's physics actually resolves contact (measured ~13.6mm off on
        # this table URDF) -- cubes spawned at the AABB estimate start slightly
        # embedded in the table and get pushed out during settling. Correct
        # surface_z from where a cube actually comes to rest, once, here, then
        # re-place all cubes cleanly at the corrected height.
        for _ in range(CALIBRATION_SETTLE_STEPS):
            p.stepSimulation()
        probe_name = next(iter(self.cubes.body_ids))
        (_, _, settled_z), _ = self.cubes.get_pose(probe_name)
        self.surface_z = settled_z - CUBE_HALF_EXTENT
        self.cubes.surface_z = self.surface_z
        self.cubes.reposition(CANONICAL_OBJECT_POSITIONS)

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0.5, 0, 0],
            )

    def reset(
        self, randomize: bool = True, seed: int | None = None
    ) -> dict[str, dict[str, tuple]]:
        self.robot.home()

        if randomize:
            rng = random.Random(seed)
            names = list(OBJECT_COLORS.keys())
            positions = sample_free_positions(len(names), rng)
            position_map = dict(zip(names, positions))
        else:
            position_map = CANONICAL_OBJECT_POSITIONS

        self.cubes.reposition(position_map)

        for _ in range(SETTLE_STEPS):
            p.stepSimulation()

        return self.get_object_states()

    def get_object_states(self) -> dict[str, dict[str, tuple]]:
        return {
            name: dict(zip(("position", "orientation"), self.cubes.get_pose(name)))
            for name in self.cubes.body_ids
        }

    def get_rgb_image(self, camera_config: CameraConfig | None = None) -> np.ndarray:
        """Render an RGB image from a fixed overhead camera. Unused by the Phase 1
        demo; exists so the Phase 2 Perception agent isn't blocked on env changes."""
        cfg = camera_config or CAMERA_CONFIG
        view_matrix = p.computeViewMatrix(cfg.eye_position, cfg.target_position, cfg.up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(
            cfg.fov_deg, cfg.width / cfg.height, cfg.near_val, cfg.far_val
        )
        _, _, rgba, _, _ = p.getCameraImage(
            cfg.width,
            cfg.height,
            view_matrix,
            projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        return np.reshape(rgba, (cfg.height, cfg.width, 4))[:, :, :3].astype(np.uint8)

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

"""Cube objects: spawning, non-overlapping random placement, and GUI labels."""

import math
import random

import pybullet as p

from agentic_grasp_stack.config.scene_config import (
    CANONICAL_OBJECT_POSITIONS,
    CUBE_HALF_EXTENT,
    CUBE_MASS,
    CUBE_MIN_SEPARATION,
    OBJECT_COLORS,
    WORKSPACE_BOUNDS,
)

LABEL_HEIGHT_OFFSET = 0.06


def sample_free_positions(
    n: int,
    rng: random.Random,
    min_dist: float = CUBE_MIN_SEPARATION,
    max_attempts: int = 200,
) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    for _ in range(n):
        for _ in range(max_attempts):
            x = rng.uniform(*WORKSPACE_BOUNDS.x)
            y = rng.uniform(*WORKSPACE_BOUNDS.y)
            if all(math.hypot(x - px, y - py) >= min_dist for px, py in positions):
                positions.append((x, y))
                break
        else:
            raise RuntimeError(
                f"Could not place object {len(positions) + 1}/{n} without overlap "
                f"after {max_attempts} attempts; workspace too small for min_dist={min_dist}."
            )
    return positions


class CubeSet:
    """Owns the cube bodies and their debug-text labels for one connection."""

    def __init__(self, surface_z: float):
        self.surface_z = surface_z
        self.body_ids: dict[str, int] = {}
        self._label_ids: dict[str, int] = {}

    def spawn(self) -> None:
        """Create each cube body once, at its canonical position, with a label."""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[CUBE_HALF_EXTENT] * 3
        )
        for name, color in OBJECT_COLORS.items():
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[CUBE_HALF_EXTENT] * 3, rgbaColor=color
            )
            x, y = CANONICAL_OBJECT_POSITIONS[name]
            position = (x, y, self.surface_z + CUBE_HALF_EXTENT)
            body_id = p.createMultiBody(
                baseMass=CUBE_MASS,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
            )
            self.body_ids[name] = body_id
            self._label_ids[name] = p.addUserDebugText(
                name, [x, y, self.surface_z + LABEL_HEIGHT_OFFSET]
            )

    def reposition(self, positions: dict[str, tuple[float, float]]) -> None:
        """Move existing cube bodies (and their labels) to new XY positions."""
        for name, (x, y) in positions.items():
            body_id = self.body_ids[name]
            p.resetBasePositionAndOrientation(
                body_id, [x, y, self.surface_z + CUBE_HALF_EXTENT], [0, 0, 0, 1]
            )
            p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
            self._label_ids[name] = p.addUserDebugText(
                name,
                [x, y, self.surface_z + LABEL_HEIGHT_OFFSET],
                replaceItemUniqueId=self._label_ids[name],
            )

    def get_pose(self, name: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return p.getBasePositionAndOrientation(self.body_ids[name])

"""Static scene assets: ground plane and table. Loaded once per connection."""

import pybullet as p

from agentic_grasp_stack.config.scene_config import TABLE_CONFIG


def load_plane() -> int:
    return p.loadURDF("plane.urdf")


def load_table() -> tuple[int, float]:
    """Load the table and measure its top surface height via its AABB.

    Returns (body_id, surface_z).
    """
    table_id = p.loadURDF("table/table.urdf", basePosition=TABLE_CONFIG.base_position)
    _aabb_min, aabb_max = p.getAABB(table_id)
    surface_z = aabb_max[2]
    return table_id, surface_z

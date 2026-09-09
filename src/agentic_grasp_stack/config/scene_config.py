"""Scene constants shared by env.py, scene.py, objects.py, and robot.py.

TABLE_SURFACE_Z_DEFAULT is measured (via p.getAABB() in scene.load_table(), not
guessed) with table/table.urdf loaded at TableConfig.base_position below.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TableConfig:
    urdf_path: str = "table/table.urdf"
    base_position: tuple[float, float, float] = (0.5, 0.0, -0.65)


@dataclass(frozen=True)
class WorkspaceBounds:
    """XY range on the tabletop where objects may be placed, in world coordinates."""

    x: tuple[float, float] = (0.35, 0.65)
    y: tuple[float, float] = (-0.25, 0.25)


@dataclass(frozen=True)
class CameraConfig:
    """Overhead render used by env.get_rgb_image(). width/height/fov_deg are tuned
    so the 0.5x0.3m workspace fills a comfortable fraction of the frame -- at the
    original 320x240/60deg a 4cm cube was only ~10px tall, too small for reliable
    open-vocabulary detection. eye/target/up/near/far are unchanged from Phase 1;
    this is a resolution/zoom change only.
    """

    eye_position: tuple[float, float, float] = (0.5, 0.0, 0.9)
    target_position: tuple[float, float, float] = (0.5, 0.0, 0.0)
    up_vector: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 42.0
    width: int = 640
    height: int = 480
    near_val: float = 0.05
    far_val: float = 3.0


TABLE_SURFACE_Z_DEFAULT = -0.024

ROBOT_BASE_POSITION: tuple[float, float, float] = (0.0, 0.0, 0.0)

HOME_JOINT_POSITIONS: tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0

CUBE_HALF_EXTENT = 0.02  # 4 cm cube
CUBE_MASS = 0.05
CUBE_MIN_SEPARATION = CUBE_HALF_EXTENT * 2 * 1.5  # margin beyond worst-case diagonal

OBJECT_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red_cube": (0.85, 0.1, 0.1, 1.0),
    "green_cube": (0.1, 0.75, 0.15, 1.0),
    "blue_cube": (0.1, 0.2, 0.85, 1.0),
    "yellow_cube": (0.9, 0.85, 0.1, 1.0),
}

# Fixed canonical layout used when GraspEnv.reset(randomize=False) — repeatable demo runs.
CANONICAL_OBJECT_POSITIONS: dict[str, tuple[float, float]] = {
    "red_cube": (0.45, -0.15),
    "green_cube": (0.45, 0.15),
    "blue_cube": (0.6, -0.15),
    "yellow_cube": (0.6, 0.15),
}

DEFAULT_PLACE_TARGET: tuple[float, float] = (0.55, 0.0)

TABLE_CONFIG = TableConfig()
WORKSPACE_BOUNDS = WorkspaceBounds()
CAMERA_CONFIG = CameraConfig()

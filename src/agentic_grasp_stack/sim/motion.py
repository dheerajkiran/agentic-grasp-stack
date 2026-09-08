"""Phase-1-only scripted pick-place state machine.

This module is deliberately isolated from the reusable core (env.py, robot.py):
it's the piece meant to be replaced once an Execution agent generates its own
action sequences from perception + grounding, rather than a hardcoded target.

"Hardcoded poses" here means the *choice* of which object to pick and where to
place it is hardcoded (no perception/agents yet) -- but the actual pick pose is
read from PyBullet ground truth at pick time, not a fixed number, so the same
script works under both fixed and randomized env.reset() layouts. Ground truth
stands in for what a future Perception/Grounding agent will eventually provide.
"""

from agentic_grasp_stack.config.scene_config import CUBE_HALF_EXTENT
from agentic_grasp_stack.sim.env import GraspEnv

APPROACH_HEIGHT = 0.15


def run_pick_place(
    env: GraspEnv, pick_object_name: str, place_xy: tuple[float, float]
) -> None:
    robot = env.robot

    pick_position, _ = env.cubes.get_pose(pick_object_name)
    pick_x, pick_y, pick_z = pick_position
    place_x, place_y = place_xy
    place_z = env.surface_z + CUBE_HALF_EXTENT
    lift_z = max(pick_z, place_z) + APPROACH_HEIGHT

    robot.open_gripper()
    robot.move_to_pose((pick_x, pick_y, lift_z))  # pre_pick
    robot.move_to_pose((pick_x, pick_y, pick_z))  # descend
    robot.close_gripper()
    robot.move_to_pose((pick_x, pick_y, lift_z))  # lift
    robot.move_to_pose((place_x, place_y, lift_z))  # pre_place
    robot.move_to_pose((place_x, place_y, place_z))  # place_descend
    robot.open_gripper()
    robot.move_to_pose((place_x, place_y, lift_z))  # retreat

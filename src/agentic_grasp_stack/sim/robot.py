"""Franka Panda wrapper: loading, IK-driven Cartesian motion, and gripper control.

Joint/link *indices* are discovered at load time by matching URDF joint/link
*names*, since index layout can shift across pybullet_data revisions. Only the
names (panda_joint1..7, panda_finger_joint1/2, panda_grasptarget) are hardcoded.
"""

from dataclasses import dataclass

import pybullet as p

from agentic_grasp_stack.config.scene_config import (
    GRIPPER_CLOSED_WIDTH,
    GRIPPER_OPEN_WIDTH,
    HOME_JOINT_POSITIONS,
    ROBOT_BASE_POSITION,
)

ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
EE_LINK_NAME = "panda_grasptarget"

DOWN_ORIENTATION = p.getQuaternionFromEuler([3.14159265, 0.0, 0.0])


@dataclass
class Pose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] = DOWN_ORIENTATION


class PandaRobot:
    def __init__(self, base_position: tuple[float, float, float] = ROBOT_BASE_POSITION):
        self.base_position = base_position
        self.body_id: int | None = None
        self.arm_joint_indices: list[int] = []
        self.finger_joint_indices: list[int] = []
        self.movable_joint_indices: list[int] = []
        self.ee_link_index: int | None = None

    def load(self) -> None:
        self.body_id = p.loadURDF(
            "franka_panda/panda.urdf", basePosition=self.base_position, useFixedBase=True
        )
        num_joints = p.getNumJoints(self.body_id)

        name_to_index: dict[str, int] = {}
        link_name_to_index: dict[str, int] = {}
        movable: list[int] = []
        for i in range(num_joints):
            info = p.getJointInfo(self.body_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            name_to_index[joint_name] = i
            link_name_to_index[link_name] = i
            if info[2] != p.JOINT_FIXED:
                movable.append(i)

        self.movable_joint_indices = movable
        self.arm_joint_indices = [name_to_index[n] for n in ARM_JOINT_NAMES]
        self.finger_joint_indices = [name_to_index[n] for n in FINGER_JOINT_NAMES]
        self.ee_link_index = link_name_to_index[EE_LINK_NAME]

        assert self.movable_joint_indices == self.arm_joint_indices + self.finger_joint_indices, (
            "Unexpected joint ordering in panda.urdf; IK rest-pose mapping assumes "
            "movable joints are [arm x7, fingers x2] in ascending index order."
        )

        self.home()

    def home(self) -> None:
        for joint_index, angle in zip(self.arm_joint_indices, HOME_JOINT_POSITIONS):
            p.resetJointState(self.body_id, joint_index, angle)
        for joint_index in self.finger_joint_indices:
            p.resetJointState(self.body_id, joint_index, GRIPPER_OPEN_WIDTH)

    def get_ee_pose(self) -> Pose:
        link_state = p.getLinkState(self.body_id, self.ee_link_index)
        return Pose(position=link_state[4], orientation=link_state[5])

    def move_to_pose(
        self,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float] = DOWN_ORIENTATION,
        steps: int = 120,
    ) -> None:
        start_position = self.get_ee_pose().position
        rest_poses = list(HOME_JOINT_POSITIONS) + [GRIPPER_OPEN_WIDTH, GRIPPER_OPEN_WIDTH]

        for step in range(1, steps + 1):
            t = step / steps
            interp_position = [
                start_position[i] + t * (position[i] - start_position[i]) for i in range(3)
            ]
            ik_solution = p.calculateInverseKinematics(
                self.body_id,
                self.ee_link_index,
                interp_position,
                orientation,
                restPoses=rest_poses,
                maxNumIterations=100,
                residualThreshold=1e-4,
            )
            arm_targets = ik_solution[: len(self.arm_joint_indices)]
            p.setJointMotorControlArray(
                self.body_id,
                self.arm_joint_indices,
                p.POSITION_CONTROL,
                targetPositions=arm_targets,
            )
            p.stepSimulation()

    def set_gripper(self, width: float, force: float = 20.0, steps: int = 60) -> None:
        p.setJointMotorControlArray(
            self.body_id,
            self.finger_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[width, width],
            forces=[force, force],
        )
        for _ in range(steps):
            p.stepSimulation()

    def open_gripper(self) -> None:
        self.set_gripper(GRIPPER_OPEN_WIDTH)

    def close_gripper(self) -> None:
        self.set_gripper(GRIPPER_CLOSED_WIDTH)

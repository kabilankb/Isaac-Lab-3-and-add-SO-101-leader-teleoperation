"""Layout of the YAM bimanual action vector.

The 14-D action is ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`` —
absolute joint-position targets. :class:`YamActionLayout` is the one place that maps this
layout to per-arm slices and gripper indices, so the environment, teleoperation server, and
MimicGen code never hard-code an offset.
"""


class YamActionLayout:
    """Index layout of the 14-D YAM bimanual action vector.

    ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.
    """

    DIM = 14
    ARM_DOF = 6

    LEFT_ARM = slice(0, 6)
    LEFT_GRIPPER = 6
    RIGHT_ARM = slice(7, 13)
    RIGHT_GRIPPER = 13

    # Within a single arm's joint vector ``[joint1..6, finger]``, the gripper joint index.
    ARM_GRIPPER_JOINT_INDEX = 6
    # The gripper command drives this finger joint; the opposing finger mirrors it.
    GRIPPER_DRIVE_JOINT = "left_finger"
    MIRROR_JOINT = "right_finger"

    _PER_ARM = {
        "left_arm": (LEFT_ARM, LEFT_GRIPPER),
        "right_arm": (RIGHT_ARM, RIGHT_GRIPPER),
    }

    @classmethod
    def arm_slice(cls, arm_name: str) -> slice:
        """Slice of an arm's six joint targets in the action vector.

        Args:
            arm_name (str): ``"left_arm"`` or ``"right_arm"``.

        Returns:
            slice: The arm's joint-target columns.
        """
        return cls._PER_ARM[arm_name][0]

    @classmethod
    def gripper_index(cls, arm_name: str) -> int:
        """Column index of an arm's gripper command in the action vector.

        Args:
            arm_name (str): ``"left_arm"`` or ``"right_arm"``.

        Returns:
            int: The gripper command column.
        """
        return cls._PER_ARM[arm_name][1]

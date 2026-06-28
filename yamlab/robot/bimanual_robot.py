"""Bimanual specialization of the runtime robot model.

:class:`BimanualRobot` is a two-arm :class:`~yamlab.robot.robot.Robot` that exposes
``left_arm``/``right_arm`` and returns grasp results as a ``(left, right)`` pair, the form
the bimanual tasks expect. The generic parts (``Finger``/``Gripper``/``Arm``/``Robot``) live
in :mod:`yamlab.robot.robot`.
"""

import torch

from .robot import Robot


class BimanualRobot(Robot):
    """Two-arm robot: exposes ``left_arm``/``right_arm`` and returns ``(left, right)`` results."""

    def __init__(self, scene, spec, grasp_target: str):
        """Build the two-arm robot.

        Args:
            scene (InteractiveScene): The live scene.
            spec (RobotSpec): The robot's hardware spec (must expose ``left_arm``/``right_arm``).
            grasp_target (str): Default grasp target object.
        """
        super().__init__(scene, spec, grasp_target)
        self.left_arm = self.arms["left_arm"]
        self.right_arm = self.arms["right_arm"]

    def is_grasping(self, target_object: str = None, env_ids: torch.Tensor = None,
                    normal_force_thresh: float = None):
        """Whether each arm is grasping a target object, per environment.

        Args:
            target_object (str | None): Scene-object name; ``None`` uses ``grasp_target``.
            env_ids (None | torch.Tensor): Environment indices, shape ``(n,)``; ``None`` = all.
            normal_force_thresh (None | float): Override the contact-force threshold (N).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(left_grasping, right_grasping)``, each a bool
                tensor of shape ``(num_envs,)`` or ``(n,)``.
        """
        return (self.left_arm.is_grasping(target_object, env_ids, normal_force_thresh),
                self.right_arm.is_grasping(target_object, env_ids, normal_force_thresh))

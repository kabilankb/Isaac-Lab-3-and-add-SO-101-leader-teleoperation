"""Shared MimicGen embodiment base for the YAM bimanual robot.

The IsaacLab Mimic interface mixes two concerns: embodiment methods (eef pose <->
joint action, via FK/IK) and task methods (subtask signals, object poses). The
embodiment methods are identical for every YAM task, so they live here once;
task envs subclass this and implement only the task methods
(``get_subtask_term_signals`` and ``get_object_poses``).
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.math import subtract_frame_transforms

from yamlab.utils.jparse_ik import compute_ik_jparse
from yamlab.robot.yam import YamActionLayout


class YamMimicEnv(ManagerBasedRLMimicEnv):
    """YAM bimanual embodiment implementation of the IsaacLab Mimic API.

    Provides the robot-kinematics half of the Mimic interface (eef pose, target-pose
    -> joint action via J-PARSE IK, and the inverse), shared by every YAM MimicGen
    task. Assumes the bimanual 6-DOF + 1-gripper layout (action format
    ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``) and the two
    scene arms ``"left_arm"`` / ``"right_arm"``.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return the current end-effector pose as a homogeneous transform.

        Args:
            eef_name (str): ``"left_arm"`` or ``"right_arm"``.
            env_ids (Sequence[int] | None): Environment indices. If None, returns all envs.

        Returns:
            torch.Tensor: float, shape (N, 4, 4) end-effector pose matrices.
        """
        if env_ids is None:
            env_ids = slice(None)

        arm = self.scene["left_arm"] if eef_name == "left_arm" else self.scene["right_arm"]

        eef_body_idx = arm.num_bodies - 1
        eef_pose_w = arm.data.body_link_pose_w.torch[:, eef_body_idx, :]  # (num_envs, 7) [pos, quat xyzw]

        if isinstance(env_ids, slice):
            eef_pos = eef_pose_w[:, :3]
            eef_quat = eef_pose_w[:, 3:]
        else:
            eef_pos = eef_pose_w[env_ids, :3]
            eef_quat = eef_pose_w[env_ids, 3:]

        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target end-effector poses and gripper commands to a 14-dim joint action.

        Uses J-PARSE differential IK to solve for the arm joint positions.

        Args:
            target_eef_pose_dict (dict): Maps arm name -> target pose, float tensor of shape (4, 4).
            gripper_action_dict (dict): Maps arm name -> scalar gripper action.
            action_noise_dict (dict | None): Optional per-arm joint-position noise magnitude.
            env_id (int): Environment index (single-env generation assumed).

        Returns:
            torch.Tensor: float, shape (14,), layout
                ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.
        """
        device = self.device

        left_arm = self.scene["left_arm"]
        right_arm = self.scene["right_arm"]

        left_joint_pos = left_arm.data.joint_pos.torch[env_id, :6].clone()
        right_joint_pos = right_arm.data.joint_pos.torch[env_id, :6].clone()

        if "left_arm" in target_eef_pose_dict:
            target_pose = target_eef_pose_dict["left_arm"]
            if target_pose.dim() == 2:
                target_pose = target_pose.unsqueeze(0)
            left_joint_pos = self._compute_ik(
                arm=left_arm,
                target_pose=target_pose[0],
                initial_joint_pos=left_joint_pos,
                env_id=env_id,
            )

        if "right_arm" in target_eef_pose_dict:
            target_pose = target_eef_pose_dict["right_arm"]
            if target_pose.dim() == 2:
                target_pose = target_pose.unsqueeze(0)
            right_joint_pos = self._compute_ik(
                arm=right_arm,
                target_pose=target_pose[0],
                initial_joint_pos=right_joint_pos,
                env_id=env_id,
            )

        left_gripper = gripper_action_dict.get("left_arm", torch.zeros(1, device=device))
        right_gripper = gripper_action_dict.get("right_arm", torch.zeros(1, device=device))

        if not isinstance(left_gripper, torch.Tensor):
            left_gripper = torch.tensor([left_gripper], dtype=torch.float32, device=device)
        if not isinstance(right_gripper, torch.Tensor):
            right_gripper = torch.tensor([right_gripper], dtype=torch.float32, device=device)

        if left_gripper.dim() > 0:
            left_gripper = left_gripper.flatten()[:1]
        if right_gripper.dim() > 0:
            right_gripper = right_gripper.flatten()[:1]

        action = torch.cat([left_joint_pos, left_gripper, right_joint_pos, right_gripper], dim=0)

        if action_noise_dict is not None:
            if "left_arm" in action_noise_dict and action_noise_dict["left_arm"] > 0:
                action[YamActionLayout.LEFT_ARM] += action_noise_dict["left_arm"] * torch.randn(6, device=device)
            if "right_arm" in action_noise_dict and action_noise_dict["right_arm"] > 0:
                action[YamActionLayout.RIGHT_ARM] += action_noise_dict["right_arm"] * torch.randn(6, device=device)

        return action

    def _compute_ik(
        self,
        arm,
        target_pose: torch.Tensor,
        initial_joint_pos: torch.Tensor,
        env_id: int,
    ) -> torch.Tensor:
        """Compute one-step J-PARSE IK toward ``target_pose``.

        J-PARSE selectively dampens only near-singular Jacobian directions
        (identified via SVD), maintaining full-speed motion elsewhere.
        Assumes ``num_envs=1`` for MimicGen data generation.

        Args:
            arm (Articulation): Articulation object for the robot arm.
            target_pose (torch.Tensor): float, shape (4, 4), target EEF pose in the world frame.
            initial_joint_pos (torch.Tensor): float, shape (6,), current joint positions
                (returned unchanged if already at target).
            env_id (int): Environment index.

        Returns:
            torch.Tensor: float, shape (6,), desired joint positions clamped to USD joint limits.
        """
        base_pose_w = arm.data.root_pose_w.torch[env_id]
        base_pos_w = base_pose_w[:3].unsqueeze(0)
        base_quat_w = base_pose_w[3:7].unsqueeze(0)

        eef_body_idx = arm.num_bodies - 1
        eef_jacobi_idx = eef_body_idx - 1 if arm.is_fixed_base else eef_body_idx
        joint_ids = list(range(6))

        ee_pose_w = arm.data.body_pose_w.torch[env_id, eef_body_idx]
        ee_pos_w = ee_pose_w[:3].unsqueeze(0)
        ee_quat_w = ee_pose_w[3:7].unsqueeze(0)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w, ee_pos_w, ee_quat_w
        )

        target_pos_w = target_pose[:3, 3].unsqueeze(0)
        target_rot_w = target_pose[:3, :3].unsqueeze(0)
        target_quat_w = PoseUtils.quat_from_matrix(target_rot_w)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w, target_pos_w, target_quat_w
        )

        if torch.norm(target_pos_b - ee_pos_b) < 1e-3:
            return initial_joint_pos.clone()

        # Raw PhysX (COM-referenced) Jacobian, as root_physx_view.get_jacobians() returned in 2.x.
        jacobians_full = arm.data.body_com_jacobian_w.torch
        jacobian = jacobians_full[env_id : env_id + 1, eef_jacobi_idx, :, joint_ids]  # (1, 6, 6)
        joint_pos = arm.data.joint_pos.torch[env_id : env_id + 1, joint_ids]                # (1, 6)

        joint_pos_des = compute_ik_jparse(
            ee_pos_b, ee_quat_b,
            target_pos_b, target_quat_b,
            jacobian, joint_pos,
        )

        joint_limits = arm.data.joint_pos_limits.torch[env_id, :6, :]  # (6, 2)
        return torch.clamp(joint_pos_des[0], min=joint_limits[:, 0], max=joint_limits[:, 1])

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return current end-effector poses as the effective targets for an applied action.

        Args:
            action (torch.Tensor): The action tensor already applied to the sim.

        Returns:
            dict[str, torch.Tensor]: Maps arm name -> pose, float tensor of shape (N, 4, 4).
        """
        return {
            "left_arm": self.get_robot_eef_pose("left_arm", env_ids=None),
            "right_arm": self.get_robot_eef_pose("right_arm", env_ids=None),
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract per-arm gripper action columns from an action tensor.

        Args:
            actions (torch.Tensor): float, shape (T, 14) (single env) or (B, T, 14) (batched).

        Returns:
            dict[str, torch.Tensor]: Maps arm name -> gripper action column,
                shape (T, 1) or (B, T, 1).
        """
        lg, rg = YamActionLayout.LEFT_GRIPPER, YamActionLayout.RIGHT_GRIPPER
        if actions.dim() == 2:
            return {"left_arm": actions[:, lg:lg + 1], "right_arm": actions[:, rg:rg + 1]}
        return {"left_arm": actions[:, :, lg:lg + 1], "right_arm": actions[:, :, rg:rg + 1]}

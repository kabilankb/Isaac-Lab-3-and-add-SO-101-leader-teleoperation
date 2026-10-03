"""MimicGen-compatible environment for the PutPotOnCooktop bimanual task.

Both arms cooperatively grasp the pot and place it on top of the cooktop.
Implements the YAM MimicGen API (via ``YamMimicEnv``) with latched,
sequentially-gated subtask termination signals.
"""

import gymnasium as gym
import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils

import yamlab.utils.mimic_patches as mimic_utils
from yamlab.envs.mimic.yam_mimic_env import YamMimicEnv

from yamlab.envs.tasks.put_pot_on_cooktop_manager import PutPotOnCooktopManager
from yamlab.envs.tasks.yam_bimanual_env import make_task_env
from .put_pot_on_cooktop_mimic_env_cfg import PutPotOnCooktopMimicEnvCfg


def make_put_pot_on_cooktop_mimic_env(**kwargs):
    """Factory function for the PutPotOnCooktop-Mimic gymnasium environment.

    Args:
        **kwargs: Configuration parameters forwarded to the env constructor
            or applied to the cfg object when one is provided directly.

    Returns:
        A ``PutPotOnCooktopMimicEnv`` instance.
    """
    kwargs['pick_by_two_hands'] = True

    if 'check_gripper_release_for_ontop' not in kwargs:
        kwargs['check_gripper_release_for_ontop'] = True

    cfg = kwargs.pop('cfg', None)

    if cfg is not None:
        physics_kwargs = {}
        if 'enable_self_collisions' in kwargs:
            physics_kwargs['enable_self_collisions'] = kwargs.pop('enable_self_collisions')

        for key, value in physics_kwargs.items():
            if hasattr(cfg.sim.physics, key):
                setattr(cfg.sim.physics, key, value)

        config_only_keys = ['assets_instance_paths', 'objects_randomization',
                           'init_joint_pos_randomization', 'teleoperation',
                           'observation_modalities', 'num_envs', 'data_generation',
                           'pick_by_two_hands', 'check_gripper_release_for_ontop']
        constructor_kwargs = {k: v for k, v in kwargs.items() if k not in config_only_keys}

        return PutPotOnCooktopMimicEnv(cfg, **constructor_kwargs)
    else:
        return make_task_env(PutPotOnCooktopMimicEnvCfg, PutPotOnCooktopMimicEnv, **kwargs)


class PutPotOnCooktopMimicEnv(PutPotOnCooktopManager, YamMimicEnv):
    """MimicGen-compatible bimanual environment for the PutPotOnCooktop task.

    Combines task logic from ``PutPotOnCooktopManager`` with the MimicGen trajectory
    generation interface from ``ManagerBasedRLMimicEnv``. Subtask signals are
    latched and sequentially gated: left_grasping -> both_grasping ->
    pot_ready -> above_obj1.
    """

    def __init__(self, cfg, **kwargs):
        """Initialize the bimanual mimic environment and its subtask latch tensors."""
        mimic_utils.hold_sequential_constraint_at_zero()
        mimic_utils.drop_unused_trajectory_buffers()

        # Declared None before super().__init__ so attribute access is safe
        # in any pre-init hook; populated to zero tensors after init.
        self._subtask_latch_left_grasping = None
        self._subtask_latch_both_grasping = None
        self._subtask_latch_pot_ready = None
        self._subtask_latch_above_obj1 = None

        super().__init__(cfg, **kwargs)

        self._subtask_latch_left_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_both_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_pot_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_above_obj1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        print("[INFO] PutPotOnCooktopBimanualMimicEnv initialized (using J-PARSE IK)")
        print("[INFO] - Subtask signals: left_grasping -> both_grasping -> pot_ready -> above_obj1")
        print("[INFO] - Task decomposition: approach+grasp -> pick+rotate -> transport+place")

    def reset_subtask_latch_states(self, env_ids: Sequence[int] | None = None):
        """Reset subtask latch tensors for the specified environments.

        Args:
            env_ids (Sequence[int] | None): Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self._subtask_latch_left_grasping.fill_(False)
            self._subtask_latch_both_grasping.fill_(False)
            self._subtask_latch_pot_ready.fill_(False)
            self._subtask_latch_above_obj1.fill_(False)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            self._subtask_latch_left_grasping[env_ids] = False
            self._subtask_latch_both_grasping[env_ids] = False
            self._subtask_latch_pot_ready[env_ids] = False
            self._subtask_latch_above_obj1[env_ids] = False

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments and clear latch states."""
        super()._reset_idx(env_ids)
        self.reset_subtask_latch_states(env_ids)

    def get_subtask_term_signals(self) -> dict[str, torch.Tensor]:
        """Return latched, sequentially-gated subtask termination signals.

        Signals latch on their first True step and remain True until reset.
        Each signal requires the previous latch to have been True in a prior
        step (MimicGen's sequential-boundary requirement).

        Signal sequence: left_grasping -> both_grasping -> pot_ready -> above_obj1

        Signals:
            left_grasping: left arm contacts pot.
            both_grasping: both arms contact pot (requires prev left_grasping).
            pot_ready: pot lifted >5 cm AND both arms grasping (requires prev both_grasping).
            above_obj1: pot is above cooktop in XY+Z (requires prev pot_ready).

        Returns:
            dict[str, torch.Tensor]: Maps signal name -> latched signal,
                bool tensor of shape (num_envs,).
        """
        prev_left_grasping = self._subtask_latch_left_grasping.clone()
        prev_both_grasping = self._subtask_latch_both_grasping.clone()
        prev_pot_ready = self._subtask_latch_pot_ready.clone()

        left_grasping_raw, right_grasping_raw = self.robot.is_grasping(
            normal_force_thresh=0.1, env_ids=None,
        )
        both_grasping_raw = left_grasping_raw & right_grasping_raw
        pot_ready_raw = self._check_pot_ready_raw()
        above_obj1_raw = self._check_obj0_above_obj1_raw()

        left_grasping_current = left_grasping_raw
        both_grasping_current = both_grasping_raw & prev_left_grasping
        pot_ready_current = pot_ready_raw & prev_both_grasping
        above_obj1_current = above_obj1_raw & prev_pot_ready

        self._subtask_latch_left_grasping |= left_grasping_current
        self._subtask_latch_both_grasping |= both_grasping_current
        self._subtask_latch_pot_ready |= pot_ready_current
        self._subtask_latch_above_obj1 |= above_obj1_current

        return {
            "left_grasping": self._subtask_latch_left_grasping.clone(),
            "both_grasping": self._subtask_latch_both_grasping.clone(),
            "pot_ready": self._subtask_latch_pot_ready.clone(),
            "above_obj1": self._subtask_latch_above_obj1.clone(),
        }

    def _check_pot_ready_raw(self) -> torch.Tensor:
        """Return True where the pot is lifted >=5 cm above its initial height with both arms grasping it.

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        pot = self.scene["pot"]
        lifted = (pot.data.root_pos_w.torch[:, 2] - self.pot_init_z) > 0.05

        left_grasping, right_grasping = self.robot.is_grasping(
            normal_force_thresh=0.1, env_ids=None,
        )
        return lifted & (left_grasping & right_grasping)
    def _check_obj0_above_obj1_raw(self) -> torch.Tensor:
        """Return True where the pot is >=5 cm above the cooktop in Z and within 10 cm in XY.

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        pot = self.scene["pot"]
        cooktop = self.scene["cooktop"]

        pot_pos = pot.data.root_pos_w.torch
        cooktop_pos = cooktop.data.root_pos_w.torch

        above = (pot_pos[:, 2] - cooktop_pos[:, 2]) > 0.05
        aligned = torch.norm(pot_pos[:, :2] - cooktop_pos[:, :2], dim=-1) < 0.10

        return above & aligned

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return 4x4 pose matrices for the pot and cooktop.

        Args:
            env_ids (Sequence[int] | None): Environment indices. If None, returns all envs.

        Returns:
            dict[str, torch.Tensor]: Maps object name -> pose, float tensor of shape (N, 4, 4).
        """
        if env_ids is None:
            env_ids = slice(None)

        object_poses = {}

        for obj_name in ["pot", "cooktop"]:
            obj = self.scene[obj_name]

            if isinstance(env_ids, slice):
                obj_pos = obj.data.root_pos_w.torch
                obj_quat = obj.data.root_quat_w.torch
            else:
                obj_pos = obj.data.root_pos_w.torch[env_ids]
                obj_quat = obj.data.root_quat_w.torch[env_ids]

            object_poses[obj_name] = PoseUtils.make_pose(
                obj_pos,
                PoseUtils.matrix_from_quat(obj_quat),
            )

        return object_poses


gym.register(
    id="PutPotOnCooktop-Mimic-v0",
    entry_point=make_put_pot_on_cooktop_mimic_env,
    disable_env_checker=True,
)

"""PutPotOnCooktop task environment: pick pot and place it on top of cooktop.

Two-stage bimanual task implemented with IsaacLab's ManagerBasedRLEnv framework.
Stage flags latch True once earned and are cleared on episode reset.
"""

import os
import torch
import gymnasium as gym
import math
from typing import Dict, Any

from isaaclab.managers import SceneEntityCfg

from .yam_bimanual_env import YamBimanualEnv, make_task_env
from .put_pot_on_cooktop_manager_cfg import PutPotOnCooktopManagerEnvCfg
from yamlab.utils.task_logic import check_pick_success, check_ontop_success
from yamlab.utils.assets import load_asset_size, OBJ_Z_OFFSET
from yamlab.robot.yam import ROBOT
# Stage-success thresholds are read from configs/tasks/put_pot_on_cooktop.yaml (success:).
from yamlab.configs import get_task_config

_SUCCESS = get_task_config("PutPotOnCooktop-v0")["success"]
PICK_LIFT_THRESHOLD_M = _SUCCESS["pick_lift_threshold_m"]
PICK_REQUIRED_CONSECUTIVE_STEPS = _SUCCESS["pick_required_consecutive_steps"]
ONTOP_HEIGHT_TOLERANCE_M = _SUCCESS["ontop_height_tolerance_m"]
ONTOP_ORIENTATION_TOLERANCE_RAD = _SUCCESS["ontop_orientation_tolerance_rad"]
ONTOP_REQUIRED_CONSECUTIVE_STEPS_EVAL = _SUCCESS["ontop_required_consecutive_steps_eval"]
ONTOP_REQUIRED_CONSECUTIVE_STEPS_TELEOP = _SUCCESS["ontop_required_consecutive_steps_teleop"]


def make_put_pot_on_cooktop_env(**kwargs):
    """Factory that creates a PutPotOnCooktop environment via ``make_task_env``.

    Args:
        **kwargs: Forwarded to ``PutPotOnCooktopManager.__init__`` and ``make_task_env``.
            pick_by_two_hands (bool): Stage 1 requires both grippers in contact.
                Defaults to True.
            check_gripper_release_for_ontop (bool): Stage 2 requires both grippers
                off pot. Defaults to True.

    Returns:
        PutPotOnCooktopManager instance.
    """
    if 'pick_by_two_hands' not in kwargs:
        kwargs['pick_by_two_hands'] = True
    if 'check_gripper_release_for_ontop' not in kwargs:
        kwargs['check_gripper_release_for_ontop'] = True
    return make_task_env(PutPotOnCooktopManagerEnvCfg, PutPotOnCooktopManager, **kwargs)


class PutPotOnCooktopManager(YamBimanualEnv):
    """PutPotOnCooktop task: two-stage pick-and-place with two rigid objects.

    Stage 1 (pick): lift pot above its initial height with both arms grasping.
    Stage 2 (place): rest pot on top of cooktop with grippers released.
    Both stages require a configurable number of consecutive successful steps before latching.
    """

    def __init__(self, cfg: PutPotOnCooktopManagerEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the PutPotOnCooktop environment.

        Args:
            cfg: Environment configuration.
            render_mode: Rendering mode passed to ManagerBasedRLEnv.
            **kwargs: Task-specific keyword arguments:
                pick_by_two_hands (bool): Require both grippers for stage 1. Default True.
                check_gripper_release_for_ontop (bool): Require gripper release for stage 2.
                    Default True.
                is_evaluation_mode (bool): Use stricter 30-step threshold instead of 8.
                    Default False.
        """
        pick_by_two_hands = kwargs.pop('pick_by_two_hands', True)
        check_gripper_release_for_ontop = kwargs.pop('check_gripper_release_for_ontop', True)
        is_evaluation_mode = kwargs.pop('is_evaluation_mode', False)

        # Compute stable resting heights from asset_size.json before env creation
        # so reward config can be patched before IsaacLab reads it.
        usd_path_0 = cfg.scene.pot.spawn.usd_path
        size_info_0 = load_asset_size(os.path.dirname(usd_path_0))
        pot_height = size_info_0['size']['z']
        self.pot_init_z = ROBOT.table_position[2] + pot_height / 2.0 + OBJ_Z_OFFSET

        usd_path_1 = cfg.scene.cooktop.spawn.usd_path
        size_info_1 = load_asset_size(os.path.dirname(usd_path_1))
        cooktop_height = size_info_1['size']['z']
        self.cooktop_init_z = ROBOT.table_position[2] + cooktop_height / 2.0 + OBJ_Z_OFFSET

        self.pot_height = pot_height
        self.cooktop_height = cooktop_height

        pot_xy_radius = math.sqrt(size_info_0['size']['x'] ** 2 + size_info_0['size']['y'] ** 2) / 2.0
        cooktop_xy_radius = math.sqrt(size_info_1['size']['x'] ** 2 + size_info_1['size']['y'] ** 2) / 2.0
        self.ontop_xy_threshold = max(pot_xy_radius, cooktop_xy_radius)

        self._update_reward_parameters_in_config(cfg)

        super().__init__(cfg, render_mode, **kwargs)

        self.pick_by_two_hands = pick_by_two_hands
        self.check_gripper_release_for_ontop = check_gripper_release_for_ontop
        self.is_evaluation_mode = is_evaluation_mode
        # Both hands must release pot for the on-top stage to latch.
        self.gripper_release_mode = "both"

        self._apply_physics_materials_to_objects()

        self.task_stage = 0

        num_envs = self.num_envs
        self.stage1_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.stage2_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._prev_stage1_success = self.stage1_success.clone()
        self._prev_stage2_success = self.stage2_success.clone()
        self._consecutive_pick_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._consecutive_ontop_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

        self._pick_required_consecutive_steps = PICK_REQUIRED_CONSECUTIVE_STEPS
        self._ontop_required_consecutive_steps = (
            ONTOP_REQUIRED_CONSECUTIVE_STEPS_EVAL if is_evaluation_mode
            else ONTOP_REQUIRED_CONSECUTIVE_STEPS_TELEOP
        )

        print(f"[INFO] PutPotOnCooktop Manager Environment initialized")
        print(f"[INFO] - Two-stage task: pick pot + place pot on top of cooktop")
        print(f"[INFO] - Pick by two hands: {self.pick_by_two_hands}")
        print(f"[INFO] - Check gripper release for on-top: {self.check_gripper_release_for_ontop}")
        print(f"[INFO] - Evaluation mode: {self.is_evaluation_mode} (gripper release: {self.gripper_release_mode})")
        print("\033[93m[TASK PROGRESS] pick: False; place: False\033[0m")

    def get_object_names_for_physics_material(self) -> list[str]:
        """Return object names that should receive physics material overrides.

        Returns:
            List containing "pot" and "cooktop".
        """
        return ["pot", "cooktop"]

    def reset_success_check(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset stage success flags and consecutive-step counters for the given environments.

        Args:
            env_ids: Indices of environments to reset. If None, all environments are reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.stage1_success[env_ids] = False
        self.stage2_success[env_ids] = False
        self._prev_stage1_success[env_ids] = False
        self._prev_stage2_success[env_ids] = False
        self._consecutive_pick_steps[env_ids] = 0
        self._consecutive_ontop_steps[env_ids] = 0
        if hasattr(self, "_success_reward_given"):
            self._success_reward_given[env_ids] = False

    def _update_reward_parameters_in_config(self, cfg: PutPotOnCooktopManagerEnvCfg):
        """Patch reward config with dynamically computed object heights before env creation."""
        if hasattr(cfg, 'rewards') and hasattr(cfg.rewards, 'pick_reward'):
            pick_reward = cfg.rewards.pick_reward
            if hasattr(pick_reward, 'params') and 'init_obj_height' in pick_reward.params:
                pick_reward.params['init_obj_height'] = self.pot_init_z

        if hasattr(cfg, 'rewards') and hasattr(cfg.rewards, 'ontop_reward'):
            ontop_reward = cfg.rewards.ontop_reward
            if hasattr(ontop_reward, 'params'):
                if 'init_pot_height' in ontop_reward.params:
                    ontop_reward.params['init_pot_height'] = self.pot_init_z
                if 'init_cooktop_height' in ontop_reward.params:
                    ontop_reward.params['init_cooktop_height'] = self.cooktop_init_z

    def get_current_task_info(self) -> Dict[str, Any]:
        """Return task state for env 0, used for episode metadata and logging.

        Returns:
            Dictionary with object poses, episode count, and per-stage success flags.
        """
        pot = self.scene["pot"]
        pot_pos = pot.data.root_pos_w[0] if hasattr(pot.data, 'root_pos_w') else torch.zeros(3)
        pot_quat = pot.data.root_quat_w[0] if hasattr(pot.data, 'root_quat_w') else torch.tensor([1, 0, 0, 0])

        cooktop = self.scene["cooktop"]
        cooktop_pos = cooktop.data.root_pos_w[0] if hasattr(cooktop.data, 'root_pos_w') else torch.zeros(3)
        cooktop_quat = cooktop.data.root_quat_w[0] if hasattr(cooktop.data, 'root_quat_w') else torch.tensor([1, 0, 0, 0])

        stage1_success = self.stage1_success[0].item()
        stage2_success = self.stage2_success[0].item()
        task_success = stage1_success and stage2_success

        return {
            "pot_position": pot_pos.cpu().numpy().tolist(),
            "pot_orientation": pot_quat.cpu().numpy().tolist(),
            "cooktop_position": cooktop_pos.cpu().numpy().tolist(),
            "cooktop_orientation": cooktop_quat.cpu().numpy().tolist(),
            "total_objs": 2,
            "episode_count": self._episode_count,
            "intermediate_success": {
                "pick": bool(stage1_success),
                "place": bool(stage2_success),
                "task_success": bool(task_success),
            },
        }

    def update_stage_success(self):
        """Evaluate and latch per-stage success after each step.

        Stage 1 (pick): requires ``_pick_required_consecutive_steps`` of continuous pick success.
        Stage 2 (on-top): only checked once stage 1 has latched; requires consecutive on-top success.
        Both stages latch permanently until the next episode reset.
        """
        num_envs = self.num_envs

        pick_success = check_pick_success(
            self,
            SceneEntityCfg("pot"),
            init_obj_height=self.pot_init_z,
            pick_threshold=PICK_LIFT_THRESHOLD_M,
            check_contact=True,
            pick_by_two_hands=self.pick_by_two_hands,
            grasp=self.robot
        )

        mask_increment = pick_success & ~self.stage1_success
        mask_reset = ~pick_success & ~self.stage1_success
        self._consecutive_pick_steps[mask_increment] += 1
        self._consecutive_pick_steps[mask_reset] = 0
        self.stage1_success |= (self._consecutive_pick_steps >= self._pick_required_consecutive_steps)

        ontop_success = check_ontop_success(
            self,
            SceneEntityCfg("pot"),
            SceneEntityCfg("cooktop"),
            pot_height=self.pot_height,
            cooktop_height=self.cooktop_height,
            height_tolerance=ONTOP_HEIGHT_TOLERANCE_M,
            xy_tolerance=self.ontop_xy_threshold,
            orientation_tolerance=ONTOP_ORIENTATION_TOLERANCE_RAD,
            check_gripper_release=self.check_gripper_release_for_ontop,
            gripper_release_mode=self.gripper_release_mode,
            grasp=self.robot
        )

        mask_check = self.stage1_success & ~self.stage2_success
        mask_increment = ontop_success & mask_check
        mask_reset = ~ontop_success & mask_check
        self._consecutive_ontop_steps[mask_increment] += 1
        self._consecutive_ontop_steps[mask_reset] = 0
        self.stage2_success |= (self._consecutive_ontop_steps >= self._ontop_required_consecutive_steps)

        # Log stage transitions in yellow when recording.
        changed = torch.any(self.stage1_success != self._prev_stage1_success) or torch.any(self.stage2_success != self._prev_stage2_success)
        if changed and self.is_recording():
            for env_idx in range(num_envs):
                if self.stage1_success[env_idx] != self._prev_stage1_success[env_idx] or self.stage2_success[env_idx] != self._prev_stage2_success[env_idx]:
                    print(f"\033[93m[TASK PROGRESS] Env {env_idx}: pick: {self.stage1_success[env_idx].item()}; place: {self.stage2_success[env_idx].item()}\033[0m")
        self._prev_stage1_success = self.stage1_success.clone()
        self._prev_stage2_success = self.stage2_success.clone()

    def get_task_success(self):
        """Return per-environment task success (both stages latched).

        Returns:
            torch.Tensor: Boolean tensor of shape (num_envs,).
        """
        return self.stage1_success & self.stage2_success

    def step(self, actions):
        """Step the environment and compute per-episode success labels.

        Pre-step success is cached before IsaacLab's internal auto-reset so
        terminated environments retain their final episode success value.

        Args:
            actions: Action tensor for all environments.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        # Cache before super().step(): IsaacLab auto-resets terminated envs there,
        # which clears their stage flags via reset_success_check().
        pre_step_success = self.get_task_success().clone()

        obs, reward, terminated, truncated, info = super().step(actions)

        if hasattr(self, 'reset_buf') and torch.any(self.reset_buf):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.reset_success_check(reset_env_ids)

        self.update_stage_success()

        current_success = self.get_task_success()
        final_success = torch.where(terminated, pre_step_success, current_success)

        any_done = terminated | truncated
        final_q_score = torch.where(any_done, pre_step_success.float(), current_success.float())

        info['stage1_success'] = self.stage1_success
        info['stage2_success'] = self.stage2_success
        info['success'] = final_success
        info['q_score'] = final_q_score
        return obs, reward, terminated, truncated, info

    def reset(self, warm_up=True, seed=None, env_ids=None, options=None):
        """Reset the environment and clear all stage tracking state.

        Args:
            warm_up: Run sim steps + render after reset. Set False when calling
                from an RPC server (``sim.render()`` is main-thread only).
            seed: RNG seed forwarded to the base env.
            env_ids: Specific environment indices to reset; None resets all.
            options: Extra reset options forwarded to the base env.

        Returns:
            obs_dict: Observation dictionary.
            extras: Extra info dictionary.
        """
        # Clear latched stage flags only for the envs being reset (env_ids=None resets all).
        self.reset_success_check(env_ids)
        obs_dict, extras = super().reset(warm_up=warm_up, seed=seed, env_ids=env_ids, options=options)
        n_reset = self.num_envs if env_ids is None else len(env_ids)
        print(f"\033[93m[TASK PROGRESS] {n_reset} environment(s) reset: pick: False; place: False\033[0m")
        return obs_dict, extras


# Register the environment with gymnasium (after all classes are defined)
gym.register(
    id="PutPotOnCooktop-v0",
    entry_point="yamlab.envs.tasks.put_pot_on_cooktop_manager:make_put_pot_on_cooktop_env",
)

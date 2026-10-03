"""HangMugOnTree task environment: pick a mug, hand it over, and hang it on a tree.

Three-stage bimanual task implemented with IsaacLab's ManagerBasedRLEnv framework.
Stage flags latch True once earned and are cleared on episode reset.
"""

import os
import torch
import gymnasium as gym
import math
from typing import Dict, Any

from isaaclab.managers import SceneEntityCfg

from .yam_bimanual_env import YamBimanualEnv, make_task_env
from .hang_mug_on_tree_manager_cfg import HangMugOnTreeManagerEnvCfg
from yamlab.utils.task_logic import check_pick_success, check_hang_success
from yamlab.utils.assets import load_asset_size, OBJ_Z_OFFSET
from yamlab.robot.yam import ROBOT


def make_hang_mug_on_tree_env(**kwargs):
    """Factory that creates a HangMugOnTree environment via ``make_task_env``.

    Args:
        **kwargs: Forwarded to ``HangMugOnTreeManager.__init__`` and ``make_task_env``.
            check_gripper_release_for_hang (bool): Stage 3 requires both grippers
                released. Defaults to True.

    Returns:
        HangMugOnTreeManager: The constructed environment instance.
    """
    if 'check_gripper_release_for_hang' not in kwargs:
        kwargs['check_gripper_release_for_hang'] = True

    return make_task_env(HangMugOnTreeManagerEnvCfg, HangMugOnTreeManager, **kwargs)


class HangMugOnTreeManager(YamBimanualEnv):
    """HangMugOnTree task environment.

    Three-stage bimanual task: pick up a mug (mug) with the left arm, hand it
    over to the right arm, and hang it on a mug tree (mug_tree).

    Stage 1 (Pick):     Left arm grasps mug handle and lifts it above initial height.
                        Once latched, stays True for the episode.
    Stage 2 (Handover): Right arm grasping the mug, mug elevated, left arm released.
                        Once latched, stays True for the episode.
    Stage 3 (Hang):     Mug near tree (XY), elevated above table (Z), both grippers
                        released, stays for consecutive steps (proving it's hanging).
    """

    def __init__(self, cfg: HangMugOnTreeManagerEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the HangMugOnTree environment.

        Args:
            cfg (HangMugOnTreeManagerEnvCfg): Environment configuration.
            render_mode (None or str): Rendering mode passed to ManagerBasedRLEnv.
            **kwargs: Task-specific keyword arguments:
                check_gripper_release_for_hang (bool): Require both grippers released
                    for stage 3. Default True.
                is_evaluation_mode (bool): Require 30 consecutive steps (instead of 8)
                    for stages 2 and 3. Default False.
        """
        check_gripper_release_for_hang = kwargs.pop('check_gripper_release_for_hang', True)
        is_evaluation_mode = kwargs.pop('is_evaluation_mode', False)
        kwargs.pop('pick_by_two_hands', None)

        usd_path_0 = cfg.scene.mug.spawn.usd_path
        asset_folder_0 = os.path.dirname(usd_path_0)
        size_info_0 = load_asset_size(asset_folder_0)
        mug_height = size_info_0['size']['z']
        self.mug_init_z = ROBOT.table_position[2] + mug_height / 2.0 + OBJ_Z_OFFSET

        usd_path_1 = cfg.scene.mug_tree.spawn.usd_path
        asset_folder_1 = os.path.dirname(usd_path_1)
        size_info_1 = load_asset_size(asset_folder_1)
        mug_tree_height = size_info_1['size']['z']
        self.mug_tree_init_z = ROBOT.table_position[2] + mug_tree_height / 2.0 + OBJ_Z_OFFSET

        self.mug_height = mug_height
        self.mug_tree_height = mug_tree_height

        # XY tolerance for hang check: use mug tree's XY extent as reference, minimum 10cm.
        mug_tree_xy_radius = math.sqrt(size_info_1['size']['x'] ** 2 + size_info_1['size']['y'] ** 2) / 2.0
        self.hang_xy_tolerance = max(mug_tree_xy_radius, 0.10)

        super().__init__(cfg, render_mode, **kwargs)

        self.check_gripper_release_for_hang = check_gripper_release_for_hang
        self.is_evaluation_mode = is_evaluation_mode

        num_envs = self.num_envs
        self.stage1_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.stage2_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.stage3_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._prev_stage1_success = self.stage1_success.clone()
        self._prev_stage2_success = self.stage2_success.clone()
        self._prev_stage3_success = self.stage3_success.clone()
        self._consecutive_pick_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._consecutive_handover_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._consecutive_hang_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

        self._pick_required_consecutive_steps = 8
        self._handover_required_consecutive_steps = 30 if is_evaluation_mode else 8
        self._hang_required_consecutive_steps = 30

        # Stage 3 stability check: Z position must stay within threshold of the anchor
        # recorded on the first hang frame. XY motion (dangling/swinging) is ignored;
        # a free-falling mug drops >3cm within ~2 control steps and is rejected.
        self._hang_stability_threshold = 0.03  # 3 cm (Z-axis only)
        self._stage3_anchor_mug_pos = torch.zeros(num_envs, 3, device=self.device)

        # One-shot per-env flags: set True when the stage first latches, cleared on reset.
        # Prevents compute_sparse_hang_reward from re-crediting +0.5 on every latched step.
        self._stage2_reward_given = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._stage3_reward_given = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # One-shot latches for runtime left-finger friction modulation (per-env, cleared on reset).
        # _friction_raised fires "left_grasp" event on first detected left-arm grasp (HIGH friction).
        # _friction_released fires "handover" event when both arms grasp (drops back to LOW).
        self._friction_raised = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._friction_released = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        print(f"[INFO] HangMugOnTree Manager Environment initialized")
        print(f"[INFO] - Three-stage task: pick -> handover -> hang")
        print(f"[INFO] - Hang XY tolerance: {self.hang_xy_tolerance:.4f}m")
        print(f"[INFO] - Stage 1: left arm picks mug (height + left grasp)")
        print(f"[INFO] - Stage 2: handover complete (right grasp + elevated + left released)")
        print(f"[INFO] - Stage 3: mug hanging on tree (XY + height + both grippers released)")
        print(f"[INFO] - Stage 3 stability: mug Z must stay within {self._hang_stability_threshold*100:.1f} cm of initial hang Z for {self._hang_required_consecutive_steps} consecutive steps")
        print(f"[INFO] - Check gripper release for hang: {self.check_gripper_release_for_hang}")
        print(f"[INFO] - Evaluation mode: {self.is_evaluation_mode}")
        print("\033[93m[TASK PROGRESS] pick: False; handover: False; hang: False\033[0m")

    def reset_success_check(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset all stage success flags and counters for the given environments.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.stage1_success[env_ids] = False
        self.stage2_success[env_ids] = False
        self.stage3_success[env_ids] = False
        self._prev_stage1_success[env_ids] = False
        self._prev_stage2_success[env_ids] = False
        self._prev_stage3_success[env_ids] = False
        self._consecutive_pick_steps[env_ids] = 0
        self._consecutive_handover_steps[env_ids] = 0
        self._consecutive_hang_steps[env_ids] = 0
        self._stage3_anchor_mug_pos[env_ids] = 0.0
        self._stage2_reward_given[env_ids] = False
        self._stage3_reward_given[env_ids] = False
        self._friction_raised[env_ids] = False
        self._friction_released[env_ids] = False

    def get_current_task_info(self) -> Dict[str, Any]:
        """Return current task state for env 0, including object poses and stage success flags.

        Returns:
            Dict with mug/mug_tree position and orientation, episode count, and intermediate success.
        """
        mug = self.scene["mug"]
        mug_pos = mug.data.root_pos_w.torch[0] if hasattr(mug.data, 'root_pos_w') else torch.zeros(3)
        mug_quat = mug.data.root_quat_w.torch[0] if hasattr(mug.data, 'root_quat_w') else torch.tensor([0, 0, 0, 1])  # xyzw identity

        mug_tree = self.scene["mug_tree"]
        mug_tree_pos = mug_tree.data.root_pos_w.torch[0] if hasattr(mug_tree.data, 'root_pos_w') else torch.zeros(3)
        mug_tree_quat = mug_tree.data.root_quat_w.torch[0] if hasattr(mug_tree.data, 'root_quat_w') else torch.tensor([0, 0, 0, 1])  # xyzw identity

        stage1_success = self.stage1_success[0].item()
        stage2_success = self.stage2_success[0].item()
        stage3_success = self.stage3_success[0].item()
        task_success = stage1_success and stage2_success and stage3_success

        return {
            "mug_position": mug_pos.cpu().numpy().tolist(),
            "mug_orientation": mug_quat.cpu().numpy().tolist(),
            "mug_tree_position": mug_tree_pos.cpu().numpy().tolist(),
            "mug_tree_orientation": mug_tree_quat.cpu().numpy().tolist(),
            "total_objs": 2,
            "episode_count": self._episode_count,
            "intermediate_success": {
                "pick": bool(stage1_success),
                "handover": bool(stage2_success),
                "hang": bool(stage3_success),
                "task_success": bool(task_success),
            },
        }

    def update_stage_success(self):
        """Evaluate and latch the three task stages each control step.

        Stage 1 (pick):     Mug lifted >5cm above initial height with LEFT arm grasping.
                            Once latched True, stays True for the episode.
        Stage 2 (handover): Right arm grasping the mug, mug elevated above initial height,
                            left arm released. Once latched True, stays True.
        Stage 3 (hang):     Mug near tree (XY), elevated above table, both grippers released,
                            stays for consecutive steps.
        """
        # Raise left-finger friction to HIGH the first step left arm grasps the mug,
        # independent of stage progression. Latched so the event fires at most once per episode.
        left_grasping_now, _ = self.robot.is_grasping()
        newly_raise = left_grasping_now & ~self._friction_raised
        if torch.any(newly_raise):
            ids = newly_raise.nonzero(as_tuple=False).squeeze(-1)
            print(
                f"\033[96m[GRASP FRICTION] Env(s) {ids.tolist()}: "
                "left arm grasping detected - firing 'left_grasp' event to "
                "raise left-finger friction.\033[0m"
            )
            self.event_manager.apply(mode="left_grasp", env_ids=ids)
            self._friction_raised[newly_raise] = True

        pick_success = check_pick_success(
            self,
            SceneEntityCfg("mug"),
            init_obj_height=self.mug_init_z,
            pick_threshold=0.05,
            check_contact=True,
            pick_arm="left",
            grasp=self.robot
        )

        mask_increment = pick_success & ~self.stage1_success
        mask_reset = ~pick_success & ~self.stage1_success
        self._consecutive_pick_steps[mask_increment] += 1
        self._consecutive_pick_steps[mask_reset] = 0
        self.stage1_success |= (self._consecutive_pick_steps >= self._pick_required_consecutive_steps)

        # Stage 2: right arm grasping + mug elevated + left arm NOT grasping.
        # Only checked after stage 1 is latched.
        if torch.any(self.stage1_success & ~self.stage2_success):
            mask_check = self.stage1_success & ~self.stage2_success
            left_grasping, right_grasping = self.robot.is_grasping()

            # Drop left-arm friction back to LOW on the first step both arms grasp simultaneously,
            # so the right arm can take over cleanly. Latched per episode.
            both_grasping = left_grasping & right_grasping
            newly_release = mask_check & both_grasping & ~self._friction_released
            if torch.any(newly_release):
                ids = newly_release.nonzero(as_tuple=False).squeeze(-1)
                print(
                    f"\033[96m[HANDOVER FRICTION] Env(s) {ids.tolist()}: "
                    "both arms grasping detected - firing 'handover' event to "
                    "drop left-finger friction.\033[0m"
                )
                self.event_manager.apply(mode="handover", env_ids=ids)
                self._friction_released[newly_release] = True

            mug = self.scene["mug"]
            mug_z = mug.data.root_pos_w.torch[:, 2]
            elevated = mug_z > (self.mug_init_z + 0.05)  # 5cm above resting

            handover_success = right_grasping & elevated & (~left_grasping)

            mask_increment = handover_success & mask_check
            mask_reset = ~handover_success & mask_check
            self._consecutive_handover_steps[mask_increment] += 1
            self._consecutive_handover_steps[mask_reset] = 0
            self.stage2_success |= (self._consecutive_handover_steps >= self._handover_required_consecutive_steps)

        # Stage 3: mug near tree XY, elevated, both grippers released, Z-stable.
        # Only checked after stage 2 is latched. Mug Z must stay within
        # _hang_stability_threshold of the anchor set on the first hang frame.
        if torch.any(self.stage2_success & ~self.stage3_success):
            hang_success = check_hang_success(
                self,
                SceneEntityCfg("mug"),
                SceneEntityCfg("mug_tree"),
                mug_height=self.mug_height,
                table_z=ROBOT.table_position[2],
                xy_tolerance=self.hang_xy_tolerance,
                min_hang_height=0.05,
                check_gripper_release=self.check_gripper_release_for_hang,
                gripper_release_mode="both",
                grasp=self.robot
            )

            mask_check = self.stage2_success & ~self.stage3_success

            mug_pos = self.scene["mug"].data.root_pos_w.torch

            # Record the mug position as stability anchor on the first frame of each hang attempt.
            # Only the Z component is checked; XY drift from dangling is ignored.
            newly_starting = mask_check & hang_success & (self._consecutive_hang_steps == 0)
            self._stage3_anchor_mug_pos[newly_starting] = mug_pos[newly_starting]

            dz = torch.abs(mug_pos[:, 2] - self._stage3_anchor_mug_pos[:, 2])
            stable = dz < self._hang_stability_threshold

            mask_increment = mask_check & hang_success & (
                (self._consecutive_hang_steps == 0) | stable
            )
            mask_reset = mask_check & (
                ~hang_success
                | ((self._consecutive_hang_steps > 0) & ~stable)
            )
            self._consecutive_hang_steps[mask_increment] += 1
            self._consecutive_hang_steps[mask_reset] = 0
            self.stage3_success |= (self._consecutive_hang_steps >= self._hang_required_consecutive_steps)

        changed = (
            torch.any(self.stage1_success != self._prev_stage1_success)
            or torch.any(self.stage2_success != self._prev_stage2_success)
            or torch.any(self.stage3_success != self._prev_stage3_success)
        )
        if changed and self.is_recording():
            for env_idx in range(self.num_envs):
                if (self.stage1_success[env_idx] != self._prev_stage1_success[env_idx]
                    or self.stage2_success[env_idx] != self._prev_stage2_success[env_idx]
                    or self.stage3_success[env_idx] != self._prev_stage3_success[env_idx]):
                    print(f"\033[93m[TASK PROGRESS] Env {env_idx}: "
                          f"pick: {self.stage1_success[env_idx].item()}; "
                          f"handover: {self.stage2_success[env_idx].item()}; "
                          f"hang: {self.stage3_success[env_idx].item()}\033[0m")
        self._prev_stage1_success = self.stage1_success.clone()
        self._prev_stage2_success = self.stage2_success.clone()
        self._prev_stage3_success = self.stage3_success.clone()

    def get_task_success(self):
        """Return per-env task success: True iff all three stages are latched.

        Returns:
            Boolean tensor of shape (num_envs,).
        """
        return self.stage1_success & self.stage2_success & self.stage3_success

    def step(self, actions):
        """Advance the simulation one control step and update stage success flags.

        Args:
            actions: Joint position actions of shape (num_envs, action_dim).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info) with stage success
            and q_score fields added to info.
        """
        pre_step_success = self.get_task_success().clone()
        pre_step_stage2 = self.stage2_success.clone()
        pre_step_stage3 = self.stage3_success.clone()

        obs, reward, terminated, truncated, info = super().step(actions)

        # Compute q_score before resetting stage flags so terminated envs
        # retain their final episode values (0 / 0.5 / 1.0, unscaled by dt).
        q_score = 0.5 * pre_step_stage2.float() + 0.5 * pre_step_stage3.float()

        reset_env_ids = None
        if hasattr(self, 'reset_buf') and torch.any(self.reset_buf):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.reset_success_check(reset_env_ids)

        self.update_stage_success()

        current_success = self.get_task_success()
        final_success = torch.where(terminated, pre_step_success, current_success)

        # Both terminated and truncated envs use pre-reset q_score values since
        # reset_success_check() has already cleared stage flags for those envs.
        current_q_score = 0.5 * self.stage2_success.float() + 0.5 * self.stage3_success.float()
        any_done = terminated | truncated
        final_q_score = torch.where(any_done, q_score, current_q_score)

        info['stage1_success'] = self.stage1_success
        info['stage2_success'] = self.stage2_success
        info['stage3_success'] = self.stage3_success
        info['success'] = final_success
        info['q_score'] = final_q_score
        return obs, reward, terminated, truncated, info

    def reset(self, warm_up=True, seed=None, env_ids=None, options=None):
        """Reset all environments and clear stage success state.

        Args:
            warm_up: Whether to run warm-up simulation steps after reset.
            seed: Optional random seed.
            env_ids: Optional subset of environment indices to reset.
            options: Optional reset options passed to the base class.

        Returns:
            Tuple of (obs_dict, extras).
        """
        # Clear latched stage flags only for the envs being reset (env_ids=None resets all).
        self.reset_success_check(env_ids)
        obs_dict, extras = super().reset(warm_up=warm_up, seed=seed, env_ids=env_ids, options=options)
        n_reset = self.num_envs if env_ids is None else len(env_ids)
        print(f"\033[93m[TASK PROGRESS] {n_reset} environment(s) reset: pick: False; handover: False; hang: False\033[0m")
        return obs_dict, extras


gym.register(
    id="HangMugOnTree-v0",
    entry_point="yamlab.envs.tasks.hang_mug_on_tree_manager:make_hang_mug_on_tree_env",
)

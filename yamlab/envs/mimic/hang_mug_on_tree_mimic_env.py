"""MimicGen-compatible environment for the HangMugOnTree bimanual task.

Implements the required MimicGen API (get_robot_eef_pose, target_eef_pose_to_action,
action_to_target_eef_pose, actions_to_gripper_actions, get_object_poses,
get_subtask_term_signals) on top of HangMugOnTreeManager.  Subtask termination
signals are latched and sequentially gated across 11 stages:
left_grasping -> mug_transported -> transport_near_end -> mug_at_handover
-> both_grasping -> left_released -> handover_done -> left_backward_done
-> post_handover_settled -> right_motion_started -> mug_hung.
"""

import gymnasium as gym
import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils

import yamlab.utils.mimic_patches as mimic_utils
from yamlab.envs.mimic.yam_mimic_env import YamMimicEnv

from yamlab.envs.tasks.hang_mug_on_tree_manager import HangMugOnTreeManager
from yamlab.envs.tasks.yam_bimanual_env import make_task_env
from yamlab.robot.yam import ROBOT
from .hang_mug_on_tree_mimic_env_cfg import HangMugOnTreeMimicEnvCfg

# Per-task tunable thresholds and time delays are read from
# configs/tasks/hang_mug_on_tree.yaml (mimic_signals:) via the layered loader.
from yamlab.configs import get_task_config

_SIG = get_task_config("HangMugOnTree-Mimic-v0")["mimic_signals"]
MUG_TRANSPORTED_X_THRESHOLD = _SIG["mug_transported_x_threshold"]
MUG_AT_HANDOVER_X_THRESHOLD = _SIG["mug_at_handover_x_threshold"]
MUG_AT_HANDOVER_DELAY = _SIG["mug_at_handover_delay"]
POST_HANDOVER_SETTLED_DELAY = _SIG["post_handover_settled_delay"]
RIGHT_MOTION_GATE_DELAY = _SIG["right_motion_gate_delay"]
TRANSPORT_NEAR_END_DELAY = _SIG["transport_near_end_delay"]
LEFT_BACKWARD_DONE_DELAY = _SIG["left_backward_done_delay"]
RIGHT_MOTION_STARTED_WINDOW_FRAMES = _SIG["right_motion_started_window_frames"]
RIGHT_MOTION_STARTED_DELTA_M = _SIG["right_motion_started_delta_m"]
RIGHT_MOTION_WINDOW_DONE_DELAY = _SIG["right_motion_window_done_delay"]


def make_hang_mug_on_tree_mimic_env(**kwargs):
    """Factory function for the HangMugOnTree-Mimic gymnasium environment.

    Args:
        **kwargs: Configuration parameters forwarded to the env constructor,
            or applied to the cfg object when one is provided directly.

    Returns:
        HangMugOnTreeMimicEnv: The constructed environment.
    """
    if 'check_gripper_release_for_hang' not in kwargs:
        kwargs['check_gripper_release_for_hang'] = True

    cfg = kwargs.pop('cfg', None)

    if cfg is not None:
        physics_kwargs = {}
        if 'enable_self_collisions' in kwargs:
            physics_kwargs['enable_self_collisions'] = kwargs.pop('enable_self_collisions')

        for key, value in physics_kwargs.items():
            if hasattr(cfg.sim.physics, key):
                setattr(cfg.sim.physics, key, value)

        config_only_keys = [
            'assets_instance_paths', 'objects_randomization',
            'init_joint_pos_randomization', 'teleoperation',
            'observation_modalities', 'num_envs', 'data_generation',
            'check_gripper_release_for_hang',
        ]
        constructor_kwargs = {k: v for k, v in kwargs.items() if k not in config_only_keys}

        return HangMugOnTreeMimicEnv(cfg, **constructor_kwargs)
    else:
        return make_task_env(HangMugOnTreeMimicEnvCfg, HangMugOnTreeMimicEnv, **kwargs)


class HangMugOnTreeMimicEnv(HangMugOnTreeManager, YamMimicEnv):
    """MimicGen-compatible environment for the HangMugOnTree task.

    Inherits task logic from HangMugOnTreeManager and the MimicGen interface
    from ManagerBasedRLMimicEnv.  All subtask termination signals are latched
    (once True, stay True until reset) and sequentially gated.
    """

    def __init__(self, cfg, **kwargs):
        """Initialize the bimanual mimic environment."""
        mimic_utils.hold_sequential_constraint_at_zero()
        mimic_utils.drop_unused_trajectory_buffers()

        # Latch state tensors (None until after super().__init__ allocates num_envs).
        self._subtask_latch_left_grasping = None
        self._subtask_latch_mug_transported = None
        self._subtask_latch_transport_near_end = None
        self._subtask_latch_mug_at_handover_raw = None
        self._subtask_latch_mug_at_handover = None
        self._subtask_latch_both_grasping = None
        self._subtask_latch_left_released = None
        self._subtask_latch_handover_done = None
        self._subtask_latch_left_backward_done = None
        self._subtask_latch_post_handover_settled = None
        self._subtask_latch_right_motion_gate = None
        self._subtask_latch_right_motion_started = None
        self._subtask_latch_right_motion_window_done = None
        self._subtask_latch_mug_hung = None

        # Per-env step counters for time-delay signals; each increments once
        # the corresponding base latch is True at the start of the step.
        self._steps_since_mug_transported = None
        self._steps_since_mug_at_handover_raw = None
        self._steps_since_handover_done = None
        self._steps_since_right_motion_started = None

        # Rolling buffer of right wrist (link_6) world-frame XYZ over the last
        # RIGHT_MOTION_STARTED_WINDOW_FRAMES steps for right_motion_started detection.
        self._right_eef_xyz_history = None

        # Rolling buffer of right arm joint2 angle over the same window.
        # right_motion_started requires a positive joint2 delta to reject the
        # post-handover backward retreat (which rotates joint2 negatively).
        self._right_joint2_history = None

        # Tracks whether the right_motion_gate has ever opened in this episode.
        # On the first frame the gate opens the XYZ/joint2 buffers are reseeded
        # so pre-gate motion cannot leak into the rolling-window check.
        self._right_motion_gate_was_open = None

        super().__init__(cfg, **kwargs)

        self._subtask_latch_left_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_mug_transported = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_transport_near_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_mug_at_handover_raw = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_mug_at_handover = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_both_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_left_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_handover_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_left_backward_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_post_handover_settled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_right_motion_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_right_motion_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_right_motion_window_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._subtask_latch_mug_hung = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._steps_since_mug_transported = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_mug_at_handover_raw = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_handover_done = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_right_motion_started = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._right_eef_xyz_history = torch.zeros(
            self.num_envs, RIGHT_MOTION_STARTED_WINDOW_FRAMES, 3,
            device=self.device, dtype=torch.float32,
        )
        self._right_joint2_history = torch.zeros(
            self.num_envs, RIGHT_MOTION_STARTED_WINDOW_FRAMES,
            device=self.device, dtype=torch.float32,
        )
        self._right_motion_gate_was_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        print("[INFO] HangMugOnTreeMimicEnv initialized (using J-PARSE IK)")
        print("[INFO] - Subtask signals: left_grasping -> mug_transported -> transport_near_end -> mug_at_handover -> both_grasping -> left_released -> handover_done -> left_backward_done -> post_handover_settled -> right_motion_started -> mug_hung")
        print(f"[INFO] - mug_transported threshold: mug X within {MUG_TRANSPORTED_X_THRESHOLD}m of left arm base X ({ROBOT.arm_position('left')[0]})")
        print(f"[INFO] - transport_near_end fires {TRANSPORT_NEAR_END_DELAY} source steps after mug_transported")
        print(f"[INFO] - mug_at_handover_raw threshold: mug X within {MUG_AT_HANDOVER_X_THRESHOLD}m of left arm base X ({ROBOT.arm_position('left')[0]})")
        print(f"[INFO] - mug_at_handover fires {MUG_AT_HANDOVER_DELAY} source steps after mug_at_handover_raw")
        print(f"[INFO] - left_backward_done fires {LEFT_BACKWARD_DONE_DELAY} source steps after handover_done")
        print(f"[INFO] - post_handover_settled fires {POST_HANDOVER_SETTLED_DELAY} source steps after handover_done (terminates left L5 camera approach)")
        print(f"[INFO] - right_motion_gate opens {RIGHT_MOTION_GATE_DELAY} source steps after handover_done (gates right_motion_started)")
        print(f"[INFO] - right_motion_started fires when right link_6 forward (+X) displacement >= {RIGHT_MOTION_STARTED_DELTA_M} m AND joint2 rotates forward (positive delta) over {RIGHT_MOTION_STARTED_WINDOW_FRAMES} steps, gated on right_motion_gate")
        print(f"[INFO] - right_motion_window_done fires {RIGHT_MOTION_WINDOW_DONE_DELAY} source steps after right_motion_started")

    def reset_subtask_latch_states(self, env_ids: Sequence[int] | None = None):
        """Reset subtask latch states and step counters for the specified environments.

        Args:
            env_ids (Sequence[int] | None): Environment indices to reset. None resets all.
        """
        if env_ids is None:
            self._subtask_latch_left_grasping.fill_(False)
            self._subtask_latch_mug_transported.fill_(False)
            self._subtask_latch_transport_near_end.fill_(False)
            self._subtask_latch_mug_at_handover_raw.fill_(False)
            self._subtask_latch_mug_at_handover.fill_(False)
            self._subtask_latch_both_grasping.fill_(False)
            self._subtask_latch_left_released.fill_(False)
            self._subtask_latch_handover_done.fill_(False)
            self._subtask_latch_left_backward_done.fill_(False)
            self._subtask_latch_post_handover_settled.fill_(False)
            self._subtask_latch_right_motion_gate.fill_(False)
            self._subtask_latch_right_motion_started.fill_(False)
            self._subtask_latch_right_motion_window_done.fill_(False)
            self._subtask_latch_mug_hung.fill_(False)
            self._steps_since_mug_transported.fill_(0)
            self._steps_since_mug_at_handover_raw.fill_(0)
            self._steps_since_handover_done.fill_(0)
            self._steps_since_right_motion_started.fill_(0)
            self._right_eef_xyz_history.fill_(0)
            self._right_joint2_history.fill_(0)
            self._right_motion_gate_was_open.fill_(False)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            self._subtask_latch_left_grasping[env_ids] = False
            self._subtask_latch_mug_transported[env_ids] = False
            self._subtask_latch_transport_near_end[env_ids] = False
            self._subtask_latch_mug_at_handover_raw[env_ids] = False
            self._subtask_latch_mug_at_handover[env_ids] = False
            self._subtask_latch_both_grasping[env_ids] = False
            self._subtask_latch_left_released[env_ids] = False
            self._subtask_latch_handover_done[env_ids] = False
            self._subtask_latch_left_backward_done[env_ids] = False
            self._subtask_latch_post_handover_settled[env_ids] = False
            self._subtask_latch_right_motion_gate[env_ids] = False
            self._subtask_latch_right_motion_started[env_ids] = False
            self._subtask_latch_right_motion_window_done[env_ids] = False
            self._subtask_latch_mug_hung[env_ids] = False
            self._steps_since_mug_transported[env_ids] = 0
            self._steps_since_mug_at_handover_raw[env_ids] = 0
            self._steps_since_handover_done[env_ids] = 0
            self._steps_since_right_motion_started[env_ids] = 0
            self._right_eef_xyz_history[env_ids] = 0
            self._right_joint2_history[env_ids] = 0
            self._right_motion_gate_was_open[env_ids] = False

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments and clear latch states."""
        super()._reset_idx(env_ids)
        self.reset_subtask_latch_states(env_ids)

    def get_subtask_term_signals(self) -> dict[str, torch.Tensor]:
        """Get latched subtask termination signals with sequential gating.

        Signals latch on first True and remain True until env reset.  Each signal
        can only become True if the previous signal was already latched in a prior
        step (MimicGen requirement for sequential annotation).

        Signal sequence and definitions:
            left_grasping:          left arm contacts mug (handle).
            mug_transported:        mug lifted >5 cm (requires left_grasping prev).
            transport_near_end:     TRANSPORT_NEAR_END_DELAY steps after mug_transported; time-delay only.
            mug_at_handover:        mug X within MUG_AT_HANDOVER_X_THRESHOLD of left base X for MUG_AT_HANDOVER_DELAY steps (requires mug_transported prev).
            both_grasping:          both arms grasping mug (requires mug_transported prev).
            left_released:          right grasping AND left NOT grasping AND mug elevated (requires both_grasping prev).
            handover_done:          left_released AND left EEF >= 10 cm from mug (requires left_released prev).
            left_backward_done:     LEFT_BACKWARD_DONE_DELAY steps after handover_done; time-delay only.
            post_handover_settled:  POST_HANDOVER_SETTLED_DELAY steps after handover_done; terminates left L5 camera approach.
            right_motion_gate:      RIGHT_MOTION_GATE_DELAY steps after handover_done; internal gate for right_motion_started.
            right_motion_started:   right link_6 forward (+X) displacement >= threshold AND joint2 positive delta over a rolling window, gated on right_motion_gate.
            right_motion_window_done: RIGHT_MOTION_WINDOW_DONE_DELAY steps after right_motion_started; time-delay only.
            mug_hung:               mug near tree XY AND elevated AND right arm NOT grasping (requires post_handover_settled prev).

        Returns:
            dict[str, torch.Tensor]: Maps signal name -> latched signal,
                bool tensor of shape (num_envs,).
        """
        prev_left_grasping = self._subtask_latch_left_grasping.clone()
        prev_mug_transported = self._subtask_latch_mug_transported.clone()
        prev_mug_at_handover_raw = self._subtask_latch_mug_at_handover_raw.clone()
        prev_both_grasping = self._subtask_latch_both_grasping.clone()
        prev_left_released = self._subtask_latch_left_released.clone()
        prev_handover_done = self._subtask_latch_handover_done.clone()
        prev_post_handover_settled = self._subtask_latch_post_handover_settled.clone()
        prev_right_motion_gate = self._subtask_latch_right_motion_gate.clone()
        prev_right_motion_started = self._subtask_latch_right_motion_started.clone()

        # Advance time-delay counters for envs whose base latch was already set.
        self._steps_since_mug_transported[prev_mug_transported] += 1
        self._steps_since_mug_at_handover_raw[prev_mug_at_handover_raw] += 1
        self._steps_since_handover_done[prev_handover_done] += 1
        self._steps_since_right_motion_started[prev_right_motion_started] += 1

        # On the first frame the right_motion_gate opens, reseed both rolling
        # buffers with the current values so pre-gate motion (the backward retreat)
        # cannot falsely trigger right_motion_started immediately after the gate clears.
        gate_just_opened = prev_right_motion_gate & ~self._right_motion_gate_was_open
        if gate_just_opened.any():
            right_arm = self.scene["right_arm"]
            gripper_body_idx = right_arm.body_names.index("link_6")
            current_xyz_seed = right_arm.data.body_pos_w.torch[:, gripper_body_idx, :3]
            self._right_eef_xyz_history[gate_just_opened] = (
                current_xyz_seed[gate_just_opened].unsqueeze(1).expand(-1, RIGHT_MOTION_STARTED_WINDOW_FRAMES, -1)
            )
            joint2_idx = right_arm.joint_names.index("joint2")
            current_j2_seed = right_arm.data.joint_pos.torch[:, joint2_idx]
            self._right_joint2_history[gate_just_opened] = (
                current_j2_seed[gate_just_opened].unsqueeze(1).expand(-1, RIGHT_MOTION_STARTED_WINDOW_FRAMES)
            )
        self._right_motion_gate_was_open |= prev_right_motion_gate

        left_grasping_raw, right_grasping_raw = self.robot.is_grasping(
            normal_force_thresh=0.1, env_ids=None,
        )
        mug_transported_raw = self._check_mug_transported_raw()
        transport_near_end_raw = self._steps_since_mug_transported >= TRANSPORT_NEAR_END_DELAY
        mug_at_handover_proximity_raw = self._check_mug_at_handover_proximity_raw()
        mug_at_handover_delay_done_raw = self._steps_since_mug_at_handover_raw >= MUG_AT_HANDOVER_DELAY
        both_grasping_raw = left_grasping_raw & right_grasping_raw
        left_released_raw = self._check_left_released_raw()
        handover_done_raw = self._check_handover_done_raw()
        left_backward_done_raw = self._steps_since_handover_done >= LEFT_BACKWARD_DONE_DELAY
        post_handover_settled_raw = self._steps_since_handover_done >= POST_HANDOVER_SETTLED_DELAY
        right_motion_gate_raw = self._steps_since_handover_done >= RIGHT_MOTION_GATE_DELAY
        right_motion_started_raw = self._check_right_motion_started_raw()
        right_motion_window_done_raw = self._steps_since_right_motion_started >= RIGHT_MOTION_WINDOW_DONE_DELAY
        mug_hung_raw = self._check_mug_hung_raw()

        # Sequential gating: each signal is additionally masked by the previous
        # step's latched state so signals can only fire in order.
        # right_motion_started is gated on prev_right_motion_gate (pure time delay)
        # so the brief post-handover transient cannot fire it.
        # mug_hung is gated on prev_post_handover_settled; the ordering
        # right_motion_started < mug_hung is guaranteed by the data-collection protocol.
        left_grasping_current = left_grasping_raw
        mug_transported_current = mug_transported_raw & prev_left_grasping
        transport_near_end_current = transport_near_end_raw & prev_mug_transported
        # Two-stage latch: mug_at_handover_raw latches the first step the proximity
        # zone is entered; mug_at_handover fires MUG_AT_HANDOVER_DELAY steps later.
        # This keeps the counter monotonic even if the mug briefly leaves the zone.
        mug_at_handover_raw_current = mug_at_handover_proximity_raw & prev_mug_transported
        mug_at_handover_current = mug_at_handover_delay_done_raw & prev_mug_at_handover_raw
        both_grasping_current = both_grasping_raw & prev_mug_transported
        left_released_current = left_released_raw & prev_both_grasping
        handover_done_current = handover_done_raw & prev_left_released
        left_backward_done_current = left_backward_done_raw & prev_handover_done
        post_handover_settled_current = post_handover_settled_raw & prev_handover_done
        right_motion_gate_current = right_motion_gate_raw & prev_handover_done
        right_motion_started_current = right_motion_started_raw & prev_right_motion_gate
        right_motion_window_done_current = right_motion_window_done_raw & prev_right_motion_started
        mug_hung_current = mug_hung_raw & prev_post_handover_settled

        self._subtask_latch_left_grasping |= left_grasping_current
        self._subtask_latch_mug_transported |= mug_transported_current
        self._subtask_latch_transport_near_end |= transport_near_end_current
        self._subtask_latch_mug_at_handover_raw |= mug_at_handover_raw_current
        self._subtask_latch_mug_at_handover |= mug_at_handover_current
        self._subtask_latch_both_grasping |= both_grasping_current
        self._subtask_latch_left_released |= left_released_current
        self._subtask_latch_handover_done |= handover_done_current
        self._subtask_latch_left_backward_done |= left_backward_done_current
        self._subtask_latch_post_handover_settled |= post_handover_settled_current
        self._subtask_latch_right_motion_gate |= right_motion_gate_current
        self._subtask_latch_right_motion_started |= right_motion_started_current
        self._subtask_latch_right_motion_window_done |= right_motion_window_done_current
        self._subtask_latch_mug_hung |= mug_hung_current

        return {
            "left_grasping": self._subtask_latch_left_grasping.clone(),
            "mug_transported": self._subtask_latch_mug_transported.clone(),
            "transport_near_end": self._subtask_latch_transport_near_end.clone(),
            "mug_at_handover": self._subtask_latch_mug_at_handover.clone(),
            "both_grasping": self._subtask_latch_both_grasping.clone(),
            "left_released": self._subtask_latch_left_released.clone(),
            "handover_done": self._subtask_latch_handover_done.clone(),
            "left_backward_done": self._subtask_latch_left_backward_done.clone(),
            "post_handover_settled": self._subtask_latch_post_handover_settled.clone(),
            "right_motion_started": self._subtask_latch_right_motion_started.clone(),
            "right_motion_window_done": self._subtask_latch_right_motion_window_done.clone(),
            "mug_hung": self._subtask_latch_mug_hung.clone(),
        }

    def _check_mug_transported_raw(self) -> torch.Tensor:
        """Return True when the mug is lifted >5 cm above its initial resting height.

        No horizontal distance check is applied; the L2 subtask handles all horizontal
        transport via interpolation to the fixed handover pose, keeping L1 short (grasp+lift only).

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        mug = self.scene["mug"]
        mug_z = mug.data.root_pos_w.torch[:, 2]
        lifted = (mug_z - self.mug_init_z) > 0.05
        return lifted

    def _check_mug_at_handover_proximity_raw(self) -> torch.Tensor:
        """Return True when the mug X position is within MUG_AT_HANDOVER_X_THRESHOLD of the left arm base.

        Only the world X axis (front-back) is checked; Y (left-right) and Z (height) are ignored.
        The latched mug_at_handover signal fires MUG_AT_HANDOVER_DELAY steps after this first latches.

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        mug = self.scene["mug"]
        mug_x = mug.data.root_pos_w.torch[:, 0]
        left_arm_x = ROBOT.arm_position('left')[0]
        return torch.abs(mug_x - left_arm_x) < MUG_AT_HANDOVER_X_THRESHOLD

    def _check_right_motion_started_raw(self) -> torch.Tensor:
        """Detect the right arm starting its forward tree approach via a rolling-window check.

        Both criteria must hold over the last RIGHT_MOTION_STARTED_WINDOW_FRAMES steps:
        (a) right link_6 moved >= RIGHT_MOTION_STARTED_DELTA_M FORWARD along world +X
            (negative/backward displacement is rejected).
        (b) right joint2 has a positive delta over the window (rejects the backward retreat,
            which rotates joint2 negatively; joint2 limit [0, pi], default 0 = retracted).

        Side effect: rolls the XYZ and joint2 history buffers each call.

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        right_arm = self.scene["right_arm"]
        gripper_body_idx = right_arm.body_names.index("link_6")
        current_xyz = right_arm.data.body_pos_w.torch[:, gripper_body_idx, :3]  # (num_envs, 3)

        joint2_idx = right_arm.joint_names.index("joint2")
        current_j2 = right_arm.data.joint_pos.torch[:, joint2_idx]  # (num_envs,)

        # Read oldest values before rolling so the comparison spans the full window.
        oldest_xyz = self._right_eef_xyz_history[:, 0, :].clone()
        oldest_j2 = self._right_joint2_history[:, 0].clone()

        self._right_eef_xyz_history[:, :-1, :] = self._right_eef_xyz_history[:, 1:, :].clone()
        self._right_eef_xyz_history[:, -1, :] = current_xyz
        self._right_joint2_history[:, :-1] = self._right_joint2_history[:, 1:].clone()
        self._right_joint2_history[:, -1] = current_j2

        # Criterion (a): forward (+X) displacement meets or exceeds threshold.
        forward_disp = current_xyz[:, 0] - oldest_xyz[:, 0]  # (num_envs,) world X
        link6_moved_forward = forward_disp >= RIGHT_MOTION_STARTED_DELTA_M

        # Criterion (b): joint2 rotated forward (positive delta) over the window.
        joint2_forward = (current_j2 - oldest_j2) > 0.0

        return link6_moved_forward & joint2_forward

    def _check_left_released_raw(self) -> torch.Tensor:
        """Return True when the right arm is grasping, left is not, and mug is elevated.

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        left_grasping, right_grasping = self.robot.is_grasping(
            normal_force_thresh=0.1, env_ids=None,
        )

        mug = self.scene["mug"]
        mug_z = mug.data.root_pos_w.torch[:, 2]
        elevated = (mug_z - self.mug_init_z) > 0.05

        return right_grasping & (~left_grasping) & elevated

    def _check_handover_done_raw(self) -> torch.Tensor:
        """Return True when the handover is complete and the left EEF has cleared the mug.

        Extends left_released by requiring the left EEF to be >= 10 cm from the mug, ensuring the
        left arm's retraction stays inside the handover subtask (L3) rather than leaking into L4.

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        left_grasping, right_grasping = self.robot.is_grasping(
            normal_force_thresh=0.1, env_ids=None,
        )

        mug = self.scene["mug"]
        mug_pos = mug.data.root_pos_w.torch
        mug_z = mug_pos[:, 2]
        elevated = (mug_z - self.mug_init_z) > 0.05

        left_arm = self.scene["left_arm"]
        left_eef_idx = left_arm.num_bodies - 1
        left_eef_pos = left_arm.data.body_pos_w.torch[:, left_eef_idx, :]
        left_eef_dist = torch.norm(left_eef_pos - mug_pos, dim=-1)
        left_clear = left_eef_dist > 0.10

        return right_grasping & (~left_grasping) & elevated & left_clear

    def _check_mug_hung_raw(self) -> torch.Tensor:
        """Return True when the mug is on the tree and the right arm has released.

        Criteria: mug XY within hang_xy_tolerance of tree XY, mug above table resting height, right arm not grasping.

        Returns:
            torch.Tensor: bool, shape (num_envs,).
        """
        mug = self.scene["mug"]
        mug_tree = self.scene["mug_tree"]

        mug_pos = mug.data.root_pos_w.torch
        tree_pos = mug_tree.data.root_pos_w.torch

        xy_dist = torch.norm(mug_pos[:, :2] - tree_pos[:, :2], dim=-1)
        xy_aligned = xy_dist < self.hang_xy_tolerance

        from yamlab.robot.yam import ROBOT
        min_z = ROBOT.table_position[2] + self.mug_height / 2.0 + 0.05
        elevated = mug_pos[:, 2] > min_z

        _, right_grasping = self.robot.is_grasping(
            normal_force_thresh=0.1, env_ids=None,
        )

        return xy_aligned & elevated & (~right_grasping)

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get poses of all relevant objects for MimicGen data generation.

        Object poses (mug, mug_tree) are returned in the world frame, so they include
        each environment's origin offset. This is what makes object-relative subtask
        transforms re-anchor correctly per environment in multi-env generation.

        Includes a "fixed_ref" pseudo-object positioned at each environment's origin
        (identity rotation). Subtasks with object_ref="fixed_ref" are object-
        independent ("absolute"): their source trajectory is re-anchored to the
        environment origin rather than a moving object. Anchoring at the env origin
        (instead of the world origin) is essential for num_envs > 1 -- otherwise the
        source demo's world-frame poses (collected at env 0) would be replayed
        verbatim in every environment, sending arms toward env 0's coordinates. For a
        single environment the origin is (0, 0, 0), so this reduces to the identity
        pose and matches single-env behavior exactly.

        Args:
            env_ids (Sequence[int] | None): Environment indices. None returns all envs.

        Returns:
            dict[str, torch.Tensor]: Maps object name (incl. ``"fixed_ref"``) -> pose,
                float tensor of shape (N, 4, 4).
        """
        if env_ids is None:
            env_ids = slice(None)

        object_poses = {}

        for obj_name in ["mug", "mug_tree"]:
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

        # "fixed_ref" anchors absolute (object-independent) subtasks at the per-env
        # origin: translation = env origin, identity rotation. With src_fixed_ref also
        # at the (zero) collection origin, the transform delta is a pure translation by
        # this env's origin, re-anchoring world-frame source trajectories per env.
        origins = self.scene.env_origins if isinstance(env_ids, slice) else self.scene.env_origins[env_ids]
        n = origins.shape[0]
        identity_rot = torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        object_poses["fixed_ref"] = PoseUtils.make_pose(origins, identity_rot)

        return object_poses


gym.register(
    id="HangMugOnTree-Mimic-v0",
    entry_point=make_hang_mug_on_tree_mimic_env,
    disable_env_checker=True,
)

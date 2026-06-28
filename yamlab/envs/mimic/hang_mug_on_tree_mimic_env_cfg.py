"""MimicGen configuration for the HangMugOnTree task environment.

Bimanual data generation config built for source demos where the right arm holds
its source-initial pose while the left arm transports the mug, then moves toward
the mug only after mug_at_handover fires.

Task decomposition (11 signals):
  left_grasping -> mug_transported -> transport_near_end -> mug_at_handover
  -> both_grasping -> left_released -> handover_done -> left_backward_done
  -> post_handover_settled -> right_motion_started -> mug_hung

  transport_near_end and right_motion_window_done are emitted but drive no subtask boundary.

Left arm (8 subtasks):  approach -> grasp+lift -> transport -> hold+release
                        -> retract -> camera_approach -> camera_hold -> back_to_zero
Right arm (6 subtasks): ready_pose -> approach+grasp+hold -> post_handover_hold
                        -> tree_hang -> back_to_zero

Constraints:
  COORDINATION(L3 <-> R1, TRANSFORM, sync_start): mug-frame handover synchronized.
  SEQUENTIAL(R3 -> L7): left back_to_zero waits until right finishes tree hang.
  R0_pre runs in parallel with L0/L1/L2 (no SEQUENTIAL gate).
"""

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
    SubTaskConstraintCoordinationScheme,
    DataGenConfig,
)
from isaaclab.utils import configclass

from yamlab.envs.tasks.hang_mug_on_tree_manager_cfg import HangMugOnTreeManagerEnvCfg
from yamlab.configs import get_task_config

_SIG = get_task_config("HangMugOnTree-Mimic-v0")["mimic_signals"]
RIGHT_MOTION_INTERP_LOOKAHEAD = _SIG["right_motion_interp_lookahead"]
POST_HANG_ARM_AWAY_DELAY = _SIG["post_hang_arm_away_delay"]


@configclass
class HangMugOnTreeMimicEnvCfg(HangMugOnTreeManagerEnvCfg, MimicEnvCfg):
    """MimicGen config for bimanual HangMugOnTree."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config = DataGenConfig()
        self.datagen_config.name = "hang_mug_on_tree_bimanual_mimic"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 1

        left_subtasks = []

        # L0: Interpolate from default pose to pre-grasp; mug-relative for
        # source selection.  Large start offset skips the approach in source.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="left_grasping",
                first_subtask_start_offset_range=(40, 50),
                subtask_term_offset_range=(-90, -90),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 1,
                    "pos_weight": 1.0,
                    "rot_weight": 3.0,
                },
                action_noise=0.0,
                num_interpolation_steps=40,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm interpolates from default to pre-grasp pose",
            )
        )

        # L1: Faithful mug-relative replay from pre-contact through grasp and
        # lift; ends shortly after mug_transported (mug lifted >5 cm).
        left_subtasks.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="mug_transported",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 0.5,
                    "rot_weight": 5.0,
                },
                action_noise=0.003,
                num_interpolation_steps=2,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm grasps handle, lifts mug",
            )
        )

        # L2: Absolute replay (fixed_ref) from lift to handover region, ending at
        # mug_at_handover.  Fixed-ref delivers the mug to the same absolute handover
        # XY regardless of mug start pose, keeping the COORD TRANSFORM delta near zero
        # at the L3 boundary.  num_interpolation_steps=25 bridges the L1 mug-frame exit.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="fixed_ref",
                subtask_term_signal="mug_at_handover",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=25,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm transports mug to handover region (absolute replay)",
            )
        )

        # L3: Mug-relative hold and release, COORDINATED with R1 via TRANSFORM.
        # Source window [mug_at_handover, left_released].  2-step interp is intentional:
        # L2 absolute end pose == L3 first mug-frame waypoint (zero TRANSFORM delta at
        # COORD start), so longer interp would cause a visible no-op dwell.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="left_released",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 0.5,
                    "rot_weight": 5.0,
                },
                action_noise=0.0,
                num_interpolation_steps=2,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm holds mug at handover and releases (mug-frame, coord with R1)",
            )
        )

        # L4: Mug-relative backward retreat for ~85-95 source steps past handover_done
        # (~2.8-3.2 s at 30 Hz).  Range must cover the full source retreat; any steps
        # that leak into L5's tree-relative window get shifted by tree randomization
        # and appear artificially shortened.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="handover_done",
                subtask_term_offset_range=(85, 95),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 0.5,
                    "rot_weight": 5.0,
                },
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm retracts briefly past handover_done (mug-relative)",
            )
        )

        # L5: Tree-relative move to left wrist camera observation pose, ending at
        # post_handover_settled (pure time-delay after handover_done).
        # num_interpolation_steps=30 absorbs the mug->mug_tree frame change at a
        # natural pace (20 was too fast, 40 caused a mid-motion dwell).
        left_subtasks.append(
            SubTaskConfig(
                object_ref="mug_tree",
                subtask_term_signal="post_handover_settled",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 3.0,
                    "rot_weight": 1.0,
                },
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm retracts further and moves to camera pose near tree",
            )
        )

        # L6: Tree-relative static hold at camera pose through the hang,
        # source window [post_handover_settled, mug_hung].  SEQUENTIAL(R3->L7)
        # keeps the left arm here until R3 finishes, preventing early home return.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="mug_tree",
                subtask_term_signal="mug_hung",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 3.0,
                    "rot_weight": 1.0,
                },
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm holds at camera pose while right arm hangs",
            )
        )

        # L7: Absolute return to home (fixed_ref); source window [mug_hung, episode_end].
        # subtask_term_signal="mug_hung" is a placeholder — last-subtask end is episode_end.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="fixed_ref",
                subtask_term_signal="mug_hung",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left arm returns to zero / home pose (absolute)",
            )
        )

        self.subtask_configs["left_arm"] = left_subtasks

        right_subtasks = []

        # R0_pre: Absolute hold at source-initial pose through entire left transport,
        # source window [0, mug_at_handover].  Runs in parallel with L0/L1/L2.
        # num_interpolation_steps=45 (~1.5 s) blends from env-default to source pose.
        # object_ref="fixed_ref" (env-origin anchor) rather than None so the absolute
        # source pose is re-anchored to each environment's origin; with object_ref=None
        # the raw world-frame source pose would be replayed in every env and break
        # num_envs > 1 (the right arm would track env 0's coordinates).
        right_subtasks.append(
            SubTaskConfig(
                object_ref="fixed_ref",
                subtask_term_signal="mug_at_handover",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=45,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm interpolates to source-initial pose at episode start, holds through left arm transport (absolute, env-origin anchored)",
            )
        )

        # R1: Mug-relative approach, grasp, hold through left release, COORDINATED
        # with L3 via TRANSFORM.  Source window [mug_at_handover, handover_done].
        # 2-step interp is intentional: R0_pre absolute end == R1 first mug-frame
        # waypoint (same zero-delta argument as L2->L3).
        right_subtasks.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="handover_done",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 0.5,
                    "rot_weight": 5.0,
                },
                action_noise=0.0,
                num_interpolation_steps=2,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm plunges fingers into mug body, grasps rim, holds through left release (mug-frame, coord with L3)",
            )
        )

        # R2a: Mug-relative post-handover hold ending exactly at right_motion_started
        # (LOOKAHEAD=0).  LOOKAHEAD>0 would replay part of the forward approach in mug
        # frame, causing a visible "move forward then re-anchor" artifact in R3.
        right_subtasks.append(
            SubTaskConfig(
                object_ref="mug",
                subtask_term_signal="right_motion_started",
                subtask_term_offset_range=(
                    RIGHT_MOTION_INTERP_LOOKAHEAD,
                    RIGHT_MOTION_INTERP_LOOKAHEAD,
                ),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 0.5,
                    "rot_weight": 5.0,
                },
                action_noise=0.0,
                num_interpolation_steps=15,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm holds mug at handover through post-handover phase (mug-frame); ends RIGHT_MOTION_INTERP_LOOKAHEAD steps past the right_motion_started latch",
            )
        )

        # R3: Tree-relative approach, hang, and release.  Source window
        # [right_motion_started, mug_hung + POST_HANG_ARM_AWAY_DELAY]; extends 0.5 s past
        # mug_hung so the post-release forward push is covered in tree frame.  R4's
        # source then starts with the arm already clear of the tree, so R4's interp
        # jumps to a safe world-frame waypoint without replaying the forward push.
        right_subtasks.append(
            SubTaskConfig(
                object_ref="mug_tree",
                subtask_term_signal="mug_hung",
                subtask_term_offset_range=(POST_HANG_ARM_AWAY_DELAY, POST_HANG_ARM_AWAY_DELAY),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 3.0,
                    "rot_weight": 3.0,
                },
                action_noise=0.003,
                num_interpolation_steps=15,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm interpolates to tree-frame pre-hang pose, then approaches tree, hangs mug, releases (tree-relative)",
            )
        )

        # R4: Absolute return to home (fixed_ref); source window [mug_hung, episode_end].
        # subtask_term_signal="mug_hung" is a placeholder — last-subtask end is episode_end.
        right_subtasks.append(
            SubTaskConfig(
                object_ref="fixed_ref",
                subtask_term_signal="mug_hung",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm returns to zero / home pose (absolute)",
            )
        )

        self.subtask_configs["right_arm"] = right_subtasks

        self.task_constraint_configs = [
            # L3 and R1 share a mug-frame TRANSFORM delta; sync_start ensures both
            # arms enter the coordination region together regardless of relative timing.
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 3), ("right_arm", 1)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=True,
            ),
            # Right arm R3 plays at natural speed in parallel with the left arm;
            # SEQUENTIAL(R3->L7) holds L7 at step 0 (= observation pose, end of L6)
            # until R3 finishes so the left arm does not return home early.
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("right_arm", 3), ("left_arm", 7)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
        ]

        print(f"[INFO] HangMugOnTreeMimicEnvCfg initialized:")
        print(f"  - Signals: left_grasping -> mug_transported -> transport_near_end "
              f"-> mug_at_handover -> both_grasping -> left_released -> handover_done "
              f"-> left_backward_done -> post_handover_settled -> right_motion_started -> mug_hung")
        print(f"  - Left arm:  {len(left_subtasks)} subtasks "
              f"(approach, grasp+lift, transport, hold+release, "
              f"retract, camera_approach, camera_hold, back_to_zero)")
        print(f"  - Right arm: {len(right_subtasks)} subtasks "
              f"(ready_pose, approach+grasp+hold, post_handover_hold, "
              f"tree_hang, back_to_zero)")
        print(f"  - Constraints: "
              f"COORD(left[3]<->right[1], TRANSFORM, sync_start), "
              f"SEQ(right[3]->left[7])")

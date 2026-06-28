"""MimicGen configuration for the PutPotOnCooktop bimanual task environment.

Signal sequence: left_grasping -> both_grasping -> pot_ready -> above_obj1

Subtask structure:
  Left arm  (3): approach -> grasp+pick+rotate -> transport+place
  Right arm (3): approach -> grasp+pick+rotate -> transport+place

  approach:          Pure interpolation (~60 steps) from zero pose to the
                     pre-grasp pose. Aggressive start-skip + negative offset
                     leaves only 1-2 source waypoints at the pre-grasp position.

  grasp+pick+rotate: Source-demo replay from ~2 s before contact through
                     grasp -> lift -> rotate, transformed relative to pot.
                     Positive offset extends ~1-1.3 s past pot_ready, eating
                     into the early transport phase.

  transport+place:   Short interpolation bridges from the lifted pose; only
                     the final ~1.5 s near the cooker is source-replayed,
                     transformed relative to cooktop.

Constraints:
  SEQUENTIAL:   left_arm[0] -> right_arm[0]
  COORDINATION: left_arm[1] <-> right_arm[1]
  COORDINATION: left_arm[2] <-> right_arm[2]
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

from yamlab.envs.tasks.put_pot_on_cooktop_manager_cfg import PutPotOnCooktopManagerEnvCfg


@configclass
class PutPotOnCooktopMimicEnvCfg(PutPotOnCooktopManagerEnvCfg, MimicEnvCfg):
    """MimicGen config for bimanual PutPotOnCooktop."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config = DataGenConfig()
        self.datagen_config.name = "put_pot_on_cooktop_bimanual_mimic"
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

        # Left 0: Approach - 40-step interpolation to a 1-2-waypoint pre-grasp window.
        # Skip (55-65) + offset (-75,-70) collapses source span to ~1 step at T_lg - 70.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="pot",
                subtask_term_signal="left_grasping",
                first_subtask_start_offset_range=(55, 65),
                subtask_term_offset_range=(-75, -70),
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
                description="Left arm interpolates from zero to pre-grasp pose",
            )
        )

        # Left 1: Grasp + pick + rotate in pot frame.
        # Offset (10,15) extends ~0.4 s past pot_ready, eating into early transport.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="pot",
                subtask_term_signal="pot_ready",
                subtask_term_offset_range=(10, 15),
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
                description="Left arm grasps, picks up, and rotates pot",
            )
        )

        # Left 2: Transport + place in cooktop frame.
        # Starts late in transport (subtask 1 consumed the early phase);
        # source-replays only the final placement approach.
        left_subtasks.append(
            SubTaskConfig(
                object_ref="cooktop",
                subtask_term_signal="above_obj1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 5.0,
                    "rot_weight": 1.0,
                },
                action_noise=0.003,
                num_interpolation_steps=2,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Both arms transport pot and place on cooktop",
            )
        )

        self.subtask_configs["left_arm"] = left_subtasks

        right_subtasks = []

        # Right 0: Approach - SEQUENTIAL pin holds at step 0 until left[0] finishes.
        # Skip (110-130) jumps past the idle phase; offset (-75,-70) trims to pre-grasp.
        # 50-step interpolation plays once the SEQUENTIAL constraint releases.
        right_subtasks.append(
            SubTaskConfig(
                object_ref="pot",
                subtask_term_signal="both_grasping",
                first_subtask_start_offset_range=(110, 130),
                subtask_term_offset_range=(-75, -70),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 1.0,
                    "rot_weight": 3.0,
                },
                action_noise=0.0,
                num_interpolation_steps=50,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right arm interpolates from zero to pre-grasp pose",
            )
        )

        # Right 1: Grasp + pick + rotate (COORDINATION with left 1).
        right_subtasks.append(
            SubTaskConfig(
                object_ref="pot",
                subtask_term_signal="pot_ready",
                subtask_term_offset_range=(10, 15),
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
                description="Right arm grasps, picks up, and rotates pot",
            )
        )

        # Right 2: Transport + place in cooktop frame (COORDINATION with left 2).
        right_subtasks.append(
            SubTaskConfig(
                object_ref="cooktop",
                subtask_term_signal="above_obj1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": 2,
                    "pos_weight": 5.0,
                    "rot_weight": 1.0,
                },
                action_noise=0.003,
                num_interpolation_steps=2,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Both arms transport pot and place on cooktop",
            )
        )

        self.subtask_configs["right_arm"] = right_subtasks

        self.task_constraint_configs = [
            # Right arm approach waits for left arm approach to complete.
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 0), ("right_arm", 0)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=-1,
            ),
            # Grasp + pick + rotate: both arms synchronized.
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 1), ("right_arm", 1)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=False,
            ),
            # Transport + place: both arms synchronized.
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left_arm", 2), ("right_arm", 2)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=False,
            ),
        ]

        print(f"[INFO] PutPotOnCooktopMimicEnvCfg initialized:")
        print(f"  - Signals: left_grasping -> both_grasping -> pot_ready -> above_obj1")
        print(f"  - Left arm:  {len(left_subtasks)} subtasks (approach, grasp+pick+rot, transport+place)")
        print(f"  - Right arm: {len(right_subtasks)} subtasks (approach, grasp+pick+rot, transport+place)")
        print(f"  - Constraints: SEQ(left[0]->right[0]), "
              f"COORD(left[1]<->right[1]), COORD(left[2]<->right[2])")
        print(f"  - Approach: pure interpolation (60 steps) to pre-grasp pose")
        print(f"  - Transport: 30-step interpolation + final placement only")

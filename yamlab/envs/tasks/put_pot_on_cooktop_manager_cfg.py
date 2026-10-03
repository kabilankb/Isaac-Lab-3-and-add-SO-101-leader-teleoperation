"""Configuration for PutPotOnCooktop task using ManagerBasedRLEnv."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from yamlab.envs.yam_bimanual_scene import YamBimanualSceneCfg
from yamlab.robot.yam import ROBOT

from .yam_bimanual_env_cfg import YamBimanualEnvCfg, BaseEventsCfg
from yamlab.utils.task_logic import check_obj_below_table, compute_sparse_success_reward

# Object randomization is read from configs/tasks/put_pot_on_cooktop.yaml via the loader.
from yamlab.configs import pose_range_for

TASK_NAME = "PutPotOnCooktop-v0"


@configclass
class PutPotOnCooktopSceneCfg(YamBimanualSceneCfg):
    """Bimanual workstation scene with two rigid objects for the put-on-top task."""

    # pot: object to be picked and placed on top of cooktop.
    pot = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # set by configure_assets_usd_paths
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=1.3,  # 1300g
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Z is updated by configure_assets_usd_paths from asset_size.json.
            # 90-degree clockwise rotation around z-axis: (cos(-pi/4), 0, 0, sin(-pi/4))
            pos=(ROBOT.table_position[0] + 0.06, ROBOT.table_position[1] + 0.15, ROBOT.table_position[2] + 1.0),
            rot=(0.0, 0.0, -0.7071067811865476, 0.7071067811865476),  # xyzw, -90 deg yaw
        ),
    )

    # cooktop: base object that pot is placed on top of.
    cooktop = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cooktop",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # set by configure_assets_usd_paths
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=5.0,  # 5kg
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Z is updated by configure_assets_usd_paths from asset_size.json.
            # 90-degree clockwise rotation around z-axis: (cos(-pi/4), 0, 0, sin(-pi/4))
            pos=(ROBOT.table_position[0] + 0.055, ROBOT.table_position[1] - 0.3, ROBOT.table_position[2] + 1.0),
            rot=(0.0, 0.0, -0.7071067811865476, 0.7071067811865476),  # xyzw, -90 deg yaw
        ),
    )


@configclass
class PutPotOnCooktopEventsCfg(BaseEventsCfg):
    """Reset events for the PutPotOnCooktop task.

    Pose ranges come from configs/tasks/put_pot_on_cooktop.yaml via ``pose_range_for``;
    edit that file's objects_randomization entry to retune the region.
    """

    reset_pot_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": pose_range_for(TASK_NAME, "pot"),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("pot", body_names=".*"),
        },
    )

    reset_cooktop_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": pose_range_for(TASK_NAME, "cooktop"),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cooktop", body_names=".*"),
        },
    )


@configclass
class PutPotOnCooktopRewardsCfg:
    """Reward terms for PutPotOnCooktop task."""

    sparse_success_reward = RewardTermCfg(
        func=compute_sparse_success_reward,
        params={},
        weight=1.0,
    )


def check_task_success(env):
    """Return per-environment task success (both stage flags latched).

    Args:
        env: The environment instance.

    Returns:
        torch.Tensor: Boolean tensor of shape (num_envs,).
    """
    return torch.logical_and(env.stage1_success, env.stage2_success)


@configclass
class PutPotOnCooktopTerminationsCfg:
    """Termination conditions for the PutPotOnCooktop task."""

    # Truncation on episode timeout; length configured via episode_length_s.
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    # Terminal success when both stages are latched.
    task_success = TerminationTermCfg(
        func=check_task_success,
        params={},
    )

    # Terminate if either object falls off the table (1 cm margin).
    pot_below_table = TerminationTermCfg(
        func=check_obj_below_table,
        params={
            "asset_cfg": SceneEntityCfg("pot"),
            "table_z": ROBOT.table_position[2],
            "margin": 0.01,
        },
    )

    cooktop_below_table = TerminationTermCfg(
        func=check_obj_below_table,
        params={
            "asset_cfg": SceneEntityCfg("cooktop"),
            "table_z": ROBOT.table_position[2],
            "margin": 0.01,
        },
    )


@configclass
class PutPotOnCooktopManagerEnvCfg(YamBimanualEnvCfg):
    """Full configuration for the PutPotOnCooktop ManagerBasedRLEnv.

    num_envs defaults to 1 and can be overridden by passing ``num_envs`` to ``make_task_env``.
    """

    scene: PutPotOnCooktopSceneCfg = PutPotOnCooktopSceneCfg(num_envs=1, env_spacing=3.0)

    events: PutPotOnCooktopEventsCfg = PutPotOnCooktopEventsCfg()
    rewards: PutPotOnCooktopRewardsCfg = PutPotOnCooktopRewardsCfg()
    terminations: PutPotOnCooktopTerminationsCfg = PutPotOnCooktopTerminationsCfg()

    obj_name_to_event_name: dict[str, str] = {
        "pot": "reset_pot_position",
        "cooktop": "reset_cooktop_position"
    }


# Fallback grasp targets, used only if the task YAML has no grasp.detect block;
# normally the per-arm grasp.detect in configs/tasks/put_pot_on_cooktop.yaml drives this.
# Attached outside @configclass so it is not treated as a scene asset field.
PutPotOnCooktopManagerEnvCfg.CONTACT_OBJECT_NAMES = ["pot"]

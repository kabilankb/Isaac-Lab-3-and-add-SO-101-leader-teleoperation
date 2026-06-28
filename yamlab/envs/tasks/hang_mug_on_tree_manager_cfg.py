"""Configuration for HangMugOnTree task using ManagerBasedRLEnv."""


import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from yamlab.envs.yam_bimanual_scene import YamBimanualSceneCfg
from yamlab.robot.yam import YAM_CONFIG_DEFAULT, ROBOT

from .yam_bimanual_env_cfg import YamBimanualEnvCfg, BaseEventsCfg
from yamlab.utils.physics_events import set_rigid_body_friction
from yamlab.utils.task_logic import compute_sparse_hang_reward, check_obj_below_table

# Per-task config (object randomization + friction) is read from
# configs/tasks/hang_mug_on_tree.yaml via the layered loader. ``pose_range_for``
# yields the canonical XY/yaw ranges (overwritten at runtime with the effective
# values by configure_objects_randomization).
from yamlab.configs import pose_range_for, get_task_config

TASK_NAME = "HangMugOnTree-v0"
_CFG = get_task_config(TASK_NAME)

# LEFT_FINGER_FRICTION_HIGH (100): stable grasp on the thin mug handle where PhysX
# generates only 1-2 contact points (combine_mode=average halves it to ~50 at
# finger<->mug, still enough to resist rotation/slip). LOW (0.5): scene default,
# used at reset and after handover.
LEFT_FINGER_FRICTION_HIGH = _CFG["friction"]["left_finger_high"]
LEFT_FINGER_FRICTION_LOW = _CFG["friction"]["left_finger_low"]


@configclass
class HangMugOnTreeSceneCfg(YamBimanualSceneCfg):
    """Bimanual workstation scene with a mug (mug) and mug tree (mug_tree).

    Both arms load the stock ``yam.usd`` (PhysicsMaterial 0.5/0.5, combine=average).
    Left-arm fingertip friction is raised at runtime when the left arm grasps the mug
    and dropped back at handover via the friction events in HangMugOnTreeEventsCfg.
    """

    def __post_init__(self):
        super().__post_init__()

        left_arm_cfg = YAM_CONFIG_DEFAULT.copy()
        left_arm_cfg.init_state.pos = ROBOT.arm_position('left')
        left_arm_cfg.init_state.rot = ROBOT.arm_quaternion('left')
        self.left_arm = left_arm_cfg.replace(prim_path="{ENV_REGEX_NS}/LeftArm")

        right_arm_cfg = YAM_CONFIG_DEFAULT.copy()
        right_arm_cfg.init_state.pos = ROBOT.arm_position('right')
        right_arm_cfg.init_state.rot = ROBOT.arm_quaternion('right')
        self.right_arm = right_arm_cfg.replace(prim_path="{ENV_REGEX_NS}/RightArm")

    # Mug to be hung on the tree; USD path set by configure_assets_usd_paths at runtime.
    mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.20,  # 200g
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(ROBOT.table_position[0] + 0.075, ROBOT.table_position[1] + 0.15, ROBOT.table_position[2] + 1.0),
            # 135 deg yaw default; quat = (cos(67.5 deg), 0, 0, sin(67.5 deg)).
            rot=(0.3826834323650898, 0.0, 0.0, 0.9238795325112867),
        ),
    )

    # Mug tree target; heavy (5 kg) so it stays stable when the mug is hung.
    mug_tree = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug_tree",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=5.0,  # 5kg - heavy so it stays stable
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(ROBOT.table_position[0] + 0.105, ROBOT.table_position[1] - 0.3, ROBOT.table_position[2] + 1.0),
            # 15 deg yaw default; quat = (cos(7.5 deg), 0, 0, sin(7.5 deg)).
            rot=(0.9914448613738104, 0.0, 0.0, 0.13052619222005157),
        ),
    )


@configclass
class HangMugOnTreeEventsCfg(BaseEventsCfg):
    """Event terms for HangMugOnTree: object resets and runtime left-finger friction modulation."""

    reset_mug_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": pose_range_for(TASK_NAME, "mug"),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mug", body_names=".*"),
        },
    )

    reset_mug_tree_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": pose_range_for(TASK_NAME, "mug_tree"),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mug_tree", body_names=".*"),
        },
    )

    # Reset left-finger friction to LOW at every episode boundary. Required because
    # PhysX retains per-shape runtime material values across episodes.
    reset_left_finger_friction = EventTermCfg(
        func=set_rigid_body_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("left_arm"),
            "body_names": ["left_finger", "right_finger"],
            "static_friction": LEFT_FINGER_FRICTION_LOW,
            "verbose": False,
        },
    )

    # Raise left-finger friction to HIGH on first detected left-arm grasp; latched per episode.
    on_left_grasp_raise_left_friction = EventTermCfg(
        func=set_rigid_body_friction,
        mode="left_grasp",
        params={
            "asset_cfg": SceneEntityCfg("left_arm"),
            "body_names": ["left_finger", "right_finger"],
            "static_friction": LEFT_FINGER_FRICTION_HIGH,
            "verbose": False,
        },
    )

    # Drop left-finger friction back to LOW when both arms grasp simultaneously during handover.
    on_handover_release_left_friction = EventTermCfg(
        func=set_rigid_body_friction,
        mode="handover",
        params={
            "asset_cfg": SceneEntityCfg("left_arm"),
            "body_names": ["left_finger", "right_finger"],
            "static_friction": LEFT_FINGER_FRICTION_LOW,
            "verbose": False,
        },
    )


@configclass
class HangMugOnTreeRewardsCfg:
    """Sparse reward terms for HangMugOnTree task.

    Per-episode totals: 0.0 (no handover), 0.5 (handover only), 1.0 (hang achieved).
    ``compute_sparse_hang_reward`` credits +0.5 at the first step each stage latches,
    using per-env one-shot flags to avoid re-crediting on subsequent latched steps.
    """

    sparse_success_reward = RewardTermCfg(
        func=compute_sparse_hang_reward,
        params={},
        weight=1.0,
    )


def check_task_success(env):
    """Check if all three stages (pick, handover, hang) are successful.

    Args:
        env: HangMugOnTreeManager instance.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    return env.stage1_success & env.stage2_success & env.stage3_success


@configclass
class HangMugOnTreeTerminationsCfg:
    """Termination terms for HangMugOnTree task."""

    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    task_success = TerminationTermCfg(
        func=check_task_success,
        params={},
    )

    mug_below_table = TerminationTermCfg(
        func=check_obj_below_table,
        params={
            "asset_cfg": SceneEntityCfg("mug"),
            "table_z": ROBOT.table_position[2],
            "margin": 0.01,
        },
    )

    mug_tree_below_table = TerminationTermCfg(
        func=check_obj_below_table,
        params={
            "asset_cfg": SceneEntityCfg("mug_tree"),
            "table_z": ROBOT.table_position[2],
            "margin": 0.01,
        },
    )


@configclass
class HangMugOnTreeManagerEnvCfg(YamBimanualEnvCfg):
    """Full environment configuration for the HangMugOnTree task."""

    scene: HangMugOnTreeSceneCfg = HangMugOnTreeSceneCfg(num_envs=1, env_spacing=3.0)

    events: HangMugOnTreeEventsCfg = HangMugOnTreeEventsCfg()
    rewards: HangMugOnTreeRewardsCfg = HangMugOnTreeRewardsCfg()
    terminations: HangMugOnTreeTerminationsCfg = HangMugOnTreeTerminationsCfg()

    obj_name_to_event_name: dict[str, str] = {
        "mug": "reset_mug_position",
        "mug_tree": "reset_mug_tree_position"
    }


# Fallback grasp targets, used only if the task YAML has no grasp.detect block;
# normally the per-arm grasp.detect in configs/tasks/hang_mug_on_tree.yaml drives this.
# Attached outside @configclass so it is not treated as a scene asset field.
HangMugOnTreeManagerEnvCfg.CONTACT_OBJECT_NAMES = ["mug"]

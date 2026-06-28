"""YAM bimanual robot: hardware spec, articulation configs, and runtime robot class.

The robot's hardware facts — camera intrinsics/poses, arm base poses, table geometry,
gripper finger positions, finger-pad keypoints, and controller gain sets — live in
``configs/robot/yam.yaml`` and are read through :class:`yamlab.robot.spec.RobotSpec`. This
module exposes that spec as :obj:`ROBOT`, builds the single-arm ArticulationCfgs from it, and
provides :class:`YamRobot` (the generic :class:`~yamlab.robot.bimanual_robot.BimanualRobot`
bound to the YAM spec).

* :obj:`YAM_CONFIG`: YAM arm + gripper with the ``base`` actuator gains.
* :obj:`YAM_CONFIG_HIGH_PD_CFG`: same arm with the ``high_pd`` position-control gains (sim).
* :obj:`YAM_CONFIG_DEFAULT`: whichever of the two the robot's ``controller.default`` selects.

Read calibration through ``ROBOT`` (e.g. ``ROBOT.table_position``,
``ROBOT.arm_position("left")``, ``ROBOT.fingers``). The YAM USD assets live in subfolders of
this package (``arm/`` and ``workstation/``).
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from ..spec import get_robot
from ..bimanual_robot import BimanualRobot

# USD asset directories. The arm/station USDs live in subfolders of this package, kept
# separate from its Python.
_YAM_PKG_DIR = Path(__file__).resolve().parent
YAM_DIR = os.path.join(_YAM_PKG_DIR, "arm")                       # arm USD (yam.usd)
YAM_STATION_ONLY_DIR = os.path.join(_YAM_PKG_DIR, "workstation")  # workstation-only USD (table + walls)

# Structured robot spec — the YAM hardware facts (configs/robot/yam.yaml). Import this and
# read its accessors rather than threading calibration values around the codebase.
ROBOT = get_robot("yam")

# The gain set the scene builds arms with (a robot property: configs/robot/yam.yaml
# controller.default). Consumed by yam_bimanual_scene / task scene cfgs via YAM_CONFIG_DEFAULT.
DEFAULT_CONTROLLER = ROBOT.default_controller

# Controller gain sets (stiffness/damping per actuator group), from the robot spec.
_BASE_GAINS = ROBOT.controller_gains("base")
_HIGH_PD_GAINS = ROBOT.controller_gains("high_pd")

YAM_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{YAM_DIR}/yam.usd",
        activate_contact_sensors=True,  # Enable contact sensors for gripper contact detection
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1
        ),
    ),
    articulation_root_prim_path="/arm",
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_finger": -0.0475,
            "right_finger": -0.0475,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        # Stiffness (kp) / damping (kv) come from the robot spec's `base` gain set;
        # joint groupings + effort/velocity limits are fixed actuator structure.
        "yam_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"], # DM4340 motors for first 3 joints
            effort_limit_sim=28.0, # forcerange
            velocity_limit_sim=100.0,
            stiffness=_BASE_GAINS["yam_shoulder"]["stiffness"],
            damping=_BASE_GAINS["yam_shoulder"]["damping"],
        ),
        "yam_elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint4"], # DM4310 elbow joint
            effort_limit_sim=10.0, # forcerange
            velocity_limit_sim=100.0,
            stiffness=_BASE_GAINS["yam_elbow"]["stiffness"],
            damping=_BASE_GAINS["yam_elbow"]["damping"],
        ),
        "yam_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-6]"], # DM4310 wrist joints
            effort_limit_sim=10.0, # forcerange
            velocity_limit_sim=100.0,
            stiffness=_BASE_GAINS["yam_wrist"]["stiffness"],
            damping=_BASE_GAINS["yam_wrist"]["damping"],
        ),
        "yam_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            effort_limit_sim=100.0, # forcerange
            velocity_limit_sim=100.0,
            stiffness=_BASE_GAINS["yam_gripper"]["stiffness"],
            damping=_BASE_GAINS["yam_gripper"]["damping"],
        ),
    },
)
"""ArticulationCfg for a single YAM arm + gripper with base actuator gains."""

YAM_CONFIG_HIGH_PD_CFG = YAM_CONFIG.copy()
for group, gains in _HIGH_PD_GAINS.items():
    YAM_CONFIG_HIGH_PD_CFG.actuators[group].stiffness = gains["stiffness"]
    YAM_CONFIG_HIGH_PD_CFG.actuators[group].damping = gains["damping"]
YAM_CONFIG_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
"""ArticulationCfg for a single YAM arm with high-PD position control (used in sim)."""

# The arm config the scene builds with. The scene reads actuator gains when it builds the
# articulation, so the active gain set is selected here from the robot's controller.default
# (configs/robot/yam.yaml): high-PD for position tracking, or the softer base gains.
YAM_CONFIG_DEFAULT = YAM_CONFIG_HIGH_PD_CFG if DEFAULT_CONTROLLER == "high_pd" else YAM_CONFIG
"""ArticulationCfg the scene builds arms with (robot's ``controller.default`` gain set)."""


class YamRobot(BimanualRobot):
    """The YAM bimanual robot: a :class:`BimanualRobot` bound to the YAM hardware spec.

    Fixes the spec to :obj:`ROBOT`, so callers construct it with just the scene and grasp target.
    """

    def __init__(self, scene, grasp_target: str = "obj_0"):
        """Create the YAM robot model bound to a scene.

        Args:
            scene (InteractiveScene): The live scene (holds the arms + contact sensors).
            grasp_target (str): Scene-object name the arms grasp by default (e.g. ``"obj_0"``).
        """
        super().__init__(scene, ROBOT, grasp_target)

"""Scene configuration for the YAM bimanual workstation (table, lighting, cameras, two arms)."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from yamlab.robot.yam import YAM_STATION_ONLY_DIR
from yamlab.robot.yam import YAM_CONFIG_DEFAULT, ROBOT
from yamlab.utils.perception import intrinsics_to_matrix

from isaaclab.sensors.camera import TiledCameraCfg


@configclass
class YamBimanualSceneCfg(InteractiveSceneCfg):
    """IsaacLab scene configuration for the YAM bimanual workstation.

    Contains ground plane, dome light, workstation mesh, two YAM arms, and three tiled
    cameras (top + two wrist).

    Each camera is a :class:`TiledCamera` (one camera location, tiled across the num_envs
    instances into a single render product). At num_envs=1 this matches a plain Camera
    (3 render products); for parallel evaluation (num_envs > 1) all N env-instances of a
    camera render in ONE pass, so total render products stay 3 instead of 3*N. Access is
    identical to Camera: ``scene["top_camera"].data.output["rgb"]`` has shape
    ``(num_envs, H, W, 3)`` uint8.
    """

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(semantic_tags=[("class", "ground")]))

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.75, 0.75))
    )

    # Workstation geometry (table + walls) - immovable kinematic collider.
    # Uses AssetBaseCfg so it is not managed as a RigidObject at runtime.
    workstation = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workstation",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(YAM_STATION_ONLY_DIR, "workstation.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    def __post_init__(self):
        """Configure arms and cameras using calibrated poses from yam.py.

        Contact sensors are NOT added here by default; call
        YamBimanualEnvCfg.configure_contact_sensors(detect_map) after construction,
        where detect_map is the per-arm grasp-target map {arm: [object, ...]}.
        """
        left_arm_cfg = YAM_CONFIG_DEFAULT.copy()
        left_arm_cfg.init_state.pos = ROBOT.arm_position("left")
        left_arm_cfg.init_state.rot = ROBOT.arm_quaternion("left")
        self.left_arm = left_arm_cfg.replace(prim_path="{ENV_REGEX_NS}/LeftArm")

        right_arm_cfg = YAM_CONFIG_DEFAULT.copy()
        right_arm_cfg.init_state.pos = ROBOT.arm_position("right")
        right_arm_cfg.init_state.rot = ROBOT.arm_quaternion("right")
        self.right_arm = right_arm_cfg.replace(prim_path="{ENV_REGEX_NS}/RightArm")

        print(f"[INFO] Left arm: pos={ROBOT.arm_position('left')}, quat={ROBOT.arm_quaternion('left')}")
        print(f"[INFO] Right arm: pos={ROBOT.arm_position('right')}, quat={ROBOT.arm_quaternion('right')}")

        # Camera intrinsics were calibrated at this (width, height); the pinhole spawn cfg
        # uses it to convert intrinsics to focal length/aperture, decoupled from render res.
        calib_width, calib_height = ROBOT.intrinsic_resolution

        # Sensor render resolution. 640x480 is the default; overridden by the task YAML.
        width, height = 640, 480

        # One TiledCamera per camera in the robot config (ROBOT.camera_names). Each camera's
        # mount is its prim_path (a world prim, or under an arm link so it follows the arm);
        # adding a camera is a config entry, not code. Scene attribute = "<name>_camera".
        for name in ROBOT.camera_names:
            setattr(self, f"{name}_camera", TiledCameraCfg(
                prim_path=ROBOT.camera_prim_path(name),
                update_period=0.0,
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=intrinsics_to_matrix(ROBOT.camera_intrinsics(name)),
                    width=calib_width,
                    height=calib_height,
                    f_stop=0.0,
                    projection_type="pinhole",
                    lock_camera=False,
                ),
                offset=TiledCameraCfg.OffsetCfg(
                    pos=ROBOT.camera_position(name),
                    rot=ROBOT.camera_quaternion_opengl(name),
                    convention="opengl",
                ),
            ))


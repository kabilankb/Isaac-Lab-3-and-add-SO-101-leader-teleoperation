"""YAM bimanual task-environment configuration.

Defines the YAM action, observation, and event specifications and ``YamBimanualEnvCfg``,
which carries the simulation/physics defaults, the YAM scene, and the per-mode configuration
helpers (teleoperation, replay, data generation, policy evaluation) used by every YAM task.
The robot-agnostic helpers (object/domain randomization, pose schedule) come from
:class:`~yamlab.envs.manipulation_env_cfg.ManipulationEnvCfg`.
"""

from dataclasses import MISSING
from typing import Optional, List

import os
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg, ActionTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import DatasetExportMode

from yamlab.envs.manipulation_env_cfg import ManipulationEnvCfg
from yamlab.envs.yam_bimanual_scene import YamBimanualSceneCfg
from yamlab.domain_randomization import DomainRandomizationCfg
from yamlab.robot.yam import ROBOT
from yamlab.utils.assets import OBJ_Z_OFFSET, find_contact_body_link, get_asset_usd_path, load_asset_size
from yamlab.utils.perception import extract_background_mask, get_arm_joint_pos, get_gripper_continuous_state
from yamlab.utils.observation_modalities import validate_observation_modalities



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="left_arm",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=False,
    )

    # Controls left_finger only; right_finger is mirrored in YamBimanualEnv.step().
    left_gripper_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="left_arm",
        joint_names=["left_finger"],
        scale=1.0,
        use_default_offset=False,
    )

    right_arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="right_arm",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=False,
    )

    # Controls left_finger only; right_finger is mirrored in YamBimanualEnv.step().
    right_gripper_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="right_arm",
        joint_names=["left_finger"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: ObservationGroupCfg = ObservationGroupCfg()

    def __post_init__(self):
        """Populate the default policy observation group with proprioceptive terms."""
        self.policy.enable_corruption = False
        self.policy.concatenate_terms = False
        self.policy.left_arm_joint_pos = ObservationTermCfg(func=get_arm_joint_pos, params={"asset_cfg": SceneEntityCfg("left_arm")})
        self.policy.right_arm_joint_pos = ObservationTermCfg(func=get_arm_joint_pos, params={"asset_cfg": SceneEntityCfg("right_arm")})
        self.policy.left_gripper_state = ObservationTermCfg(func=get_gripper_continuous_state, params={"asset_cfg": SceneEntityCfg("left_arm")})
        self.policy.right_gripper_state = ObservationTermCfg(func=get_gripper_continuous_state, params={"asset_cfg": SceneEntityCfg("right_arm")})

    def add_recording_group(self):
        """Add a recording observation group for video capture when recording is enabled."""
        if not hasattr(self, 'recording'):
            self.recording = ObservationGroupCfg()
            self.recording.enable_corruption = False
            self.recording.concatenate_terms = False
            print(f"[INFO] Added recording observation group")


@configclass
class BaseEventsCfg:
    """Configuration for reset events."""

    # reset_joint_targets=True prevents the robot from resuming motion toward stale
    # joint targets after a reset.
    reset_all = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True}
    )

    reset_robot_arms_left = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("left_arm"),
        },
    )

    reset_robot_arms_right = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("right_arm"),
        },
    )


@configclass
class YamBimanualEnvCfg(ManipulationEnvCfg):
    """Configuration shared by all YAM bimanual task environments.

    Subclass this to define a specific task, providing a scene, rewards,
    and terminations. Mode-specific helpers (configure_for_teleoperation, etc.)
    adjust observations, terminations, and domain randomization in place.
    """

    decimation = 4  # Control frequency = 120 Hz / 4 = 30 Hz
    episode_length_s = 180.0  # Default 3-minute timeout for time_out termination

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1/120.0,
        render_interval=4,
        enable_scene_query_support=True,
        device="cpu",
        physx=sim_utils.PhysxCfg(
            min_position_iteration_count=16,
            min_velocity_iteration_count=1,
        ),
        render=sim_utils.RenderCfg(),
    )

    scene: YamBimanualSceneCfg = MISSING

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: BaseEventsCfg = BaseEventsCfg()
    rewards = MISSING
    terminations = MISSING

    # Maps object scene names to their reset event names, e.g. {"obj_0": "reset_obj_0_position"}.
    obj_name_to_event_name: dict[str, str] = {}

    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    recorders.export_in_record_pre_reset = False

    # Set by make_task_env when task_description is provided for multi-task training.
    _task_description: str | None = None

    def __post_init__(self) -> None:
        """Run parent post-init and set the default viewer camera pose."""
        super().__post_init__()
        self.viewer.eye = (-1.0, 0.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.85)

    pose_schedule_data: Optional[dict] = None

    def configure_for_teleoperation(self, enable_pose_schedule: bool = False,
                                    pose_schedule: Optional[dict] = None):
        """Configure environment for single-robot teleoperation.

        Forces num_envs=1, disables timeout and task_success terminations (the
        operator controls when to save/discard), and suppresses observation recording.

        Args:
            enable_pose_schedule (bool): If True, apply the per-task ``pose_schedule`` and
                disable randomization for scheduled objects. The pose advances after
                each successful save and stays on discard. Teleop only -- other modes
                always randomize.
            pose_schedule (None or dict): The ``{obj_name: [pose, ...]}`` mapping for this
                task, resolved from configs/tasks/<task>.yaml (modes.teleoperation.pose_schedule)
                and passed in by make_task_env. Required when enable_pose_schedule.
        """
        if hasattr(self.scene, 'num_envs'):
            self.scene.num_envs = 1
            print(f"[INFO] Teleoperation mode: forcing num_envs=1")

        if hasattr(self.terminations, "time_out"):
            self.terminations.time_out = None
        if hasattr(self.terminations, "task_success"):
            self.terminations.task_success = None

        self.configure_observation_modalities(observation_modalities=[])

        # Teleoperation records only simulation state, not rendered images, so the
        # camera sensors are unused -- yet each still owns an RTX render product that
        # renders and is read back every step. Replace the camera sensors with bare
        # USD camera prims (no render product): the operator's wrist/top camera
        # viewports bind to these prims for visual feedback, while physics stepping is
        # not burdened by sensor rendering/readback. 
        for cam_name in (f"{name}_camera" for name in ROBOT.camera_names):
            cam_cfg = getattr(self.scene, cam_name, None)
            if cam_cfg is not None:
                setattr(self.scene, cam_name, AssetBaseCfg(
                    prim_path=cam_cfg.prim_path,
                    spawn=cam_cfg.spawn,
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=cam_cfg.offset.pos, rot=cam_cfg.offset.rot
                    ),
                ))

        if enable_pose_schedule:
            self.configure_pose_schedule(pose_schedule)

    # Each scheduled object's effective randomization range, used at runtime to resolve
    # fractional schedule entries (pos_fraction/rot_fraction) into absolute offsets.
    # Format: {obj_name: {"pos_range": (eff_x, eff_y), "orientation_range": float}}
    pose_schedule_effective_ranges: Optional[dict] = None

    def configure_for_data_generation(
        self,
        enable_domain_randomization: bool = False,
        hdris_path: Optional[str] = None,
        materials_dir: Optional[str] = None,
        material_randomization: bool = True,
    ):
        """Configure the environment for MimicGen data generation.

        Disables task_success and time_out terminations so MimicGen controls episode
        length and can record post-task motions. Stores the success_term reference in
        datagen_config for MimicGen's internal success tracking. Sets the recorder to
        EXPORT_NONE mode so no HDF5 file is created; the LeRobot recorder handles saving.

        Args:
            enable_domain_randomization (bool): Whether to enable domain randomization.
            hdris_path (None or str): Path to HDRI folder for lighting randomization.
            materials_dir (None or str): Root directory containing MDL material files.
            material_randomization (bool): Whether to enable material/texture randomization.
        """
        if hasattr(self.terminations, "task_success") and self.terminations.task_success is not None:
            if not hasattr(self, 'datagen_config'):
                from isaaclab.envs.mimic_env_cfg import DataGenConfig
                self.datagen_config = DataGenConfig()
            self.datagen_config.success_term = self.terminations.task_success

        if hasattr(self.terminations, "task_success"):
            self.terminations.task_success = None
            print(f"[INFO] Data generation mode: disabled task_success termination (episodes continue after success)")

        if hasattr(self.terminations, "time_out"):
            self.terminations.time_out = None
            print(f"[INFO] Data generation mode: disabled timeout termination (MimicGen controls episode length)")

        if self.recorders is None:
            self.recorders = ActionStateRecorderManagerCfg()

        # EXPORT_NONE means no HDF5 dataset is written here (the LeRobot recorder saves the
        # generated data instead), which also avoids HDF5 file-lock contention when multiple
        # MimicGen workers run in parallel.
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        self.recorders.export_in_record_pre_reset = False

        print(f"[INFO] Recorder configured for MimicGen data generation (EXPORT_NONE mode)")
        print(f"[INFO] - Using LeRobot recorder for actual saving (no HDF5 file created)")

        if enable_domain_randomization:
            lighting_randomization = hdris_path is not None
            self.configure_domain_randomization(
                enabled=True,
                material_randomization=material_randomization,
                lighting_randomization=lighting_randomization,
                hdris_path=hdris_path,
                materials_dir=materials_dir,
            )
        else:
            self.configure_domain_randomization(enabled=False)

        print(f"[INFO] Environment configured for data generation")
        print(f"  - Domain randomization: {enable_domain_randomization}")
        if enable_domain_randomization:
            print(f"  - Material randomization: {material_randomization}")
            print(f"  - Lighting randomization: {lighting_randomization} (HDRI: {hdris_path is not None})")

    def configure_for_policy_evaluation(
        self,
        observation_modalities: list[str],
        timeout_steps: Optional[int] = None,
        enable_domain_randomization: bool = False,
        hdris_path: Optional[str] = None,
        materials_dir: Optional[str] = None,
        use_unseen_materials: bool = False,
        **kwargs,
    ):
        """Configure the environment for parallel policy evaluation rollouts.

        Args:
            observation_modalities (list[str]): Modalities to expose to the policy
                (e.g., ['rgb', 'proprioception']).
            timeout_steps (None or int): Episode length in environment steps; None uses
                episode_length_s.
            enable_domain_randomization (bool): Whether to randomize materials and lighting.
            hdris_path (None or str): Path to HDRI folder for lighting randomization.
            materials_dir (None or str): Root directory for MDL material files.
            use_unseen_materials (bool): If True, use holdout materials for
                out-of-distribution eval.
            **kwargs: Additional parameters including num_envs for parallel rollouts.
        """
        if 'num_envs' in kwargs:
            num_envs = kwargs.pop('num_envs')
            if hasattr(self.scene, 'num_envs'):
                self.scene.num_envs = num_envs
                print(f"[INFO] Policy evaluation mode: setting num_envs={num_envs} for parallel evaluation")

        if timeout_steps is not None:
            # max_episode_length = ceil(episode_length_s / step_dt), step_dt = decimation * dt
            timeout_seconds = timeout_steps * self.decimation * self.sim.dt
            self.episode_length_s = timeout_seconds
            print(f"[INFO] Configured timeout termination: {timeout_steps} environment steps = {timeout_seconds:.2f}s (decimation={self.decimation}, dt={self.sim.dt}, step_dt={self.decimation * self.sim.dt:.4f}s)")
        else:
            print(f"[INFO] Using default episode_length_s={self.episode_length_s}s for timeout termination")

        self.configure_observation_modalities(
            observation_modalities=observation_modalities,
            **kwargs
        )

        # Evaluation does not write HDF5 trajectories (eval records video separately).
        self.recorders = None

        material_randomization = kwargs.pop('material_randomization', True)
        if enable_domain_randomization:
            lighting_randomization = hdris_path is not None
            self.configure_domain_randomization(
                enabled=True,
                material_randomization=material_randomization,
                lighting_randomization=lighting_randomization,
                hdris_path=hdris_path,
                materials_dir=materials_dir,
                use_unseen_materials=use_unseen_materials,
            )
        else:
            self.configure_domain_randomization(enabled=False)

        print(f"[INFO] Environment configured for policy evaluation")
        print(f"  - Domain randomization: {enable_domain_randomization}")
        if enable_domain_randomization:
            print(f"  - Material randomization: {material_randomization}")
            print(f"  - Lighting randomization: {hdris_path is not None}")
            print(f"  - Use unseen materials: {use_unseen_materials}")

    def configure_contact_sensors(self, detect_map: dict):
        """Add finger contact sensors for grasp detection, per arm.

        Only the arms named in ``detect_map`` get their two finger sensors, each filtered to
        that arm's listed objects -- so a single-arm task pays for two sensors, not four. An
        arm with an empty list (or absent from the map) gets none. No sensors are created when
        ``detect_map`` is empty (grasp detection disabled).

        Args:
            detect_map (dict): Per-arm grasp targets, ``{arm_name: [object_name, ...]}``
                (arm_name is "left_arm"/"right_arm").
                E.g. ``{"left_arm": ["obj_0"], "right_arm": ["obj_1"]}``.
        """
        detect_map = {arm: list(objs) for arm, objs in (detect_map or {}).items()
                      if objs and arm in ROBOT.arm_names}
        if not detect_map:
            return

        # Per-arm finger contact-sensor layout (sensor attr, prim subpath), a robot property.
        topology = ROBOT.contact_sensor_topology()

        from isaaclab.sensors import ContactSensorCfg

        # Filtered contact reporting (force_matrix_w / contact_pos_w between a finger and the
        # target object) requires the PhysX contact-reporter API on the OBJECT as well, not just
        # the finger. Without it, the finger's net_forces_w is still reported, but the per-object
        # force matrix and contact points come back zero / NaN. Enable it on every object any arm
        # filters so the grasp detector's force + pad checks see the contact.
        all_objects = sorted({name for objs in detect_map.values() for name in objs})
        for name in all_objects:
            obj_cfg = getattr(self.scene, name, None)
            spawn_cfg = getattr(obj_cfg, "spawn", None) if obj_cfg is not None else None
            if spawn_cfg is not None and hasattr(spawn_cfg, "activate_contact_sensors"):
                spawn_cfg.activate_contact_sensors = True

        # The filter must target the object's rigid-body prim (per-object _contact_body_links),
        # not the asset-root Xform: the root carries no collider, so filtering it gives zero force.
        # track_contact_points=True populates ContactSensorData.contact_pos_w (the
        # world-frame contact location per filtered object), which the gripper's pad check
        # uses to reject fingertip-only touches and confirm the contact lies on the finger pad.
        # It is a GPU-synced tensor (unlike PhysX scene-query raycasts), so it works on
        # both the CPU and GPU pipelines. Requires filter_prim_paths_expr to be non-empty
        # and max_contact_data_count_per_prim >= 1 (both satisfied here).
        #
        # max_contact_data_count_per_prim is the per-(finger, filter, env) contact-point buffer.
        # A finger only overlaps a few of the object's collision hulls at once, so the realistic
        # count is small (~4 measured for a fingertip on a 40-hull pot) and independent of the
        # object's total hull count. 64 gives ample margin (a too-small buffer crashes on overflow);
        # the memory cost is negligible.
        #
        # update_period = one control step (sim dt x decimation), not 0.0 (every physics substep):
        # the grasp detector reads contact data once per env.step, so updating every substep ran
        # get_contact_data ~decimation times more than needed (it was ~28% of env.step). track_pose
        # is off because the pad check uses the arm's body_link_pose_w, not the sensor's own pose.
        contact_update_period = self.sim.dt * self.decimation
        # Each arm's two finger sensors filter only that arm's objects, so a single-arm task
        # creates two sensors instead of four (and an arm filters just the objects it grasps).
        body_links = self._contact_body_links  # populated by configure_assets_instance_paths (--asset)
        for arm, objs in detect_map.items():
            contact_filter = []
            for name in objs:
                sub = body_links[name]
                contact_filter.append(f"{{ENV_REGEX_NS}}/{name}" + (f"/{sub}" if sub else ""))
            for attr, prim_subpath in topology[arm]:
                setattr(self.scene, attr, ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{prim_subpath}",
                    update_period=contact_update_period,
                    history_length=1,
                    track_contact_points=True,
                    max_contact_data_count_per_prim=64,
                    track_pose=False,
                    filter_prim_paths_expr=contact_filter,
                ))
        print(f"[INFO] Contact sensors configured per arm: "
              + ", ".join(f"{arm}->{objs}" for arm, objs in detect_map.items()))

    def configure_assets_instance_paths(self, assets_instance_paths: dict[str, str]):
        """Set USD paths and compute Z positions for each asset instance.

        Args:
            assets_instance_paths (dict[str, str]): Maps scene asset names to asset instance
                directories containing a USD file and asset_size.json.

        Raises:
            ValueError: If an asset name is not found in the scene configuration.
        """
        # Per-object rigid-body sub-path for contact filtering, read by configure_contact_sensors.
        self._contact_body_links = getattr(self, "_contact_body_links", {})

        for asset_name, asset_instance_path in assets_instance_paths.items():
            asset_usd_path = get_asset_usd_path(asset_instance_path)

            if hasattr(self.scene, asset_name):
                asset_cfg = getattr(self.scene, asset_name)
                asset_cfg.spawn.usd_path = asset_usd_path
                self._contact_body_links[asset_name] = find_contact_body_link(asset_usd_path)

                size_info = load_asset_size(asset_instance_path)
                if size_info and 'size' in size_info and 'z' in size_info['size']:
                    asset_height = size_info['size']['z']
                    z_position = ROBOT.table_position[2] + asset_height / 2.0 + OBJ_Z_OFFSET

                    new_init_pos = list(asset_cfg.init_state.pos)
                    new_init_pos[2] = z_position
                    asset_cfg.init_state.pos = tuple(new_init_pos)
                else:
                    print(f"[WARNING] No size info found for {asset_name}, using default positioning")
            else:
                raise ValueError(f"Asset '{asset_name}' not found in scene configuration")

    def configure_for_replay(
        self,
        observation_modalities: List[str],
        enable_domain_randomization: bool = False,
        hdris_path: Optional[str] = None,
        materials_dir: Optional[str] = None,
        material_randomization: bool = True,
        **kwargs,
    ):
        """Configure the environment for trajectory replay with domain randomization.

        Forces num_envs=1 and disables terminations so replayed actions are never
        cut short. Use for augmenting recorded trajectories with visual diversity.

        Args:
            observation_modalities (list[str]): Modalities to expose during replay.
            enable_domain_randomization (bool): Whether to randomize materials and lighting.
            hdris_path (None or str): Path to HDRI folder for lighting randomization.
            materials_dir (None or str): Root directory for MDL material files.
            material_randomization (bool): Whether to randomize surface materials.
            **kwargs: Additional parameters forwarded to configure_observation_modalities.
        """
        if hasattr(self.scene, 'num_envs'):
            self.scene.num_envs = 1
            print(f"[INFO] Replay mode: forcing num_envs=1")

        if hasattr(self.terminations, "time_out"):
            self.terminations.time_out = None
        if hasattr(self.terminations, "task_success"):
            self.terminations.task_success = None

        self.configure_observation_modalities(observation_modalities, **kwargs)

        if enable_domain_randomization:
            lighting_randomization = hdris_path is not None
            self.configure_domain_randomization(
                enabled=True,
                material_randomization=material_randomization,
                lighting_randomization=lighting_randomization,
                hdris_path=hdris_path,
                materials_dir=materials_dir,
            )
        else:
            self.configure_domain_randomization(enabled=False)

        print(f"[INFO] Environment configured for replay")
        print(f"  - Domain randomization: {enable_domain_randomization}")
        if enable_domain_randomization:
            print(f"  - Material randomization: {material_randomization}")
            print(f"  - Lighting randomization: {lighting_randomization} (HDRI: {hdris_path is not None})")

    def configure_init_joint_pos_randomization(self, init_joint_pos_randomization: float = 0.35):
        """Set the joint position offset range for arm reset events.

        Args:
            init_joint_pos_randomization (float): Maximum joint noise magnitude in radians (+/-).
        """
        if hasattr(self.events, 'reset_robot_arms_left'):
            self.events.reset_robot_arms_left.params["position_range"] = (-init_joint_pos_randomization, init_joint_pos_randomization)
        if hasattr(self.events, 'reset_robot_arms_right'):
            self.events.reset_robot_arms_right.params["position_range"] = (-init_joint_pos_randomization, init_joint_pos_randomization)

        print(f"[INFO] Joint randomization configured: +-{init_joint_pos_randomization} rad")

    def configure_observation_modalities(self, observation_modalities: list[str], **kwargs):
        """Attach observation terms for the requested sensor modalities.

        Args:
            observation_modalities (list[str]): Subset of ['rgb', 'depth', 'proprioception',
                'background_mask'].
            **kwargs: Optional overrides:
                - enable_recording (bool): Add RGB to a recording group even when policy
                  doesn't use RGB (for video capture).

        Raises:
            AssertionError: If observation_modalities contains unrecognised strings.
        """
        assert type(observation_modalities) == list, "observation_modalities must be a list"
        assert all(isinstance(modality, str) for modality in observation_modalities), "observation_modalities must be a list of strings"
        validate_observation_modalities(observation_modalities)

        self.observations.policy.enable_corruption = False
        self.observations.policy.concatenate_terms = False

        # One image observation term per camera in the robot config, keyed
        # "<name>_camera_<suffix>" against scene sensor "<name>_camera".
        cameras = ROBOT.camera_names

        def _add_image_terms(group, data_type: str, suffix: str):
            for name in cameras:
                setattr(group, f"{name}_camera_{suffix}", ObservationTermCfg(
                    func=mdp.image,
                    params={"sensor_cfg": SceneEntityCfg(f"{name}_camera"),
                            "data_type": data_type, "normalize": False}))

        enable_recording = kwargs.get('enable_recording', False)
        policy_needs_rgb = 'rgb' in observation_modalities
        needs_recording_rgb = enable_recording and not policy_needs_rgb

        if 'rgb' in observation_modalities:
            _add_image_terms(self.observations.policy, "rgb", "rgb")

        if needs_recording_rgb:
            self.observations.add_recording_group()
            _add_image_terms(self.observations.recording, "rgb", "rgb")
            print(f"[INFO] Added RGB observations to recording group for video recording")

        if 'depth' in observation_modalities:
            _add_image_terms(self.observations.policy, "distance_to_image_plane", "depth")

        if 'proprioception' in observation_modalities:
            self.observations.policy.left_arm_joint_pos = ObservationTermCfg(func=get_arm_joint_pos, params={"asset_cfg": SceneEntityCfg("left_arm")})
            self.observations.policy.right_arm_joint_pos = ObservationTermCfg(func=get_arm_joint_pos, params={"asset_cfg": SceneEntityCfg("right_arm")})
            self.observations.policy.left_gripper_state = ObservationTermCfg(func=get_gripper_continuous_state, params={"asset_cfg": SceneEntityCfg("left_arm")})
            self.observations.policy.right_gripper_state = ObservationTermCfg(func=get_gripper_continuous_state, params={"asset_cfg": SceneEntityCfg("right_arm")})

        if 'background_mask' in observation_modalities:
            for name in cameras:
                setattr(self.observations.policy, f"{name}_camera_background_mask", ObservationTermCfg(
                    func=extract_background_mask,
                    params={"sensor_cfg": SceneEntityCfg(f"{name}_camera")}))

        # Depth is required as input for background mask generation.
        needs_depth_for_derived = (
            'background_mask' in observation_modalities
            and 'depth' not in observation_modalities
        )
        if needs_depth_for_derived:
            print("[WARNING] Background mask observations require depth data. Enabling depth observations.")
            _add_image_terms(self.observations.policy, "distance_to_image_plane", "depth")

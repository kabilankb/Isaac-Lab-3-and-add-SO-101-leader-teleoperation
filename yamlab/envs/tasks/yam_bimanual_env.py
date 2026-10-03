"""Base task environment and configuration factory built on IsaacLab's ManagerBasedRLEnv.

Defines ``make_task_env``, the generic environment factory shared by all tasks, and
``YamBimanualEnv``, the base environment providing observation management, recorder
integration, gripper grasp protection, pose scheduling, and domain randomization.
"""

import math
import os
import torch
from abc import abstractmethod
from typing import Dict, Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg as RigidBodyMaterialCfg
from isaaclab.sim.utils import bind_physics_material

from yamlab.envs.manipulation_env import ManipulationEnv
from .yam_bimanual_env_cfg import YamBimanualEnvCfg
from yamlab.configs import loader as config_loader
from yamlab.utils.recorders import ManualControlRecorderManager, LeRobotRecorderManager
from yamlab.utils.transforms import euler2quat
from yamlab.domain_randomization.events import DomainRandomizationManager
from yamlab.robot.yam import ROBOT, YamRobot, YamActionLayout

def _apply_resolved_config(cfg: YamBimanualEnvCfg, cfg_yaml: dict) -> None:
    """Stamp resolved YAML experiment knobs onto a task config in place.

    Sets simulation/physics params (dt, decimation, render interval, PhysX iteration
    counts), the sim device, episode length, per-object mass, material include/exclude,
    and stashes the grasp-detection params on the config for the robot's grasp detector.
    Camera resolution and num_envs are applied by the caller (they interact
    with explicit call-site overrides). Values absent from ``cfg_yaml`` are left untouched.

    Args:
        cfg (YamBimanualEnvCfg): Task configuration to mutate in place.
        cfg_yaml (dict): Resolved layered YAML config for the task and mode.
    """
    sim_yaml = cfg_yaml.get('sim', {})
    if 'dt' in sim_yaml:
        cfg.sim.dt = float(sim_yaml['dt'])
    if 'render_interval' in sim_yaml:
        cfg.sim.render_interval = int(sim_yaml['render_interval'])
    if 'enable_scene_query_support' in sim_yaml:
        cfg.sim.enable_scene_query_support = bool(sim_yaml['enable_scene_query_support'])
        cfg.sim.physics.enable_scene_query_support = cfg.sim.enable_scene_query_support
    if 'decimation' in sim_yaml:
        cfg.decimation = int(sim_yaml['decimation'])
    physx_yaml = sim_yaml.get('physx', {})
    if 'min_position_iteration_count' in physx_yaml:
        cfg.sim.physics.min_position_iteration_count = int(physx_yaml['min_position_iteration_count'])
    if 'min_velocity_iteration_count' in physx_yaml:
        cfg.sim.physics.min_velocity_iteration_count = int(physx_yaml['min_velocity_iteration_count'])
    if 'device' in sim_yaml:
        cfg.sim.device = sim_yaml['device']
    if 'episode_length_s' in sim_yaml:
        cfg.episode_length_s = float(sim_yaml['episode_length_s'])
    # Grasp-detection thresholds are a robot property (configs/robot/yam.yaml grasp:) read by
    # the robot model; the per-task grasp_detect block (which arm grasps which object) drives
    # configure_contact_sensors in make_task_env.
    # Per-object mass from the task YAML objects: block, applied to rigid objects that carry a
    # spawn.mass_props. Articulated objects (no mass_props) keep the mass from their USD.
    for object_name, object_spec in (cfg_yaml.get('objects') or {}).items():
        mass_kg = object_spec.get('mass')
        scene_obj = getattr(cfg.scene, object_name, None)
        if mass_kg is not None and scene_obj is not None and getattr(getattr(scene_obj, 'spawn', None), 'mass_props', None) is not None:
            scene_obj.spawn.mass_props.mass = float(mass_kg)
    # Per-object material include/exclude (configs/tasks/<task>.yaml
    # domain_randomization:); read by configure_domain_randomization. Empty ->
    # every task object randomizes over all discovered material categories.
    cfg._dr_cfg = dict(cfg_yaml.get('domain_randomization', {}) or {})

    # Arm controller PD gains are a robot property (configs/robot/yam.yaml controller.default),
    # built into the arm articulation at scene-construction time via YAM_CONFIG_DEFAULT. They
    # are therefore set on the scene's arm cfg, not from this resolved task config.


def _resolve_num_envs(requested, cfg_yaml: dict, fallback):
    """Resolve the number of parallel environments.

    An explicit call-site value wins, else the per-mode YAML default
    (configs/defaults.yaml modes.<mode>.num_envs), else ``fallback``.

    Args:
        requested (None or int): Explicit call-site value, or None.
        cfg_yaml (dict): Resolved layered YAML config.
        fallback (int): Value to use when neither source provides one.

    Returns:
        int: The resolved number of environments.
    """
    if requested is not None:
        return requested
    return cfg_yaml.get('sim', {}).get('num_envs', fallback)


def _resolve_camera_params(cfg_yaml: dict, requested_width, requested_height, requested_downsample):
    """Resolve camera render width/height and recorded-image downsample factor.

    Explicit call-site values win, else the YAML rendering defaults
    (configs/defaults.yaml rendering:).

    Args:
        cfg_yaml (dict): Resolved layered YAML config.
        requested_width (None or int): Explicit render width, or None.
        requested_height (None or int): Explicit render height, or None.
        requested_downsample (None or int): Explicit downsample factor, or None.

    Returns:
        tuple: (width, height, downsample), each (None or int).
    """
    render_yaml = cfg_yaml.get('rendering', {})
    width = requested_width if requested_width is not None else render_yaml.get('render_width')
    height = requested_height if requested_height is not None else render_yaml.get('render_height')
    downsample = (requested_downsample if requested_downsample is not None
                  else render_yaml.get('image_downsample_factor', 2))
    return width, height, downsample


def make_task_env(cfg_class: type[YamBimanualEnvCfg], env_class: type['YamBimanualEnv'], **kwargs):
    """Create a task environment with custom configuration.

    Generic factory compatible with any task environment. Instantiates the config,
    resolves the layered YAML config, applies common and mode-specific configuration, and
    passes remaining kwargs (including task-specific ones) to the environment constructor.

    Args:
        cfg_class (type[YamBimanualEnvCfg]): Configuration class for the task environment.
        env_class (type[YamBimanualEnv]): Environment class for the task environment.
        **kwargs: Configuration parameters including assets_instance_paths,
            objects_randomization, init_joint_pos_randomization, mode,
            observation_modalities, num_envs, enable_self_collisions, and
            task-specific parameters.

    Returns:
        YamBimanualEnv: Instance of the task environment.
    """
    cfg = cfg_class()

    # Per-arm grasp-detection targets come from the task YAML grasp.detect block
    # ({arm: [object, ...]}); only the listed arms get finger contact sensors. A
    # task cfg's CONTACT_OBJECT_NAMES class attr is the fallback (applied to both
    # arms) when no grasp.detect is set (e.g. a task with no YAML config).
    _resolved = config_loader.resolve(kwargs.get('env_name'))
    detect_map = {arm: list(objs)
                  for arm, objs in (_resolved.get('grasp_detect') or {}).items()
                  if objs}
    if not detect_map:
        _fallback = list(getattr(cfg_class, 'CONTACT_OBJECT_NAMES', []))
        if _fallback:
            detect_map = {'left_arm': _fallback, 'right_arm': _fallback}
    # Bind asset USDs before contact sensors, which read the rigid-body links discovered here.
    if 'assets_instance_paths' in kwargs:
        cfg.configure_assets_instance_paths(kwargs['assets_instance_paths'])

    cfg.configure_contact_sensors(detect_map)

    if 'objects_randomization' in kwargs:
        cfg.configure_objects_randomization(kwargs['objects_randomization'])
    if 'init_joint_pos_randomization' in kwargs:
        cfg.configure_init_joint_pos_randomization(kwargs['init_joint_pos_randomization'])

    mode = kwargs.pop('mode', 'teleoperation')
    teleoperation = (mode == 'teleoperation')
    policy_evaluation = (mode == 'evaluation')
    data_generation = (mode == 'mimicgen')
    is_replay = (mode == 'replay')

    # Stashed on the cfg for the env: drives the teleop-only grasp-preview overlay
    # (GraspRayVisualizer). _object_names are the task's scene objects -- the rays turn
    # green only when they intersect one of these (not the table/robot/ground).
    cfg._mode = mode
    cfg._object_names = list((_resolved.get('objects') or {}).keys())

    # Resolve the layered YAML config (configs/defaults.yaml + per-task +
    # per-mode) and stamp the experiment knobs (sim/physics, device, grasp,
    # episode length) onto the cfg. Explicit call-site values still win as
    # the top layer (num_envs just below, camera_* further down).
    env_name = kwargs.get('env_name')
    resolved_mode = mode if mode in config_loader.VALID_MODES else None
    cfg_yaml = config_loader.resolve(env_name, resolved_mode)
    _apply_resolved_config(cfg, cfg_yaml)

    # The robot's default grasp target: the object the arms grasp (left arm's first, else any
    # arm's). Read by YamBimanualEnv.__init__ when building self.robot; without it the robot
    # falls back to "obj_0" and silently never matches a task whose object is named otherwise.
    cfg._grasp_target = 'obj_0'
    if detect_map:
        _primary = detect_map.get('left_arm') or next((objs for objs in detect_map.values() if objs), [])
        if _primary:
            cfg._grasp_target = _primary[0]

    # Explicit device (the AppLauncher --device the entry script launched with)
    # wins over the per-mode YAML default, keeping the sim device == app device.
    device_override = kwargs.pop('device', None)
    if device_override is not None:
        cfg.sim.device = device_override

    # Number of parallel envs: an explicit call-site value wins, else the
    # per-mode YAML default (teleop/replay 1, mimicgen 4, evaluation 64).
    if hasattr(cfg.scene, 'num_envs'):
        cfg.scene.num_envs = _resolve_num_envs(kwargs.get('num_envs'), cfg_yaml, cfg.scene.num_envs)

    if teleoperation:
        enable_pose_schedule = kwargs.get('enable_pose_schedule', False)
        # pose_schedule lives under modes.teleoperation in the task YAML, so it is
        # only present in the teleop-resolved config (other modes never see it).
        cfg.configure_for_teleoperation(
            enable_pose_schedule=enable_pose_schedule,
            pose_schedule=cfg_yaml.get('pose_schedule'),
        )


    lerobot_output_root = kwargs.pop('lerobot_output_root', None)

    enable_domain_randomization = kwargs.pop('enable_domain_randomization', False)
    hdris_path = kwargs.pop('hdris_path', None)
    materials_dir = kwargs.pop('materials_dir', None)
    material_randomization = kwargs.pop('material_randomization', False)
    use_unseen_materials = kwargs.pop('use_unseen_materials', False)

    # Frame-skipping params are intentionally not popped; YamBimanualEnv.__init__ reads them.

    # Optional camera sensor render resolution + recorded-image downsample (data gen /
    # replay). Lowering the sensor resolution cuts both render time and the per-step
    # GPU->CPU image readback; intrinsics stay calibrated at 640x480 so the field of view
    # is preserved. The recorded image resolution is sensor_resolution //
    # image_downsample_factor. The actual sensor resolution (after any override, or the
    # scene default) is stored on cfg so the LeRobot recorder's feature shape stays
    # consistent with what the cameras render.
    camera_width, camera_height, image_downsample_factor = _resolve_camera_params(
        cfg_yaml,
        kwargs.pop('camera_width', None),
        kwargs.pop('camera_height', None),
        kwargs.pop('image_downsample_factor', None),
    )
    cam_attrs = [f"{name}_camera" for name in ROBOT.camera_names]
    for cam_name in cam_attrs:
        cam_cfg = getattr(cfg.scene, cam_name, None)
        # Bare camera prims (e.g. teleop, AssetBaseCfg) have no render resolution; skip.
        if cam_cfg is not None and hasattr(cam_cfg, "width"):
            if camera_width is not None:
                cam_cfg.width = camera_width
            if camera_height is not None:
                cam_cfg.height = camera_height
    ref_cam = getattr(cfg.scene, cam_attrs[0], None) if cam_attrs else None
    cfg._camera_width = getattr(ref_cam, "width", camera_width or 640)
    cfg._camera_height = getattr(ref_cam, "height", camera_height or 480)
    cfg._image_downsample_factor = image_downsample_factor if image_downsample_factor is not None else 2

    if data_generation:
        # num_envs was already applied above; each env runs an independent MimicGen trial and
        # episodes are de-interleaved by the recorder.
        cfg.configure_for_data_generation(
            enable_domain_randomization=enable_domain_randomization,
            hdris_path=hdris_path,
            materials_dir=materials_dir,
            material_randomization=material_randomization,
        )
        cfg._lerobot_output_root = lerobot_output_root
        task_description = kwargs.pop('task_description', None)
        cfg._task_description = task_description

    if policy_evaluation:
        observation_modalities = kwargs.get('observation_modalities', ['rgb', 'proprioception'])
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['observation_modalities']}
        cfg.configure_for_policy_evaluation(
            observation_modalities,
            enable_domain_randomization=enable_domain_randomization,
            hdris_path=hdris_path,
            materials_dir=materials_dir,
            use_unseen_materials=use_unseen_materials,
            **filtered_kwargs,
        )
        # Signal task environments to apply evaluation-specific success criteria.
        kwargs['is_evaluation_mode'] = True
    elif is_replay:
        observation_modalities = kwargs.pop('observation_modalities', ['rgb', 'proprioception'])
        cfg.configure_for_replay(
            observation_modalities,
            enable_domain_randomization=enable_domain_randomization,
            hdris_path=hdris_path,
            materials_dir=materials_dir,
            material_randomization=material_randomization,
            **kwargs,
        )
    elif 'observation_modalities' in kwargs:
        cfg.configure_observation_modalities(**kwargs)

    if 'concatenate_terms' in kwargs:
        cfg.observations.policy.concatenate_terms = bool(kwargs['concatenate_terms'])

    if 'enable_self_collisions' in kwargs and kwargs['enable_self_collisions']:
        cfg.scene.left_arm.spawn.articulation_props.enabled_self_collisions = True
        cfg.scene.right_arm.spawn.articulation_props.enabled_self_collisions = True

    return env_class(cfg, **kwargs)


class YamBimanualEnv(ManipulationEnv):
    """Base task environment built on IsaacLab's ManagerBasedRLEnv framework.

    Provides automatic observation management, recorder integration, gripper grasp
    protection, pose scheduling for teleoperation, domain randomization, and the
    LeRobot dataset recording API.
    """

    def __init__(self, cfg: YamBimanualEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the base task environment.

        Args:
            cfg (YamBimanualEnvCfg): Environment configuration.
            render_mode (None or str): Render mode string passed to the parent class.
            **kwargs: Additional keyword arguments forwarded to the parent and used
                for optional features (enable_gripper_grasp_clamp, etc.).
        """
        # This env records via a custom ManualControlRecorderManager (manual save/discard),
        # so the cfg.recorders is withheld during super().__init__ (which would build its own
        # recorder) and the custom manager is attached just below.
        original_recorders = cfg.recorders
        cfg.recorders = None
        super().__init__(cfg, render_mode, **kwargs)

        cfg.recorders = original_recorders
        if cfg.recorders is not None:
            self.recorder_manager = ManualControlRecorderManager(cfg.recorders, self)
            self.recorder_manager.flush_steps = 100
            self.recorder_manager.compression = 'lzf'

        self._episode_count = 0
        self._recording_active = False

        self._discard_first_n_frames = kwargs.get('discard_first_n_frames', 0)
        self._discard_last_n_frames = kwargs.get('discard_last_n_frames', 0)

        # Runtime robot model. Grasp queries go through it: self.robot.is_grasping(),
        # self.robot.left_arm.is_grasping(), self.robot.<arm>.end_effector.is_grasping(), and
        # self.robot.is_grasping(target_object="<name>") for other objects. Detection thresholds
        # are a robot property (configs/robot/yam.yaml grasp:); the default grasp target is the
        # object the arms grasp (set on cfg._grasp_target in make_task_env, else "obj_0").
        grasp_target = getattr(self.cfg, "_grasp_target", "obj_0")
        self.robot = YamRobot(self.scene, grasp_target=grasp_target)

        # Teleoperation-only grasp-preview overlay: a red/green inter-finger beam showing
        # whether closing each gripper would grasp an object. Always on during teleop
        # (independent of recording); uses the CPU-only PhysX raycast (teleop is CPU).
        # Driven by a RENDER callback (not step()) so the beam is redrawn from the latest
        # finger poses on every rendered frame, tracking the fingers in real time instead
        # of lagging one control step behind (step() updates would render one frame late).
        self.grasp_ray_visualizer = None
        if getattr(self.cfg, "_mode", None) == "teleoperation" and kwargs.get("enable_grasp_ray_viz", True):
            from yamlab.utils.grasp import GraspRayVisualizer
            self.grasp_ray_visualizer = GraspRayVisualizer(
                self, target_objects=getattr(self.cfg, "_object_names", [])
            )
            _overlay_cb = lambda event: self.grasp_ray_visualizer.update()
            if hasattr(self.sim, "add_render_callback"):
                self.sim.add_render_callback("grasp_ray_overlay", _overlay_cb)
            else:
                # Isaac Lab 3.0-beta: SimulationContext.render() still invokes _render_callbacks
                # after every render, but the public add_render_callback() is gone.
                if getattr(self.sim, "_render_callbacks", None) is None:
                    self.sim._render_callbacks = {}
                self.sim._render_callbacks["grasp_ray_overlay"] = _overlay_cb

        # Clamp gripper command when a grasp is detected to avoid over-closing
        # (~1 cm extra beyond actual finger position is allowed).
        self.enable_gripper_grasp_clamp = kwargs.get('enable_gripper_grasp_clamp', False)
        self.gripper_grasp_clamp_extra = kwargs.get('gripper_grasp_clamp_extra', 0.0075)
        # Per-step [GRIPPER CLAMP] clamp-start/end logs are off unless explicitly enabled.
        self._gripper_clamp_verbose = kwargs.get('gripper_clamp_verbose', False)
        self._prev_left_grasping = False
        self._prev_right_grasping = False
        if self.enable_gripper_grasp_clamp:
            print(f"[INFO] Gripper grasp protection enabled (extra closure: {self.gripper_grasp_clamp_extra} rad)")

        self._apply_physics_materials_to_grippers()

        self._pose_schedule_index = 0
        self._pose_schedule_data = cfg.pose_schedule_data
        self._pose_schedule_effective_ranges = cfg.pose_schedule_effective_ranges

        if self._pose_schedule_data is not None:
            # Record each scheduled object's initial position and yaw. Fractional schedule
            # entries (pos_fraction/rot_fraction) are resolved at runtime as offsets from these.
            self._pose_schedule_init_state = {}
            obj_schedule = self._pose_schedule_data.get("obj_pose_schedule", {})
            for obj_name in obj_schedule.keys():
                if obj_name in self.scene.keys():
                    obj = self.scene[obj_name]
                    ds = obj.data.default_root_pose.torch  # (N, 7) [pos, quat xyzw]
                    qx, qy, qz, qw = ds[0, 3].item(), ds[0, 4].item(), ds[0, 5].item(), ds[0, 6].item()
                    default_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    self._pose_schedule_init_state[obj_name] = {
                        "pos_x": ds[0, 0].item(),
                        "pos_y": ds[0, 1].item(),
                        "default_yaw": default_yaw,
                    }
            self._apply_scheduled_poses_to_default_state()

        if hasattr(cfg, '_lerobot_output_root') and cfg._lerobot_output_root:
            task_description = getattr(cfg, '_task_description', None)
            if task_description:
                task_name = task_description
                print(f"[INFO] Using task description for LeRobot: {task_description}")
            else:
                task_name = cfg.__class__.__name__.replace("Cfg", "").replace("EnvCfg", "")
                print(f"[INFO] No task description provided, using class name: {task_name}")
            output_name = os.path.basename(cfg._lerobot_output_root)
            self.configure_for_lerobot_recording(
                output_dir=cfg._lerobot_output_root,
                repo_id=f"local/{output_name}",
                fps=30,
                task_name=task_name,
                image_downsample_factor=getattr(cfg, "_image_downsample_factor", 2),
                base_width=getattr(cfg, "_camera_width", 640),
                base_height=getattr(cfg, "_camera_height", 480),
            )
            print(f"[INFO] LeRobot recorder configured for MimicGen: {cfg._lerobot_output_root}")

        self._domain_rand_manager = None
        env_name = kwargs.get('env_name', None)
        if hasattr(cfg, 'domain_randomization') and cfg.domain_randomization is not None:
            self._domain_rand_manager = DomainRandomizationManager(cfg.domain_randomization, env_name=env_name)
            if cfg.domain_randomization.enabled:
                self._domain_rand_manager.initialize(self)

        print(f"[INFO] Base Task Environment initialized")
        print(f"[INFO] - Using ManagerBasedRLEnv framework")
        print(f"[INFO] - Automatic observation management: {len(self.observation_manager.group_obs_term_dim)} terms")
        print(f"[INFO] - Recording enabled: {self.recorder_manager is not None}")
        print(f"[INFO] - Grasp detector initialized")
        print(f"[INFO] - Physics materials applied to gripper links")
        print(f"[INFO] - Domain randomization: {self._domain_rand_manager.is_enabled if self._domain_rand_manager else False}")

    def get_object_names_for_physics_material(self) -> list[str]:
        """Return names of scene objects that should receive custom physics materials.

        Override in task subclasses to specify objects that need non-default friction.
        Default returns an empty list (no objects are modified).

        Returns:
            list[str]: Scene object names.
        """
        return []

    def _apply_physics_material_to_object(
        self,
        object_name: str,
        static_friction: float = 1.5,
        dynamic_friction: float = 1.5,
        restitution: float = 0.0,
        friction_combine_mode: str = "average",
        restitution_combine_mode: str = "average"
    ):
        """Apply a physics material to all collision meshes of a scene object.

        bind_physics_material is decorated with @apply_nested, so binding to the
        object root automatically covers the full collision hierarchy.

        Args:
            object_name (str): Scene object name (e.g., "obj_0").
            static_friction (float): Static friction coefficient.
            dynamic_friction (float): Dynamic friction coefficient.
            restitution (float): Restitution (bounciness) coefficient.
            friction_combine_mode (str): PhysX combine mode for friction
                ("average", "min", "max", "multiply").
            restitution_combine_mode (str): PhysX combine mode for restitution.
        """
        obj = self.scene[object_name]
        obj_prim_path = obj.root_prim_path if hasattr(obj, 'root_prim_path') else None

        if obj_prim_path is None:
            obj_prim_path = f"/World/envs/env_0/{object_name}"

        physics_material_cfg = RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
            friction_combine_mode=friction_combine_mode,
            restitution_combine_mode=restitution_combine_mode,
        )
        material_path = f"{obj_prim_path}/physicsMaterial"
        physics_material_cfg.func(material_path, physics_material_cfg)
        bind_physics_material(obj_prim_path, material_path)

        print(f"[INFO] Applied physics material with higher friction to {object_name} at {obj_prim_path}")

    def _apply_physics_materials_to_objects(self):
        """Apply physics materials to all objects from get_object_names_for_physics_material()."""
        object_names = self.get_object_names_for_physics_material()
        for object_name in object_names:
            self._apply_physics_material_to_object(object_name)

    def _apply_physics_material_to_gripper_link(
        self,
        arm_name: str,
        link_name: str,
        static_friction: float = 3.0,
        dynamic_friction: float = 3.0,
        restitution: float = 0.0,
        friction_combine_mode: str = "average",
        restitution_combine_mode: str = "average"
    ):
        """Apply a physics material to a single gripper finger link.

        Args:
            arm_name (str): Scene name of the arm ("left_arm" or "right_arm").
            link_name (str): Finger link name ("left_finger" or "right_finger").
            static_friction (float): Static friction coefficient.
            dynamic_friction (float): Dynamic friction coefficient.
            restitution (float): Restitution coefficient.
            friction_combine_mode (str): PhysX friction combine mode.
            restitution_combine_mode (str): PhysX restitution combine mode.
        """
        arm = self.scene[arm_name]

        arm_prim_path = None
        if hasattr(arm, 'root_prim_path'):
            arm_prim_path = arm.root_prim_path
        elif hasattr(arm, 'prim_path'):
            arm_prim_path = arm.prim_path
        elif hasattr(arm, '_root_prim_path'):
            arm_prim_path = arm._root_prim_path

        if arm_prim_path is None:
            arm_name_map = {"left_arm": "LeftArm", "right_arm": "RightArm"}
            arm_scene_name = arm_name_map.get(arm_name, arm_name.replace("_", "").title())
            arm_prim_path = f"/World/envs/env_0/{arm_scene_name}"

        # Gripper links are nested under /arm/ per the YAM USD hierarchy.
        link_prim_path = f"{arm_prim_path}/arm/{link_name}"

        physics_material_cfg = RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
            friction_combine_mode=friction_combine_mode,
            restitution_combine_mode=restitution_combine_mode,
        )
        material_path = f"{link_prim_path}/physicsMaterial"
        physics_material_cfg.func(material_path, physics_material_cfg)
        bind_physics_material(link_prim_path, material_path)

        print(f"[INFO] Applied physics material to {arm_name} gripper link {link_name} at {link_prim_path}")

    def _apply_physics_materials_to_grippers(
        self,
        static_friction: float = 3.0,
        dynamic_friction: float = 3.0,
        restitution: float = 0.0,
        friction_combine_mode: str = "average",
        restitution_combine_mode: str = "average"
    ):
        """Apply physics materials to the left_finger and right_finger links on both arms.

        Args:
            static_friction (float): Static friction coefficient.
            dynamic_friction (float): Dynamic friction coefficient.
            restitution (float): Restitution coefficient.
            friction_combine_mode (str): PhysX friction combine mode.
            restitution_combine_mode (str): PhysX restitution combine mode.
        """
        gripper_links = ["left_finger", "right_finger"]

        for link_name in gripper_links:
            self._apply_physics_material_to_gripper_link(
                arm_name="left_arm",
                link_name=link_name,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
                friction_combine_mode=friction_combine_mode,
                restitution_combine_mode=restitution_combine_mode
            )

        for link_name in gripper_links:
            self._apply_physics_material_to_gripper_link(
                arm_name="right_arm",
                link_name=link_name,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
                friction_combine_mode=friction_combine_mode,
                restitution_combine_mode=restitution_combine_mode
            )

        print(f"[INFO] Applied physics materials to all gripper links (4 total: 2 per arm)")

    def step(self, actions):
        """Execute one simulation step.

        Handles gripper grasp protection clamping, right_finger mirroring, and
        LeRobot frame recording. Task subclasses can override and call
        super().step(actions) for this base behaviour.

        Args:
            actions (torch.Tensor): float32, shape (num_envs, 14), absolute joint-position
                targets, format [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
                May be a CPU tensor or array-like; it is moved to the sim device.

        Returns:
            tuple: (obs_dict, reward, terminated, truncated, info).
        """
        # Ensure the action lives on the simulation device before it is used below.
        # The MimicGen loop builds the action tensor on CPU; without this, mirroring
        # the finger target onto a CUDA articulation buffer (or any other use of the
        # action) raises a device-mismatch error. This lets the sim `device` in
        # the sim device can be switched (e.g. to "cuda") without crashing.
        if isinstance(actions, torch.Tensor):
            actions = actions.to(self.device)
        else:
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        layout = YamActionLayout
        left_gripper_idx, right_gripper_idx = layout.LEFT_GRIPPER, layout.RIGHT_GRIPPER
        arm_gripper_joint_idx = layout.ARM_GRIPPER_JOINT_INDEX

        if self.enable_gripper_grasp_clamp:
            left_arm_ref = self.scene["left_arm"]
            right_arm_ref = self.scene["right_arm"]
            left_actual = left_arm_ref.data.joint_pos.torch[:, arm_gripper_joint_idx:arm_gripper_joint_idx + 1]
            right_actual = right_arm_ref.data.joint_pos.torch[:, arm_gripper_joint_idx:arm_gripper_joint_idx + 1]

            left_grasping, right_grasping = self.robot.is_grasping()

            max_left = torch.clamp(left_actual + self.gripper_grasp_clamp_extra,
                                   max=ROBOT.finger_closed('left'))
            max_right = torch.clamp(right_actual + self.gripper_grasp_clamp_extra,
                                    max=ROBOT.finger_closed('left'))

            left_clamping_now = bool(left_grasping[0]) and bool(actions[0, left_gripper_idx] > max_left[0, 0])
            right_clamping_now = bool(right_grasping[0]) and bool(actions[0, right_gripper_idx] > max_right[0, 0])

            if self._gripper_clamp_verbose:
                if left_clamping_now and not self._prev_left_grasping:
                    print(f"[GRIPPER CLAMP] Left arm: clamping started (cmd={actions[0, left_gripper_idx].item():.4f} -> {max_left[0, 0].item():.4f}, actual={left_actual[0, 0].item():.4f})")
                elif not left_clamping_now and self._prev_left_grasping:
                    print(f"[GRIPPER CLAMP] Left arm: clamping ended")

                if right_clamping_now and not self._prev_right_grasping:
                    print(f"[GRIPPER CLAMP] Right arm: clamping started (cmd={actions[0, right_gripper_idx].item():.4f} -> {max_right[0, 0].item():.4f}, actual={right_actual[0, 0].item():.4f})")
                elif not right_clamping_now and self._prev_right_grasping:
                    print(f"[GRIPPER CLAMP] Right arm: clamping ended")

            self._prev_left_grasping = left_clamping_now
            self._prev_right_grasping = right_clamping_now

            if left_grasping.any():
                actions[:, left_gripper_idx:left_gripper_idx + 1] = torch.where(
                    left_grasping.unsqueeze(-1) & (actions[:, left_gripper_idx:left_gripper_idx + 1] > max_left),
                    max_left,
                    actions[:, left_gripper_idx:left_gripper_idx + 1]
                )
            if right_grasping.any():
                actions[:, right_gripper_idx:right_gripper_idx + 1] = torch.where(
                    right_grasping.unsqueeze(-1) & (actions[:, right_gripper_idx:right_gripper_idx + 1] > max_right),
                    max_right,
                    actions[:, right_gripper_idx:right_gripper_idx + 1]
                )

        # The action drives only left_finger; mirror it onto right_finger for both arms.
        left_arm = self.scene["left_arm"]
        right_arm = self.scene["right_arm"]
        left_finger_pos = actions[:, left_gripper_idx:left_gripper_idx + 1]
        right_finger_pos = actions[:, right_gripper_idx:right_gripper_idx + 1]
        left_arm.set_joint_position_target_index(target=left_finger_pos.expand(-1, 1), joint_ids=[left_arm.joint_names.index(layout.MIRROR_JOINT)])
        right_arm.set_joint_position_target_index(target=right_finger_pos.expand(-1, 1), joint_ids=[right_arm.joint_names.index(layout.MIRROR_JOINT)])

        obs_dict, reward, terminated, truncated, info = super().step(actions)

        if hasattr(self, 'lerobot_recorder') and self.lerobot_recorder is not None:
            if self.lerobot_recorder.is_recording():
                self.lerobot_recorder.record_frame(obs_dict, actions)

        if self._domain_rand_manager is not None and self._domain_rand_manager.is_enabled:
            self._domain_rand_manager.maybe_apply_randomization_on_step()

        # (The teleop grasp-preview beam is redrawn by a render callback, not here, so it
        # tracks the fingers per rendered frame instead of lagging a control step behind.)

        return obs_dict, reward, terminated, truncated, info

    def reset(self, warm_up=True, seed=None, env_ids=None, options=None):
        """Reset the environment with optional warm-up simulation steps.

        Task subclasses can override and call super().reset() to inherit this logic.

        Args:
            warm_up (bool): If True, run 4 simulation steps and 4 render steps after reset
                to let physics settle before returning observations. Skipped on partial resets.
            seed (None or int): Random seed forwarded to the parent reset.
            env_ids (None or torch.Tensor): Environment indices to reset; None resets all.
            options (None or dict): Options forwarded to the parent reset.

        Returns:
            tuple: (obs_dict, extras).
        """
        obs_dict, extras = super().reset(seed=seed, env_ids=env_ids, options=options)

        if self._domain_rand_manager is not None and self._domain_rand_manager.is_enabled:
            if env_ids is None:
                reset_env_ids = torch.arange(self.num_envs, device=self.device)
            else:
                reset_env_ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
            self._domain_rand_manager.reset_step_count()
            self._domain_rand_manager.apply_randomization(reset_env_ids)

        if (hasattr(self, 'lerobot_recorder') and
                self.lerobot_recorder is not None and
                hasattr(self.cfg, 'datagen_config') and
                self.cfg.datagen_config is not None):
            if not self.lerobot_recorder.is_recording():
                self.lerobot_recorder.start_recording()

        # The warm-up steps physics for ALL environments. That is only safe when the
        # whole scene is being reset; for a partial reset (a single env finishing its
        # trial during parallel MimicGen generation) it would advance the other,
        # still-running environments with stale actions and corrupt their demos. So
        # only warm up on a full reset.
        is_full_reset = env_ids is None or len(env_ids) >= self.num_envs
        if warm_up and is_full_reset:
            for _ in range(4):
                self.sim.step(render=False)
            for _ in range(4):
                self.sim.render()
            obs_dict = self.observation_manager.compute(update_history=True)
            task_info = self.get_current_task_info()
            extras.update(task_info)

        return obs_dict, extras


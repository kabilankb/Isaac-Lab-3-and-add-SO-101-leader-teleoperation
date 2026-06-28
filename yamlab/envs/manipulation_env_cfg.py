"""Generic, robot-agnostic base configuration for manipulation task environments.

:class:`ManipulationEnvCfg` collects the configuration helpers that do not depend on a
particular robot: object position/orientation randomization, domain randomization, and the
teleoperation pose schedule. A robot-specific base config (e.g.
:class:`~yamlab.envs.tasks.yam_bimanual_env_cfg.YamBimanualEnvCfg`) subclasses it and adds the
robot's scene, actions, observations, and events.
"""

import math
from typing import Optional, List

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from yamlab.domain_randomization import DomainRandomizationCfg, build_domain_randomization_cfg


@configclass
class ManipulationEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base config: object randomization, domain randomization, pose schedule.

    Holds no robot-specific fields; subclasses provide the scene, actions, observations,
    events, and the ``obj_name_to_event_name`` map these helpers operate on.
    """

    def configure_objects_randomization(
        self,
        objects_randomization: dict[str, tuple]
    ):
        """Compute and apply effective +/- position ranges for each object's reset event.

        Two modes (exactly one set per object):
          - position_range=(dx, dy): direct +/- distance, used as-is.
          - region_size=(rx, ry): bounding box -- the effective range subtracts the object's
            max world-frame half-extent over the yaw range from the region half-size, so the
            footprint stays inside the box. Uses MAX of axis projections rather than SUM
            (round objects with handles have overlapping body/handle projections).

        Args:
            objects_randomization (dict[str, tuple]): Maps object names to
                (region_size, asset_size, orientation_range, scale_range, position_range),
                where region_size/position_range is (x, y) or None (one of the two is set),
                asset_size=(sx, sy) is the local-frame footprint, and orientation_range is
                the yaw half-range in radians.

        Raises:
            ValueError: If an object name has no corresponding randomization event.
        """
        for object_name, (region_size, asset_size, orientation_range, scale_range, position_range) in objects_randomization.items():
            if object_name not in self.obj_name_to_event_name:
                raise ValueError(f"No randomization function defined for object: {object_name}")

            # Direct mode: the +/- distance is used as-is, no footprint subtraction.
            if position_range is not None:
                eff_x, eff_y = max(0.0, position_range[0]), max(0.0, position_range[1])
                print(f"[INFO] {object_name}: direct position_range=+/-({eff_x:.4f}, {eff_y:.4f})")
                event_name = self.obj_name_to_event_name[object_name]
                self.configure_single_object_randomization(event_name, (eff_x, eff_y, 0.0), orientation_range, scale_range)
                continue

            region_x, region_y = region_size
            asset_sx, asset_sy = asset_size

            # Extract default yaw from init_state quaternion using ZYX Euler decomposition:
            #   yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
            if hasattr(self.scene, object_name):
                asset_cfg = getattr(self.scene, object_name)
                qw, qx, qy, qz = asset_cfg.init_state.rot
                default_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            else:
                default_yaw = 0.0

            a, b = asset_sx / 2.0, asset_sy / 2.0
            max_half_x, max_half_y = a, b

            if orientation_range > 0 and (a > 0 or b > 0):
                yaw_min = default_yaw - orientation_range
                yaw_max = default_yaw + orientation_range
                num_samples = max(36, int(math.degrees(orientation_range)))
                max_half_x = 0.0
                max_half_y = 0.0
                for i in range(num_samples + 1):
                    yaw = yaw_min + (yaw_max - yaw_min) * i / num_samples
                    hx = max(a * abs(math.cos(yaw)), b * abs(math.sin(yaw)))
                    hy = max(a * abs(math.sin(yaw)), b * abs(math.cos(yaw)))
                    max_half_x = max(max_half_x, hx)
                    max_half_y = max(max_half_y, hy)
            elif orientation_range == 0 and (a > 0 or b > 0):
                max_half_x = max(a * abs(math.cos(default_yaw)), b * abs(math.sin(default_yaw)))
                max_half_y = max(a * abs(math.sin(default_yaw)), b * abs(math.cos(default_yaw)))

            eff_x = max(0.0, region_x / 2.0 - max_half_x)
            eff_y = max(0.0, region_y / 2.0 - max_half_y)
            print(f"[INFO] {object_name}: default_yaw={math.degrees(default_yaw):.1f}deg, "
                  f"asset_size=({asset_sx:.4f}, {asset_sy:.4f}), "
                  f"max_world_extent=({2*max_half_x:.4f}, {2*max_half_y:.4f}), "
                  f"region_size=({region_x:.4f}, {region_y:.4f}) -> "
                  f"effective_position_range=({eff_x:.4f}, {eff_y:.4f})")

            event_name = self.obj_name_to_event_name[object_name]
            self.configure_single_object_randomization(event_name, (eff_x, eff_y, 0.0), orientation_range, scale_range)

    def configure_single_object_randomization(
        self,
        event_name: str,
        position_range: tuple = (0.05, 0.2, 0.0),
        orientation_range: float = 3.14159,
        scale_range: tuple = (1.0, 1.0),
    ):
        """Update pose_range parameters for a single object's reset event.

        Args:
            event_name (str): Name of the reset event (e.g., "reset_obj_0_position").
            position_range (tuple): Half-ranges (x, y, z) in meters for position randomization.
            orientation_range (float): Yaw half-range in radians.
            scale_range (tuple): (min, max) scale multiplier (unused by default events).

        Raises:
            ValueError: If event_name is not found in the events configuration.
        """
        if hasattr(self.events, event_name):
            event = getattr(self.events, event_name)
            event.params["pose_range"].update({
                "x": (-position_range[0], position_range[0]),
                "y": (-position_range[1], position_range[1]),
                "z": (-position_range[2], position_range[2]),
                "yaw": (-orientation_range, orientation_range),
            })
        else:
            raise ValueError(f"Event {event_name} not found in events configuration. Please define the event in the task environment class.")

    def configure_domain_randomization(
        self,
        enabled: bool = True,
        object_names: Optional[List[str]] = None,
        material_randomization: bool = True,
        lighting_randomization: bool = True,
        hdris_path: Optional[str] = None,
        materials_dir: Optional[str] = None,
        num_variants_per_material: Optional[int] = None,
        use_unseen_materials: bool = False,
    ):
        """Build and attach a DomainRandomizationCfg for material and lighting diversity.

        Should be enabled for replay, MimicGen, and evaluation; disabled for
        teleoperation to keep demonstrations clean.

        Args:
            enabled (bool): Master switch. If False, attaches a disabled config and returns.
            object_names (None or list[str]): Objects to randomize materials on; defaults to
                all task objects.
            material_randomization (bool): Whether to randomize object surface materials.
            lighting_randomization (bool): Whether to randomize scene lighting.
            hdris_path (None or str): Folder of .hdr files for HDRI lighting; None gives
                intensity/color only.
            materials_dir (None or str): Root directory for MDL material files (local);
                None gives S3 URLs.
            num_variants_per_material (None or int): Preloaded variants per base material
                (default 2).
            use_unseen_materials (bool): Use holdout materials for out-of-distribution evaluation.
        """
        if not enabled:
            self.domain_randomization = DomainRandomizationCfg(enabled=False)
            print("[INFO] Domain randomization disabled")
            return

        if object_names is None:
            object_names = list(self.obj_name_to_event_name.keys())

        # Tunable randomization knobs come from the resolved YAML config
        # (configs/defaults.yaml domain_randomization: + per-task overrides),
        # stashed on cfg as _dr_cfg by make_task_env. The builder maps them onto
        # the typed configclasses; runtime inputs (paths, split, objects) are
        # passed in. Per-object material categories live under materials.per_object.
        dr_yaml = dict(getattr(self, "_dr_cfg", {}) or {})
        if num_variants_per_material is not None:
            materials_yaml = dict(dr_yaml.get("materials", {}) or {})
            materials_yaml["num_variants_per_material"] = num_variants_per_material
            dr_yaml["materials"] = materials_yaml

        self.domain_randomization = build_domain_randomization_cfg(
            dr_yaml,
            enabled=True,
            material_randomization=material_randomization,
            lighting_randomization=lighting_randomization,
            materials_dir=materials_dir,
            hdris_path=hdris_path,
            use_unseen_materials=use_unseen_materials,
            object_names=object_names,
            prim_groups_per_object=getattr(self, "material_prim_groups", {}) or {},
        )

        print(f"[INFO] Domain randomization configured:")
        print(f"  - Materials: {material_randomization} (objects: {self.domain_randomization.materials.object_names})")
        print(f"  - Lighting: {lighting_randomization} (HDRI: {hdris_path is not None})")

    def configure_pose_schedule(self, obj_pose_schedule: dict):
        """Apply a per-task pose schedule and disable randomization for scheduled objects.

        Objects in the schedule will be placed at exact poses from a fixed sequence
        instead of using the randomization event. Absolute (pos/rot) or fractional
        (pos_fraction/rot_fraction) pose entries are supported; fractions map [0, 1]
        linearly across the pre-schedule effective randomization range.

        Args:
            obj_pose_schedule (dict): ``{obj_name: [pose_entry, ...]}`` mapping -- the
                ``pose_schedule`` section of the task's YAML config (resolved for
                the teleoperation mode).

        Raises:
            ValueError: If the schedule is empty/missing or an individual entry is invalid.
        """
        if not obj_pose_schedule:
            raise ValueError(
                "enable_pose_schedule=True but the task config has no 'pose_schedule'. "
                "Add it under modes.teleoperation in configs/tasks/<task>.yaml."
            )

        for obj_name, pose_list in obj_pose_schedule.items():
            if not isinstance(pose_list, list) or len(pose_list) == 0:
                raise ValueError(f"Pose schedule for '{obj_name}' must be a non-empty list. Got: {type(pose_list)}")

            for i, pose in enumerate(pose_list):
                has_pos = "pos" in pose
                has_pos_frac = "pos_fraction" in pose
                assert not (has_pos and has_pos_frac), (
                    f"'{obj_name}' entry {i}: specify either 'pos' or 'pos_fraction', not both"
                )

                if has_pos:
                    pos = pose["pos"]
                    if not isinstance(pos, (list, tuple)):
                        raise ValueError(
                            f"Position for '{obj_name}' entry {i} must be a list or tuple. Got: {type(pos)}"
                        )
                    if len(pos) not in [2, 3]:
                        raise ValueError(
                            f"Position for '{obj_name}' entry {i} must be [x, y] or [x, y, z] (z will be ignored). Got length {len(pos)}: {pos}"
                        )

                if has_pos_frac:
                    pf = pose["pos_fraction"]
                    if not isinstance(pf, (list, tuple)) or len(pf) != 2:
                        raise ValueError(
                            f"pos_fraction for '{obj_name}' entry {i} must be [fx, fy]. Got: {pf}"
                        )
                    for j, v in enumerate(pf):
                        if not (0.0 <= v <= 1.0):
                            raise ValueError(
                                f"pos_fraction[{j}] for '{obj_name}' entry {i} must be in [0, 1]. Got: {v}"
                            )

                has_rot = "rot" in pose
                has_rot_frac = "rot_fraction" in pose
                assert not (has_rot and has_rot_frac), (
                    f"'{obj_name}' entry {i}: specify either 'rot' or 'rot_fraction', not both"
                )

                if has_rot:
                    rot = pose["rot"]
                    if isinstance(rot, dict):
                        if not all(k in rot for k in ["roll", "pitch", "yaw"]):
                            raise ValueError(
                                f"Rotation dict for '{obj_name}' entry {i} must have 'roll', 'pitch', 'yaw'. Got: {list(rot.keys())}"
                            )
                    elif isinstance(rot, (list, tuple)):
                        if len(rot) != 4:
                            raise ValueError(
                                f"Rotation quaternion for '{obj_name}' entry {i} must be [w, x, y, z]. Got: {rot}"
                            )
                    else:
                        raise ValueError(
                            f"Rotation for '{obj_name}' entry {i} must be dict or quaternion. Got: {type(rot)}"
                        )

                if has_rot_frac:
                    rf = pose["rot_fraction"]
                    if not isinstance(rf, dict) or "yaw" not in rf:
                        raise ValueError(
                            f"rot_fraction for '{obj_name}' entry {i} must be a dict with 'yaw' key. Got: {rf}"
                        )
                    yf = rf["yaw"]
                    if not (0.0 <= yf <= 1.0):
                        raise ValueError(
                            f"rot_fraction['yaw'] for '{obj_name}' entry {i} must be in [0, 1]. Got: {yf}"
                        )

        self.pose_schedule_data = {"obj_pose_schedule": obj_pose_schedule}

        # Record each scheduled object's effective randomization range so runtime can resolve
        # fractional schedule entries; the schedule then disables randomization for these objects.
        effective_ranges = {}
        for obj_name in obj_pose_schedule.keys():
            if obj_name in self.obj_name_to_event_name:
                event_name = self.obj_name_to_event_name[obj_name]
                if hasattr(self.events, event_name):
                    event = getattr(self.events, event_name)
                    pose_range = event.params.get("pose_range", {})
                    # Ranges are stored as (-eff, +eff); extract the positive half.
                    x_range = pose_range.get("x", (0.0, 0.0))
                    y_range = pose_range.get("y", (0.0, 0.0))
                    yaw_range = pose_range.get("yaw", (0.0, 0.0))
                    effective_ranges[obj_name] = {
                        "pos_range": (x_range[1], y_range[1]),
                        "orientation_range": yaw_range[1],
                    }
        self.pose_schedule_effective_ranges = effective_ranges

        for obj_name in obj_pose_schedule.keys():
            if obj_name in self.obj_name_to_event_name:
                event_name = self.obj_name_to_event_name[obj_name]
                if hasattr(self.events, event_name):
                    event = getattr(self.events, event_name)
                    event.params["pose_range"] = {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    }

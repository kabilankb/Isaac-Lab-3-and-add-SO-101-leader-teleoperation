"""Typed robot specification loaded from ``configs/robot/<name>.yaml``.

A :class:`RobotSpec` is the single structured source of a robot's hardware facts —
gripper limits, arm base poses, camera calibration, table geometry, finger topology, and
controller gain sets. It is intentionally **IsaacLab-free** so it can be imported anywhere
(including before the simulator launches); the IsaacLab ``ArticulationCfg`` is built from it
in :mod:`yamlab.robot.yam`.

Use :func:`get_robot` to obtain the spec for a named robot, e.g. ``get_robot("yam")``.
"""

import os
from pathlib import Path

import yaml

from yamlab.utils.lab3 import quat_wxyz_to_xyzw

_ROBOT_CFG_DIR = os.path.join(Path(__file__).resolve().parent.parent, "configs", "robot")

# A finger contact sensor's scene attribute is "<arm_name>_<finger_key>" + this suffix
# (e.g. "left_arm_lf_contact"). The grasp detector keys off the same convention.
_CONTACT_SENSOR_SUFFIX = "_contact"


class RobotSpec:
    """Structured, read-only view over a robot's ``configs/robot/<name>.yaml``.

    Property accessors return the parsed sub-dicts as-is; the typed accessors below them
    return calibrated values in the shapes the rest of the codebase expects (tuples for
    poses, floats for joint positions). ``raw`` is the full parsed document — an escape
    hatch for fields that do not yet have a dedicated accessor.
    """

    def __init__(self, name: str, data: dict):
        """Wrap a parsed robot-calibration document.

        Args:
            name (str): Robot name (the YAML stem under ``configs/robot/``).
            data (dict): The parsed ``configs/robot/<name>.yaml`` document.
        """
        self.name = name
        self.raw = data

    # ---- gripper -------------------------------------------------------
    @property
    def gripper(self) -> dict:
        """Raw gripper block.

        Returns:
            dict: Finger open/closed joint positions (``left_finger_open_pos`` etc.).
        """
        return self.raw["gripper"]

    def finger_open(self, side: str) -> float:
        """Open joint position of one finger.

        Args:
            side (str): ``"left"`` or ``"right"``.

        Returns:
            float: The finger's fully-open joint position (radians).
        """
        return self.raw["gripper"][f"{side}_finger_open_pos"]

    def finger_closed(self, side: str) -> float:
        """Closed joint position of one finger.

        Args:
            side (str): ``"left"`` or ``"right"``.

        Returns:
            float: The finger's fully-closed joint position (radians).
        """
        return self.raw["gripper"][f"{side}_finger_closed_pos"]

    @property
    def gripper_delta_per_step(self) -> float:
        """Per-step gripper travel for continuous open/close control.

        Returns:
            float: One twentieth of the full open travel (radians), so a held button
                fully opens the gripper over ~20 control steps.
        """
        return abs(self.finger_open("left")) / 20

    # ---- arms ----------------------------------------------------------
    @property
    def arms(self) -> dict:
        """Raw per-arm block.

        Returns:
            dict: Keyed ``"left"``/``"right"``; each has ``position``, ``quaternion``
                (wxyz), and ``prim_root``.
        """
        return self.raw["arms"]

    @property
    def arm_names(self) -> list:
        """Runtime arm names, in left-then-right order.

        Returns:
            list[str]: e.g. ``["left_arm", "right_arm"]`` — one per side in ``arms``.
        """
        return [f"{side}_arm" for side in self.raw["arms"]]

    def arm_position(self, side: str) -> tuple:
        """World base position of one arm.

        Args:
            side (str): ``"left"`` or ``"right"``.

        Returns:
            tuple[float, float, float]: ``(x, y, z)`` in meters, world frame.
        """
        return tuple(self.raw["arms"][side]["position"])

    def arm_quaternion(self, side: str) -> tuple:
        """World base orientation of one arm.

        Args:
            side (str): ``"left"`` or ``"right"``.

        Returns:
            tuple[float, float, float, float]: Quaternion ``(x, y, z, w)`` (Isaac Lab 3 order;
            ``yam.yaml`` stores it as ``(w, x, y, z)``).
        """
        return quat_wxyz_to_xyzw(tuple(self.raw["arms"][side]["quaternion"]))

    def arm_prim_root(self, arm_name: str) -> str:
        """Scene prim-root name for an arm instance.

        Args:
            arm_name (str): Runtime arm name (``"left_arm"``/``"right_arm"``).

        Returns:
            str: The prim under ``{ENV}/`` for that arm (e.g. ``"LeftArm"``), as placed
                by :class:`~yamlab.envs.yam_bimanual_scene.YamBimanualSceneCfg`.
        """
        return self.raw["arms"][arm_name.replace("_arm", "")]["prim_root"]

    # ---- cameras -------------------------------------------------------
    @property
    def cameras(self) -> dict:
        """Raw cameras block.

        Returns:
            dict: Per-camera ``position`` / ``quaternion_opengl`` / ``intrinsics``,
                plus the shared ``intrinsic_resolution``.
        """
        return self.raw["cameras"]

    @property
    def camera_names(self) -> list:
        """Camera names, in declaration order.

        Returns:
            list[str]: The keys under ``cameras:`` excluding ``intrinsic_resolution``.
        """
        return [k for k in self.raw["cameras"] if k != "intrinsic_resolution"]

    def camera_prim_path(self, name: str) -> str:
        """Scene prim path (mount) of a camera.

        Args:
            name (str): A camera name from :attr:`camera_names`.

        Returns:
            str: Its prim path; the parent prim decides world-fixed vs arm-mounted.
        """
        return self.raw["cameras"][name]["prim_path"]

    def camera_lerobot_key(self, name: str) -> str:
        """LeRobot dataset RGB key for a camera (``observation.images.<key>``).

        Args:
            name (str): A camera name from :attr:`camera_names`.

        Returns:
            str: The recorded RGB key suffix (e.g. ``"top_rgb"``).
        """
        return self.raw["cameras"][name]["lerobot_key"]

    def camera_position(self, name: str) -> tuple:
        """Position of a camera.

        Args:
            name (str): ``"top"``, ``"left_wrist"``, or ``"right_wrist"``.

        Returns:
            tuple[float, float, float]: ``(x, y, z)`` in meters.
        """
        return tuple(self.raw["cameras"][name]["position"])

    def camera_quaternion_opengl(self, name: str) -> tuple:
        """Orientation of a camera in the OpenGL convention.

        Args:
            name (str): ``"top"``, ``"left_wrist"``, or ``"right_wrist"``.

        Returns:
            tuple[float, float, float, float]: Quaternion ``(x, y, z, w)`` (Isaac Lab 3 order;
            ``yam.yaml`` stores it as ``(w, x, y, z)``).
        """
        return quat_wxyz_to_xyzw(tuple(self.raw["cameras"][name]["quaternion_opengl"]))

    def camera_intrinsics(self, name: str) -> dict:
        """Pinhole intrinsics of a camera.

        Args:
            name (str): ``"top"``, ``"left_wrist"``, or ``"right_wrist"``.

        Returns:
            dict: ``{"fx", "fy", "cx", "cy"}`` in pixels.
        """
        return dict(self.raw["cameras"][name]["intrinsics"])

    @property
    def intrinsic_resolution(self) -> tuple:
        """Resolution the camera intrinsics were calibrated at.

        Returns:
            tuple[int, int]: ``(width, height)`` in pixels. Decoupled from the render
                resolution so a render override preserves the field of view.
        """
        return tuple(self.raw["cameras"]["intrinsic_resolution"])

    # ---- table ---------------------------------------------------------
    @property
    def table(self) -> dict:
        """Raw table block.

        Returns:
            dict: Table top-surface ``position`` and corner extents (world frame).
        """
        return self.raw["table"]

    @property
    def table_position(self) -> tuple:
        """World position of the table top surface.

        Returns:
            tuple[float, float, float]: ``(x, y, z)`` in meters.
        """
        return tuple(self.raw["table"]["position"])

    # ---- fingers / grasp geometry --------------------------------------
    @property
    def finger_keypoints(self) -> dict:
        """Finger-pad keypoints per side.

        Returns:
            dict: ``{"left": [tip, base1, base2], "right": [...]}``; each keypoint is a
                3-list in the finger-link frame.
        """
        return self.raw["finger_keypoints"]

    @property
    def fingers(self) -> dict:
        """Per-finger contact topology and pad geometry.

        Returns:
            dict: ``{finger_key: {"path", "link", "keypoints"}}`` keyed by finger key
                (``"lf"``/``"rf"``). ``path`` is the finger link's prim subpath within an
                arm articulation, ``link`` its last segment (the link name), and
                ``keypoints`` the ``[tip, base1, base2]`` pad points (3-lists) in the
                finger-link frame.
        """
        return {
            key: {
                "path": v["path"],
                "link": v["path"].split("/")[-1],
                "keypoints": [list(p) for p in v["keypoints"]],
            }
            for key, v in self.raw["fingers"].items()
        }

    def contact_sensor_name(self, arm_name: str, finger_key: str) -> str:
        """Scene attribute name for one finger's contact sensor.

        Args:
            arm_name (str): Runtime arm name (``"left_arm"``/``"right_arm"``).
            finger_key (str): Finger key (``"lf"``/``"rf"``).

        Returns:
            str: Scene attribute / sensor key, e.g. ``"left_arm_lf_contact"``.
        """
        return f"{arm_name}_{finger_key}{_CONTACT_SENSOR_SUFFIX}"

    def contact_sensor_topology(self) -> dict:
        """Finger contact-sensor layout for every arm.

        Combines the arm prim roots with the finger topology. Consumed by the contact-sensor
        setup (``envs/tasks/yam_bimanual_env_cfg.py``) and the grasp detector (``utils/grasp.py``).

        Returns:
            dict: ``{arm_name: [(sensor_name, prim_subpath), ...]}`` where ``sensor_name``
                is :meth:`contact_sensor_name` and ``prim_subpath`` is
                ``"<prim_root>/<finger_path>"`` (e.g. ``"LeftArm/arm/left_finger"``).
        """
        fingers = self.raw["fingers"]
        return {
            arm_name: [
                (self.contact_sensor_name(arm_name, fkey),
                 f"{self.arm_prim_root(arm_name)}/{f['path']}")
                for fkey, f in fingers.items()
            ]
            for arm_name in self.arm_names
        }

    # ---- grasp detection -----------------------------------------------
    @property
    def grasp_detection(self) -> dict:
        """Grasp-detection thresholds (a gripper property).

        Returns:
            dict: ``{normal_force_thresh, check_pad, pad_min_frac, pad_max_frac}`` (any
                missing key falls back to the detector default).
        """
        return self.raw.get("grasp", {})

    # ---- controller ----------------------------------------------------
    @property
    def controller(self) -> dict:
        """Raw controller block.

        Returns:
            dict: ``{"default", "base": {...}, "high_pd": {...}}``.
        """
        return self.raw["controller"]

    @property
    def default_controller(self) -> str:
        """The gain set the scene builds with by default.

        Returns:
            str: ``"high_pd"`` or ``"base"`` (defaults to ``"high_pd"`` if unset).
        """
        return self.raw["controller"].get("default", "high_pd")

    def controller_gains(self, which: str | None = None) -> dict:
        """Per-actuator-group PD gains for one gain set.

        Args:
            which (str | None): ``"high_pd"`` or ``"base"``; ``None`` uses
                :attr:`default_controller`.

        Returns:
            dict: ``{actuator_group: {"stiffness", "damping"}}``.
        """
        return self.raw["controller"][which or self.default_controller]


def get_robot(name: str = "yam") -> RobotSpec:
    """Load the :class:`RobotSpec` for a named robot from ``configs/robot/<name>.yaml``.

    Args:
        name (str): Robot name (the YAML stem under ``configs/robot/``).

    Returns:
        RobotSpec: The parsed, structured spec.
    """
    path = os.path.join(_ROBOT_CFG_DIR, f"{name}.yaml")
    with open(path, "r") as f:
        return RobotSpec(name, yaml.safe_load(f))

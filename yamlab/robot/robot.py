"""Runtime model of a manipulation robot, bound to a live scene.

A :class:`Robot` holds a set of arms (one per ``spec.arm_names``); each :class:`Arm` has an
:class:`EndEffector`. ``Robot`` and ``Arm`` are end-effector-agnostic structure: the
end-effector type is the ``END_EFFECTOR_CLS`` / ``ARM_CLS`` class attribute, so a robot with a
different end-effector (e.g. a multi-finger dexterous hand) subclasses ``Arm``/``Robot`` and
supplies its own end-effector class. The :class:`Gripper` (with :class:`Finger`) provided here
is one such end-effector — a **parallel jaw**: two opposing fingers, grasp = both fingers in
pad contact. Grasp detection is a capability of these parts — a finger reports its contact
force and contact-pad position against a target object, the gripper reports a grasp when both
of its fingers hold the target, and the robot aggregates the per-arm results:

    robot = Robot(scene, spec, grasp_target="obj_0", grasp_kwargs={...})
    robot.is_grasping()                                   # {arm_name: bool tensor}, default target
    robot.arm("left_arm").is_grasping("obj_1")            # one arm, another object
    robot.arm("left_arm").end_effector.is_grasping()      # one end-effector

The classes are generic: the finger links, contact-sensor names, and pad keypoints each part
needs are read from the :class:`~yamlab.robot.spec.RobotSpec` passed in, so a robot is
described by its YAML rather than by code here. They read the live scene (contact sensors +
arm poses) and nothing else from the environment.
:class:`~yamlab.robot.bimanual_robot.BimanualRobot` specializes :class:`Robot` to the
two-arm case used by the tasks (see also :class:`yamlab.robot.yam.YamRobot`).

A grasp is read per finger from its contact sensor, combining up to two checks (both fingers
of a gripper must pass every enabled check):

  1. Contact force (mandatory): the finger's contact force on the target object exceeds
     ``normal_force_thresh``. Reads ``force_matrix_w``, correct on both CPU and GPU pipelines.
  2. Contact point on the pad (optional, ``check_pad``): the contact lies on the finger pad
     rather than the tip/edge, localized along the finger's tip->base axis. Rejects
     fingertip-only pokes. Reads ``contact_pos_w`` (also CPU/GPU-safe).
"""

import torch

from isaaclab.utils.math import quat_apply_inverse


def _filter_index_for_target(filter_exprs, target_object: str) -> int:
    """Find the index of ``target_object`` along a contact sensor's filtered-object axis.

    Filter paths are ``"{ENV}/<object>/<body_link>"``, or ``"{ENV}/<object>"`` when the rigid
    body is the asset root, so the object name is the last or second-to-last segment.

    Args:
        filter_exprs (list[str]): The sensor's ``filter_prim_paths_expr`` entries.
        target_object (str): Scene object name to locate (e.g. ``"obj_0"``).

    Returns:
        int: Index into the filtered-object axis, or ``-1`` if no filter targets the object
            (callers treat ``-1`` as "this finger has no contact with the target").
    """
    for i, expr in enumerate(filter_exprs):
        parts = expr.rstrip("/").split("/")
        if parts and (parts[-1] == target_object or (len(parts) >= 2 and parts[-2] == target_object)):
            return i
    return -1


class Finger:
    """One gripper finger and its contact sensor.

    Owns the finger's geometry (link name, pad keypoints) and the binding to its contact
    sensor, and computes the per-finger contact force and contact-pad position against a
    given target object.
    """

    def __init__(self, scene, spec, arm_name: str, finger_key: str):
        """Bind a finger to its contact sensor.

        Args:
            scene (InteractiveScene): The live scene (holds the contact sensors and arms).
            spec (RobotSpec): The robot's hardware spec (finger geometry, sensor naming).
            arm_name (str): Runtime arm name (``"left_arm"``/``"right_arm"``).
            finger_key (str): Finger key (``"lf"``/``"rf"``) into ``spec.fingers``.
        """
        finger_spec = spec.fingers[finger_key]
        self.scene = scene
        self.arm_name = arm_name
        self.finger_key = finger_key
        self.link = finger_spec["link"]
        self.sensor_name = spec.contact_sensor_name(arm_name, finger_key)
        # Keypoints in the local finger-link frame: [tip, base1, base2]. The tip and the
        # base-corner midpoint define the finger's long axis, along which a contact's
        # fractional position is measured (0.0 at tip, 1.0 at base).
        self._keypoints = torch.tensor(finger_spec["keypoints"], dtype=torch.float32)
        try:
            self._sensor = scene[self.sensor_name]
        except KeyError:
            self._sensor = None
        self._filter_idx_by_target = {}   # target object -> M-axis index (-1 if not filtered)
        self._axis_by_device = {}         # device -> (tip, axis_unit, axis_len)

    @property
    def has_contact_sensor(self) -> bool:
        """Whether this finger's contact sensor exists in the scene.

        Returns:
            bool: ``True`` if the sensor was created (its arm is in the task's grasp.detect).
        """
        return self._sensor is not None

    def _filter_index(self, target_object: str) -> int:
        """Resolve (and cache) the target object's index on the sensor's filtered-object axis.

        Args:
            target_object (str): Scene-object name to locate.

        Returns:
            int: Index into the ``force_matrix_w`` / ``contact_pos_w`` M dimension, or ``-1``
                if this sensor does not filter the target object.
        """
        if target_object not in self._filter_idx_by_target:
            exprs = list(getattr(getattr(self._sensor, "cfg", None), "filter_prim_paths_expr", []) or [])
            self._filter_idx_by_target[target_object] = _filter_index_for_target(exprs, target_object)
        return self._filter_idx_by_target[target_object]

    def _tip_base_axis(self, device):
        """Return the finger's tip->base axis, cached per device.

        Args:
            device (torch.device): Device to place the (constant) keypoint tensors on.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(tip, axis_unit, axis_len)`` —
                tip keypoint ``(3,)``, unit tip->base axis ``(3,)``, axis length (scalar).
        """
        cached = self._axis_by_device.get(str(device))
        if cached is not None:
            return cached
        keypoints = self._keypoints.to(device)
        tip = keypoints[0]
        axis = 0.5 * (keypoints[1] + keypoints[2]) - tip
        axis_len = torch.norm(axis)
        result = (tip, axis / axis_len, axis_len)
        self._axis_by_device[str(device)] = result
        return result

    def contact_force(self, target_object: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Contact-force magnitude (N) on the target object, per environment.

        Args:
            target_object (str): Scene-object name to measure contact force against.
            env_ids (torch.Tensor): Environment indices to check, shape ``(n,)``.

        Returns:
            torch.Tensor: Force magnitude (N) per env, float, shape ``(n,)``; all zeros if
                force data is unavailable or this finger does not filter the target.
        """
        device = self.scene.device
        filter_idx = self._filter_index(target_object)
        if filter_idx < 0:
            return torch.zeros(len(env_ids), device=device)
        force_matrix = self._sensor.data.force_matrix_w                  # (N, B, M, 3)
        if force_matrix is None or force_matrix.shape[0] == 0 or force_matrix.shape[2] <= filter_idx:
            return torch.zeros(len(env_ids), device=device)
        force_vec = force_matrix[env_ids, :, filter_idx, :].sum(dim=1)   # (n, 3)
        return torch.norm(force_vec, dim=1)

    def is_touching(self, target_object: str, normal_force_thresh: float, env_ids: torch.Tensor) -> torch.Tensor:
        """Contact-force gate: force magnitude on the target object >= threshold.

        Args:
            target_object (str): Scene-object name to test contact against.
            normal_force_thresh (float): Minimum contact force (N) to count as touching.
            env_ids (torch.Tensor): Environment indices to check, shape ``(n,)``.

        Returns:
            torch.Tensor: Bool, shape ``(n,)``, ``True`` where the finger touches the target.
        """
        return self.contact_force(target_object, env_ids) >= normal_force_thresh

    def contact_pad_fraction(self, target_object: str, env_ids: torch.Tensor):
        """Localize the finger's contact point along its tip->base axis.

        Reads the world-frame contact point, transforms it into the finger frame, and projects
        it onto the tip->base axis to a fractional position (0.0 at tip, 1.0 at base).

        Args:
            target_object (str): Scene-object name whose contact point to localize.
            env_ids (torch.Tensor): Environment indices to evaluate, shape ``(n,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(fraction, valid)`` — fractional position
                ``(n,)`` (NaN where there is no contact) and a bool ``(n,)`` marking finite
                contacts.
        """
        device = self.scene.device
        n = len(env_ids)
        fraction = torch.full((n,), float("nan"), device=device)
        valid = torch.zeros(n, dtype=torch.bool, device=device)

        filter_idx = self._filter_index(target_object)
        if filter_idx < 0:
            return fraction, valid
        contact_pos = self._sensor.data.contact_pos_w                    # (N, B, M, 3); NaN if none
        if contact_pos is None:
            return fraction, valid

        contact_world = contact_pos[env_ids, 0, filter_idx, :]           # (n, 3)
        valid = ~torch.isnan(contact_world).any(dim=1)

        arm = self.scene[self.arm_name]
        body_idx = arm.data.body_names.index(self.link)
        finger_pose = arm.data.body_link_pose_w[env_ids, body_idx, :]    # (n, 7) [pos, quat wxyz]
        contact_local = quat_apply_inverse(finger_pose[:, 3:], contact_world - finger_pose[:, :3])
        tip, axis_unit, axis_len = self._tip_base_axis(device)
        fraction = ((contact_local - tip) * axis_unit).sum(dim=1) / axis_len   # 0 at tip, 1 at base
        return fraction, valid

    def is_contact_on_pad(self, target_object: str, env_ids: torch.Tensor, evaluate_mask: torch.Tensor,
                          pad_min_frac: float, pad_max_frac: float) -> torch.Tensor:
        """Whether the finger's contact lies on the pad band, for the flagged envs.

        Only the env rows flagged in ``evaluate_mask`` are computed (the coordinate transform is
        skipped for envs that already failed the force gate).

        Args:
            target_object (str): Scene-object name whose contact to test.
            env_ids (torch.Tensor): Environment indices, shape ``(n,)``.
            evaluate_mask (torch.Tensor): Bool, shape ``(n,)``; only ``True`` rows are computed.
            pad_min_frac (float): Lower bound of the valid pad band (0 = tip, 1 = base).
            pad_max_frac (float): Upper bound of the valid pad band.

        Returns:
            torch.Tensor: Bool, shape ``(n,)``; ``True`` where the contact lies on the pad band,
                ``False`` wherever ``evaluate_mask`` is ``False`` or there is no valid contact.
        """
        device = self.scene.device
        on_pad = torch.zeros(len(env_ids), dtype=torch.bool, device=device)
        if not torch.any(evaluate_mask):
            return on_pad
        rows = torch.nonzero(evaluate_mask, as_tuple=False).squeeze(-1)
        fraction, valid = self.contact_pad_fraction(target_object, env_ids[rows])
        on_pad[rows] = valid & (fraction >= pad_min_frac) & (fraction <= pad_max_frac)
        return on_pad


class EndEffector:
    """Base class for an arm's end-effector — the part that grasps.

    Concrete end-effectors (the parallel-jaw :class:`Gripper`, or a future dexterous hand)
    implement :meth:`is_grasping` and report :attr:`has_contact_sensors`. They are constructed
    as ``EndEffector(scene, spec, arm_name, grasp_cfg, default_target)``.
    """

    def is_grasping(self, target_object: str = None, env_ids=None, normal_force_thresh: float = None):
        """Whether the end-effector is grasping a target object, per environment.

        Args:
            target_object (str | None): Scene-object name; ``None`` uses ``default_target``.
            env_ids (None | torch.Tensor): Environment indices, shape ``(n,)``; ``None`` = all.
            normal_force_thresh (None | float): Override the contact-force threshold (N).

        Returns:
            torch.Tensor: Bool tensor, shape ``(num_envs,)`` or ``(n,)``.
        """
        raise NotImplementedError

    @property
    def has_contact_sensors(self) -> bool:
        """Whether this end-effector is wired for grasp detection in the scene.

        Returns:
            bool: ``True`` if the end-effector's contact sensors exist (its arm is in the
                task's grasp.detect). An end-effector without them never reports a grasp.
        """
        raise NotImplementedError


class Gripper(EndEffector):
    """One arm's parallel-jaw gripper: two :class:`Finger` objects and a grasp check.

    A :class:`Gripper` is one kind of :class:`EndEffector`.
    Reports a grasp when both fingers touch the target object with sufficient force and (when
    enabled) on the pad. Grasp parameters (force threshold, pad band) are fixed at construction;
    the target object is an argument, defaulting to the robot's ``default_target``.
    """

    def __init__(self, scene, spec, arm_name: str, grasp_cfg: dict, default_target: str):
        """Build the gripper's fingers.

        Args:
            scene (InteractiveScene): The live scene.
            spec (RobotSpec): The robot's hardware spec.
            arm_name (str): Runtime arm name (``"left_arm"``/``"right_arm"``).
            grasp_cfg (dict): ``{"normal_force_thresh", "check_pad", "pad_min_frac",
                "pad_max_frac"}``.
            default_target (str): Scene-object name used when ``is_grasping`` is called without
                an explicit target.
        """
        self.scene = scene
        self.arm_name = arm_name
        self.default_target = default_target
        self._cfg = grasp_cfg
        self.fingers = [Finger(scene, spec, arm_name, fkey) for fkey in spec.fingers]

        equipped = [f for f in self.fingers if f.has_contact_sensor]
        assert len(equipped) in (0, len(self.fingers)), (
            f"Gripper '{arm_name}' has a partial set of finger contact sensors "
            f"{[f.sensor_name for f in equipped]}; configure_contact_sensors must create both "
            f"fingers or neither."
        )
        self._has_contact_sensors = bool(equipped) and len(equipped) == len(self.fingers)

    @property
    def has_contact_sensors(self) -> bool:
        """Whether this gripper's finger contact sensors exist in the scene.

        Returns:
            bool: ``True`` if both fingers have sensors (the arm is in the task's grasp.detect).
                A gripper without sensors never reports a grasp.
        """
        return self._has_contact_sensors

    def is_grasping(self, target_object: str = None, env_ids: torch.Tensor = None,
                    normal_force_thresh: float = None) -> torch.Tensor:
        """Whether this gripper is grasping a target object, per environment.

        Both fingers must pass every enabled check, evaluated in order with short-circuiting:
        contact force (always), then the contact-pad check (only if ``check_pad``).

        Args:
            target_object (str | None): Scene-object name; ``None`` uses ``default_target``.
            env_ids (None | torch.Tensor): Environment indices to check, shape ``(n,)``;
                ``None`` means all environments.
            normal_force_thresh (None | float): Override the per-finger force threshold (N);
                ``None`` uses the configured value.

        Returns:
            torch.Tensor: Bool, shape ``(num_envs,)`` or ``(n,)``; ``True`` where the gripper
                grasps the target. All-``False`` if the gripper has no contact sensors.
        """
        target_object = target_object or self.default_target
        device = self.scene.device
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=device)
        if not self._has_contact_sensors:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=device)

        thresh = self._cfg["normal_force_thresh"] if normal_force_thresh is None else normal_force_thresh
        try:
            # (1) Contact force on the target object -- mandatory. Every finger must touch.
            grasping = torch.ones(len(env_ids), dtype=torch.bool, device=device)
            for finger in self.fingers:
                grasping &= finger.is_touching(target_object, thresh, env_ids)
            if not torch.any(grasping):
                return grasping

            # (2) Contact point on the finger pad -- optional. Localize the pad only for the
            # envs that have passed so far (evaluate_mask=grasping).
            if self._cfg["check_pad"]:
                for finger in self.fingers:
                    grasping = grasping & finger.is_contact_on_pad(
                        target_object, env_ids, grasping,
                        self._cfg["pad_min_frac"], self._cfg["pad_max_frac"])
                if not torch.any(grasping):
                    return grasping
            return grasping
        except Exception as e:
            print(f"[grasp] error for {self.arm_name} on '{target_object}': {e}")
            return torch.zeros(len(env_ids), dtype=torch.bool, device=device)


class Arm:
    """One arm and its end-effector.

    The end-effector type is :attr:`END_EFFECTOR_CLS` (default: the parallel-jaw
    :class:`Gripper`); a robot with a different end-effector subclasses ``Arm`` and overrides it.
    """

    END_EFFECTOR_CLS = Gripper

    def __init__(self, scene, spec, arm_name: str, grasp_cfg: dict, default_target: str):
        """Build the arm's end-effector.

        Args:
            scene (InteractiveScene): The live scene.
            spec (RobotSpec): The robot's hardware spec.
            arm_name (str): Runtime arm name (``"left_arm"``/``"right_arm"``).
            grasp_cfg (dict): Grasp parameters (see :class:`Gripper`).
            default_target (str): Scene-object name used when ``is_grasping`` has no explicit target.
        """
        self.name = arm_name
        self.end_effector = self.END_EFFECTOR_CLS(scene, spec, arm_name, grasp_cfg, default_target)

    def is_grasping(self, target_object: str = None, env_ids: torch.Tensor = None,
                    normal_force_thresh: float = None) -> torch.Tensor:
        """Whether this arm is grasping a target object (delegates to its end-effector).

        Args:
            target_object (str | None): Scene-object name; ``None`` uses the default target.
            env_ids (None | torch.Tensor): Environment indices, shape ``(n,)``; ``None`` = all.
            normal_force_thresh (None | float): Override the contact-force threshold (N).

        Returns:
            torch.Tensor: Bool tensor, shape ``(num_envs,)`` or ``(n,)``.
        """
        return self.end_effector.is_grasping(target_object, env_ids, normal_force_thresh)


def _grasp_cfg_from_spec(spec) -> dict:
    """Build the per-end-effector grasp config from the robot spec, filling defaults.

    Args:
        spec (RobotSpec): The robot spec; its ``grasp_detection`` block supplies the
            thresholds (grasp detection is a robot property).

    Returns:
        dict: ``{"normal_force_thresh", "check_pad", "pad_min_frac", "pad_max_frac"}``.
    """
    grasp = spec.grasp_detection
    return {
        "normal_force_thresh": grasp.get("normal_force_thresh", 0.1),
        "check_pad": grasp.get("check_pad", True),
        "pad_min_frac": grasp.get("pad_min_frac", 0.0),
        "pad_max_frac": grasp.get("pad_max_frac", 1.0),
    }


class Robot:
    """Runtime manipulation robot bound to a scene — a set of arms with grasp queries.

    Generic over the number of arms (taken from ``spec.arm_names``) and the end-effector type
    (each arm's :attr:`Arm.END_EFFECTOR_CLS`); the arm type is :attr:`ARM_CLS`.
    :class:`BimanualRobot` specializes it to the two-arm case used by the tasks.
    """

    ARM_CLS = Arm

    def __init__(self, scene, spec, grasp_target: str):
        """Build the robot's arms.

        Args:
            scene (InteractiveScene): The live scene (holds the arms + contact sensors).
            spec (RobotSpec): The robot's hardware spec (arm names, finger geometry, sensors,
                grasp-detection thresholds).
            grasp_target (str): Scene-object name the arms grasp by default (e.g. ``"obj_0"``);
                the target used when ``is_grasping`` is called without one.

        Raises:
            AssertionError: If no arm has finger contact sensors (none were created by
                ``configure_contact_sensors`` -- grasp detection cannot run).
        """
        self.scene = scene
        self.spec = spec
        self.grasp_target = grasp_target
        grasp_cfg = _grasp_cfg_from_spec(spec)
        self.arms = {name: self.ARM_CLS(scene, spec, name, grasp_cfg, grasp_target) for name in spec.arm_names}

        equipped = sorted(name for name, arm in self.arms.items() if arm.end_effector.has_contact_sensors)
        assert equipped, (
            "Grasp detection requires finger contact sensors on at least one arm, but none are "
            "in the scene. Add them via YamBimanualEnvCfg.configure_contact_sensors(detect_map) "
            "(see envs/tasks/yam_bimanual_env_cfg.py) -- the task YAML's grasp.detect block declares "
            "which arms detect which objects."
        )
        print(f"[robot] grasp model ready; default target='{grasp_target}'; arms equipped: {equipped}")

    def arm(self, arm_name: str) -> Arm:
        """Return the :class:`Arm` for one side.

        Args:
            arm_name (str): An entry of ``spec.arm_names`` (e.g. ``"left_arm"``).

        Returns:
            Arm: The requested arm.
        """
        return self.arms[arm_name]

    def is_grasping(self, target_object: str = None, env_ids: torch.Tensor = None,
                    normal_force_thresh: float = None) -> dict:
        """Whether each arm is grasping a target object, per environment.

        Args:
            target_object (str | None): Scene-object name; ``None`` uses ``grasp_target``.
            env_ids (None | torch.Tensor): Environment indices, shape ``(n,)``; ``None`` = all.
            normal_force_thresh (None | float): Override the contact-force threshold (N).

        Returns:
            dict[str, torch.Tensor]: Per-arm grasp result, keyed by arm name.
        """
        return {name: arm.is_grasping(target_object, env_ids, normal_force_thresh)
                for name, arm in self.arms.items()}


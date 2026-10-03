"""Teleoperation grasp-ray overlay, drawn from the YAM finger geometry.

:class:`GraspRayVisualizer` is a teleop-only on-screen hint: it draws a green/red beam
between each gripper's fingertips, green when closing the gripper would likely grasp an
object. (Runtime grasp *detection* lives in :mod:`yamlab.robot.robot`.)
"""
import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_from_angle_axis

from yamlab.robot.yam import ROBOT


class GraspRayVisualizer:
    """Teleop-only on-screen grasp hint: a translucent green/red beam between the fingertips.

    Per gripper, casts a sparse set of inter-finger rays (every left-finger keypoint to every
    right-finger keypoint) and draws one fingertip-to-fingertip beam, green if ANY ray hits a
    target object, else red. Sparse rays -> approximate hint, not a guarantee. Visual only (no
    collider; no effect on physics, actions, or recording). CPU-only PhysX raycast (teleop is CPU).
    """

    # Arm -> its two (finger link, finger key) pairs (rays span an arm's own jaw), derived from
    # the robot finger topology (ROBOT.fingers).
    _ARM_FINGERS = {
        arm: tuple((spec["link"], finger_key) for finger_key, spec in ROBOT.fingers.items())
        for arm in ROBOT.arm_names
    }
    _GRASPABLE_MARKER_IDX, _NON_GRASPABLE_MARKER_IDX = 0, 1   # marker prototype order: green (graspable), red (none)

    def __init__(self, env, target_objects, beam_radius: float = 0.006, opacity: float = 0.4):
        """Initialize the grasp-ray visualization and build the beam markers.

        Args:
            env (ManagerBasedRLEnv): Task environment (exposes ``scene[arm]`` with finger bodies).
            target_objects (list of str): Scene-object names counted as graspable (a ray hits when
                a collider's prim path contains one of these; the table/robot/ground are ignored).
            beam_radius (float): Beam cylinder radius in meters (visual only).
            opacity (float): Beam opacity, 0 (clear) .. 1 (opaque).
        """
        self.env = env
        self.device = env.device
        self.target_objects = [str(n) for n in (target_objects or [])]

        # Per-finger local keypoints [tip, base1, base2], keyed by finger key (lf/rf).
        self._kp = {finger_key: torch.tensor(spec["keypoints"], dtype=torch.float32, device=self.device)
                    for finger_key, spec in ROBOT.fingers.items()}
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._ray_pairs = [(i, j) for i in range(3) for j in range(3)]   # (0,0) = tip-to-tip

        from omni.physx import get_physx_scene_query_interface
        self._physx = get_physx_scene_query_interface()

        # Translucent emissive green/red rods (real USD prims, so they show in the wrist cam).
        self._markers = None
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            def _beam(diffuse, emissive):
                return sim_utils.CylinderCfg(
                    radius=beam_radius, height=1.0, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=diffuse, emissive_color=emissive,
                        opacity=opacity, roughness=1.0, metallic=0.0))

            self._markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/GraspBeam",
                markers={"hit": _beam((0.55, 0.95, 0.55), (0.25, 0.75, 0.25)),    # light green
                         "miss": _beam((0.90, 0.18, 0.12), (0.65, 0.08, 0.05))}))  # red
        except Exception as e:
            print(f"[GraspRayVisualizer] markers unavailable, overlay disabled: {e}")

    @property
    def enabled(self) -> bool:
        """Whether the overlay is active.

        Returns:
            bool: True if the beam markers were created successfully (e.g. debug-draw available).
        """
        return self._markers is not None

    def _finger_keypoints_world(self, arm, body_name, finger_key, env_id) -> torch.Tensor:
        """Compute one finger's keypoints in the world frame.

        Args:
            arm (Articulation): Arm articulation holding the finger body.
            body_name (str): Finger link name ("left_finger" / "right_finger").
            finger_key (str): Finger key ("lf"/"rf"), selecting the keypoint set.
            env_id (int): Environment index.

        Returns:
            torch.Tensor: Keypoints [tip, base1, base2] in the world frame, shape (3, 3).
        """
        idx = arm.data.body_names.index(body_name)
        pose = arm.data.body_link_pose_w.torch[env_id, idx, :]   # [pos(3), quat xyzw(4)]
        kp = self._kp[finger_key]
        return pose[:3].unsqueeze(0) + quat_apply(pose[3:].unsqueeze(0).expand(kp.shape[0], 4), kp)

    def _ray_hits_target(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Test whether a ray segment crosses any target object's collider.

        Args:
            start (np.ndarray): Ray start point in world frame, shape (3,).
            end (np.ndarray): Ray end point in world frame, shape (3,).

        Returns:
            bool: True if the segment intersects a collider whose prim path matches a
                target object (non-target hits, e.g. fingers/table, are ignored).
        """
        d = end - start
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return False
        state = {"hit": False}

        def _report(hit):
            if any(n in (getattr(hit, "collision", "") or "") for n in self.target_objects):
                state["hit"] = True
                return False   # stop
            return True        # ignore non-targets, keep going

        self._physx.raycast_all(origin=start.tolist(), dir=(d / dist).tolist(),
                                distance=dist, reportFn=_report)
        return state["hit"]

    def _z_to_dir_quat(self, d: torch.Tensor) -> torch.Tensor:
        """Compute the quaternion that rotates the cylinder's +Z axis onto a direction.

        Args:
            d (torch.Tensor): Unit direction, shape (3,).

        Returns:
            torch.Tensor: Quaternion (wxyz) rotating +Z onto d, shape (4,).
        """
        cos_a = torch.dot(self._z_axis, d).clamp(-1.0, 1.0)
        axis = torch.linalg.cross(self._z_axis, d)
        s = torch.linalg.norm(axis)
        if s < 1e-6:   # parallel / anti-parallel
            # xyzw: identity, or 180 deg about X when d points along -Z.
            return torch.tensor([0, 0, 0, 1.0] if cos_a > 0 else [1.0, 0, 0, 0], device=self.device)
        return quat_from_angle_axis(torch.atan2(s, cos_a).unsqueeze(0), (axis / s).unsqueeze(0))[0]

    def update(self, env_id: int = 0):
        """Recompute grasp state and redraw the beams.

        Driven by a render callback (once per rendered frame), so errors are reported once and
        then suppressed to avoid spamming or tearing down the render loop.

        Args:
            env_id (int): Environment index whose finger poses and objects to read (teleop uses 0).
        """
        if self._markers is None:
            return
        try:
            self._update(env_id)
        except Exception as e:
            if not getattr(self, "_warned", False):
                self._warned = True
                print(f"[GraspRayVisualizer] update() error (suppressed hereafter): {e}")

    def _update(self, env_id: int):
        """Cast the per-gripper rays and draw the tip-to-tip beams (the work behind update()).

        Args:
            env_id (int): Environment index to read finger poses and cast rays for.
        """
        translations, orientations, scales, indices = [], [], [], []
        for arm_name, ((b0, s0), (b1, s1)) in self._ARM_FINGERS.items():
            try:
                arm = self.env.scene[arm_name]
            except KeyError:
                continue
            kps0 = self._finger_keypoints_world(arm, b0, s0, env_id)   # (3, 3) [tip, b1, b2]
            kps1 = self._finger_keypoints_world(arm, b1, s1, env_id)
            kps0_np, kps1_np = kps0.detach().cpu().numpy(), kps1.detach().cpu().numpy()

            arm_hit = any(self._ray_hits_target(kps0_np[i], kps1_np[j]) for i, j in self._ray_pairs)

            # Draw only the tip-to-tip beam; color = OR over all detection rays.
            seg = kps1[0] - kps0[0]
            length = torch.linalg.norm(seg)
            if length < 1e-5:
                continue
            translations.append(0.5 * (kps0[0] + kps1[0]))
            orientations.append(self._z_to_dir_quat(seg / length))
            scales.append(torch.tensor([1.0, 1.0, float(length)], device=self.device))
            indices.append(self._GRASPABLE_MARKER_IDX if arm_hit else self._NON_GRASPABLE_MARKER_IDX)

        if not translations:
            self._markers.set_visibility(False)
            return
        self._markers.set_visibility(True)
        self._markers.visualize(translations=torch.stack(translations),
                                orientations=torch.stack(orientations),
                                scales=torch.stack(scales),
                                marker_indices=torch.tensor(indices, device=self.device))

    def clear(self):
        """Hide the beams."""
        if self._markers is not None:
            self._markers.set_visibility(False)

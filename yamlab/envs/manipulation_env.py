"""Generic, robot-agnostic base environment for manipulation tasks.

:class:`ManipulationEnv` collects the runtime behavior that does not depend on a particular
robot: the teleoperation pose schedule, the HDF5 / LeRobot recording APIs, and the abstract
task-state hooks. A robot-specific base environment (e.g.
:class:`~yamlab.envs.tasks.yam_bimanual_env.YamBimanualEnv`) subclasses it and adds the robot,
action layout, gripper handling, and scene setup.
"""

import math
import torch
from abc import abstractmethod
from typing import Dict, Any

from isaaclab.envs import ManagerBasedRLEnv

from yamlab.utils.recorders import ManualControlRecorderManager, LeRobotRecorderManager
from yamlab.utils.transforms import euler2quat


class ManipulationEnv(ManagerBasedRLEnv):
    """Robot-agnostic base environment: pose schedule, recording APIs, task-state hooks.

    Subclasses build the robot and scene in ``__init__`` and set the instance attributes
    these methods read (``recorder_manager``, ``lerobot_recorder``, ``_pose_schedule_*``,
    ``_recording_active``, ``_episode_count``).
    """

    def _apply_scheduled_poses_to_default_state(self):
        """Write the current scheduled poses into each object's default_root_state.

        Modifies default_root_state tensors so that the reset event uses the scheduled
        pose as the base (randomization is disabled for scheduled objects). Supports
        both absolute (pos/rot) and fractional (pos_fraction/rot_fraction) entries,
        where 0.0/1.0 map to the negative/positive end of the effective randomization
        range and 0.5 maps to the default (center) pose.
        """
        if self._pose_schedule_data is None:
            return

        obj_schedule = self._pose_schedule_data.get("obj_pose_schedule", {})
        current_idx = self._pose_schedule_index
        effective_ranges = self._pose_schedule_effective_ranges or {}
        init_states = self._pose_schedule_init_state or {}

        for obj_name, poses in obj_schedule.items():
            if current_idx >= len(poses):
                print(f"[WARNING] Pose schedule exhausted for '{obj_name}' (index {current_idx} >= {len(poses)})")
                continue

            pose = poses[current_idx]

            if obj_name not in self.scene.keys():
                print(f"[WARNING] Object '{obj_name}' not found in scene, skipping pose schedule")
                continue

            obj = self.scene[obj_name]

            # State layout: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel...]
            default_state = obj.data.default_root_state.clone()

            # Only x and y are applied; z is derived from asset height at reset time.
            if "pos" in pose:
                pos = pose["pos"]
                if len(pos) >= 2:
                    default_state[:, 0] = pos[0]
                    default_state[:, 1] = pos[1]
                else:
                    print(f"[WARNING] Position for '{obj_name}' has insufficient elements (need at least 2 for x, y). Got: {pos}")
            elif "pos_fraction" in pose:
                pos_fraction = pose["pos_fraction"]
                ranges = effective_ranges.get(obj_name, {})
                eff_x, eff_y = ranges.get("pos_range", (0.0, 0.0))
                init = init_states.get(obj_name, {})
                init_x = init.get("pos_x", default_state[0, 0].item())
                init_y = init.get("pos_y", default_state[0, 1].item())
                # Map [0, 1] → [-eff, +eff] offset from the default pose center.
                default_state[:, 0] = init_x + (2.0 * pos_fraction[0] - 1.0) * eff_x
                default_state[:, 1] = init_y + (2.0 * pos_fraction[1] - 1.0) * eff_y

            if "rot" in pose:
                rot = pose["rot"]
                if isinstance(rot, dict):
                    quat = euler2quat(rot["roll"], rot["pitch"], rot["yaw"])
                    quat = quat.to(self.device)
                else:
                    quat = torch.tensor(rot, device=self.device, dtype=torch.float32)
                default_state[:, 3] = quat[0]  # w
                default_state[:, 4] = quat[1]  # x
                default_state[:, 5] = quat[2]  # y
                default_state[:, 6] = quat[3]  # z
            elif "rot_fraction" in pose:
                rot_fraction = pose["rot_fraction"]
                ranges = effective_ranges.get(obj_name, {})
                orientation_range = ranges.get("orientation_range", 0.0)
                init = init_states.get(obj_name, {})
                default_yaw = init.get("default_yaw", 0.0)
                # Map [0, 1] → yaw offset within [-orientation_range, +orientation_range].
                yaw_offset = (2.0 * rot_fraction["yaw"] - 1.0) * orientation_range
                actual_yaw = default_yaw + yaw_offset
                quat = euler2quat(0.0, 0.0, math.degrees(actual_yaw))
                quat = quat.to(self.device)
                default_state[:, 3] = quat[0]  # w
                default_state[:, 4] = quat[1]  # x
                default_state[:, 5] = quat[2]  # y
                default_state[:, 6] = quat[3]  # z

            obj.data.default_root_state[:] = default_state

    def get_pose_schedule_info(self) -> Dict[str, Any]:
        """Return current pose schedule status.

        Returns:
            dict: Keys enabled (bool), current_index (int), total_poses (int),
                scheduled_objects (list[str]), is_exhausted (bool).
        """
        if self._pose_schedule_data is None:
            return {
                "enabled": False,
                "current_index": 0,
                "total_poses": 0,
                "scheduled_objects": [],
                "is_exhausted": False,
            }

        obj_schedule = self._pose_schedule_data.get("obj_pose_schedule", {})
        total_poses = min(len(poses) for poses in obj_schedule.values()) if obj_schedule else 0

        return {
            "enabled": True,
            "current_index": self._pose_schedule_index,
            "total_poses": total_poses,
            "scheduled_objects": list(obj_schedule.keys()),
            "is_exhausted": self._pose_schedule_index >= total_poses,
        }

    def advance_pose_schedule(self) -> bool:
        """Advance to the next pose in the schedule.

        Should be called after a successful trajectory save; the next reset() will
        use the updated pose.

        Returns:
            bool: True if advanced successfully, False if the schedule is exhausted or disabled.
        """
        if self._pose_schedule_data is None:
            return False

        obj_schedule = self._pose_schedule_data.get("obj_pose_schedule", {})
        if not obj_schedule:
            return False

        total_poses = min(len(poses) for poses in obj_schedule.values())
        if self._pose_schedule_index >= total_poses - 1:
            return False

        self._pose_schedule_index += 1
        self._apply_scheduled_poses_to_default_state()
        return True

    @abstractmethod
    def get_current_task_info(self) -> Dict[str, Any]:
        """Return a dictionary describing the current task state.

        Returns:
            dict: Task-specific state information.
        """
        pass

    @abstractmethod
    def reset_success_check(self) -> None:
        """Clear task-specific success flags so a new episode starts from a clean slate."""
        pass

    def configure_for_data_collection(self, output_dir: str = "./datasets", filename: str = "task_demos"):
        """Reconfigure the HDF5 recorder to write to a new output directory/filename.

        Args:
            output_dir (str): Target directory for the HDF5 dataset.
            filename (str): Base filename (without extension) for the HDF5 file.
        """
        if self.recorder_manager is not None:
            self.cfg.recorders.dataset_export_dir_path = output_dir
            self.cfg.recorders.dataset_filename = filename

            if hasattr(self, 'recorder_manager'):
                del self.recorder_manager

            self.recorder_manager = ManualControlRecorderManager(self.cfg.recorders, self)
            self.recorder_manager.flush_steps = 100
            self.recorder_manager.compression = 'lzf'

            print(f"[INFO] Data collection configured:")
            print(f"  - Output directory: {output_dir}")
            print(f"  - Filename: {filename}")
            print(f"  - Streaming recorder enabled")

    def configure_for_lerobot_recording(
        self,
        output_dir: str,
        repo_id: str = "local/dataset",
        fps: int = 30,
        task_name: str = None,
        features: dict = None,
        image_downsample_factor: int | None = None,
        base_width: int | None = None,
        base_height: int | None = None,
    ):
        """Configure the environment to write directly to a LeRobot v2 dataset.

        Writes parquet + video straight to the LeRobot format used for training (no
        intermediate HDF5 file). Used by MimicGen data generation and the replay scripts.

        Args:
            output_dir (str): Directory to save the LeRobot dataset.
            repo_id (str): Repository ID (e.g., "local/dataset_name").
            fps (int): Frames per second (should match simulation decimation rate).
            task_name (None or str): Task description string; defaults to config class name.
            features (None or dict): Feature definitions; None uses default bimanual features.
            image_downsample_factor (None or int): Recorded image resolution is the camera
                sensor resolution // this factor. None uses the value resolved at env creation.
            base_width (None or int): Camera sensor render width (= recorded width *
                downsample_factor). None uses the value resolved at env creation.
            base_height (None or int): Camera sensor render height. None uses the resolved value.
        """
        if task_name is None:
            task_name = self.cfg.__class__.__name__.replace("Cfg", "")

        # Default the recorded-image resolution to the values resolved at env creation
        # (cfg._camera_* / _image_downsample_factor, from --camera_* / rendering YAML), so a
        # caller that omits them records at the actual sensor resolution // factor rather than a
        # hardcoded 640x480 / x2 (which silently mis-sized 320x240 replays).
        if image_downsample_factor is None:
            image_downsample_factor = getattr(self.cfg, "_image_downsample_factor", 2)
        if base_width is None:
            base_width = getattr(self.cfg, "_camera_width", 640)
        if base_height is None:
            base_height = getattr(self.cfg, "_camera_height", 480)

        self.lerobot_recorder = LeRobotRecorderManager(
            repo_id=repo_id,
            root=output_dir,
            fps=fps,
            features=features,
            robot_type="bimanual_yam",
            task_name=task_name,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4,
            discard_first_n_frames=self._discard_first_n_frames,
            discard_last_n_frames=self._discard_last_n_frames,
            image_downsample_factor=image_downsample_factor,
            base_width=base_width,
            base_height=base_height,
        )

        print(f"[INFO] LeRobot recording configured:")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Repo ID: {repo_id}")
        print(f"  - FPS: {fps}")
        print(f"  - Task: {task_name}")
        print(f"  - Sensor resolution: {base_width}x{base_height}, downsample x{image_downsample_factor} "
              f"(recorded: {base_width // image_downsample_factor}x{base_height // image_downsample_factor})")

    def start_lerobot_recording(self) -> bool:
        """Start recording a new episode to the LeRobot dataset.

        Returns:
            bool: True on success, False if the recorder is not configured.
        """
        if not hasattr(self, 'lerobot_recorder') or self.lerobot_recorder is None:
            print("[WARNING] LeRobot recorder not configured. Call configure_for_lerobot_recording first.")
            return False

        self.lerobot_recorder.start_recording()
        self._recording_active = True
        return True

    def record_lerobot_frame(self, obs_dict: dict, action: torch.Tensor) -> None:
        """Record a single frame to the active LeRobot episode.

        Args:
            obs_dict (dict): Observation dictionary from the environment.
            action (torch.Tensor): float32, shape (num_envs, 14), action applied at this step.
        """
        if not hasattr(self, 'lerobot_recorder') or self.lerobot_recorder is None:
            return
        self.lerobot_recorder.record_frame(obs_dict, action)

    def save_lerobot_episode(self, encode_videos: bool = False) -> bool:
        """Save the current LeRobot episode to disk.

        Args:
            encode_videos (bool): If True, encode video files immediately after saving.

        Returns:
            bool: True if the episode was saved successfully.
        """
        if not hasattr(self, 'lerobot_recorder') or self.lerobot_recorder is None:
            print("[WARNING] LeRobot recorder not configured")
            return False

        success = self.lerobot_recorder.save_episode(encode_videos=encode_videos)
        if success:
            self._episode_count = self.lerobot_recorder.episode_count
        self._recording_active = False
        return success

    def discard_lerobot_episode(self) -> None:
        """Discard the current LeRobot episode without saving."""
        if hasattr(self, 'lerobot_recorder') and self.lerobot_recorder is not None:
            self.lerobot_recorder.discard_episode()
        self._recording_active = False

    def consolidate_lerobot_dataset(
        self,
        run_compute_stats: bool = False,
    ) -> None:
        """Consolidate the LeRobot dataset: encode videos and compute statistics.

        Should be called after all episodes have been recorded.

        Args:
            run_compute_stats (bool): Whether to compute dataset statistics.
        """
        if not hasattr(self, 'lerobot_recorder') or self.lerobot_recorder is None:
            print("[WARNING] LeRobot recorder not configured")
            return

        self.lerobot_recorder.consolidate(run_compute_stats=run_compute_stats)

    def close_lerobot_recorder(self) -> None:
        """Close the LeRobot recorder and release all resources."""
        if hasattr(self, 'lerobot_recorder') and self.lerobot_recorder is not None:
            self.lerobot_recorder.close()
            self.lerobot_recorder = None

    def start_recording(self) -> bool:
        """Begin a new HDF5 recording episode, clearing previous step data.

        Returns:
            bool: True on success, False if no recorder manager is available.
        """
        if self.recorder_manager is None:
            print("[WARNING] No recorder manager available")
            return False

        try:
            env_ids = list(range(self.num_envs))
            if hasattr(self.recorder_manager, '_episodes'):
                for env_id in env_ids:
                    if env_id in self.recorder_manager._episodes:
                        episode_data = self.recorder_manager._episodes[env_id]

                        initial_state = None
                        if "initial_state" in episode_data.data:
                            initial_state = episode_data.data["initial_state"]
                            print(f"[INFO] Preserving initial state for env {env_id}")

                        episode_data.data = dict()
                        episode_data._data = dict()

                        if initial_state is not None:
                            episode_data.data["initial_state"] = initial_state
                            print(f"[INFO] Restored initial state for env {env_id}")

                        if hasattr(self.recorder_manager, '_env_steps_record'):
                            self.recorder_manager._env_steps_record[env_id] = 0
            print("[INFO] Cleared step data, preserved initial state")
        except Exception as e:
            print(f"[WARNING] Failed to clear episode data: {e}")

        self._recording_active = True
        print("[INFO] Started recording new episode")
        return True

    def save_trajectory(self) -> bool:
        """Export the current HDF5 trajectory and reset for the next demo.

        Advances the pose schedule (if active) before resetting so the next episode
        uses the subsequent scheduled pose.

        Returns:
            bool: True on success, False if the recorder is unavailable or not recording.
        """
        if self.recorder_manager is None:
            print("[WARNING] No recorder manager available")
            return False

        if not self._recording_active:
            print("[WARNING] No active recording to save")
            return False

        env_ids = list(range(self.num_envs))
        self.recorder_manager.export_episodes_manual(env_ids)

        self._episode_count += 1
        print(f"[INFO] Trajectory saved (Episode {self._episode_count})")

        if self._pose_schedule_data is not None:
            self.advance_pose_schedule()

        self.reset(warm_up=False)
        self._recording_active = False
        print("[INFO] Environment reset for next demo")
        return True

    def is_recording(self) -> bool:
        """Return True if an HDF5 recording episode is currently active."""
        return self._recording_active

    def reset_task(self) -> bool:
        """Discard the current recording and reset without advancing the pose schedule.

        Unlike save_trajectory(), this keeps the current scheduled pose so the
        operator can retry the same configuration.

        Returns:
            bool: True after successful discard and reset.
        """
        if self._recording_active:
            try:
                env_ids = list(range(self.num_envs))
                if hasattr(self.recorder_manager, '_episodes'):
                    for env_id in env_ids:
                        if env_id in self.recorder_manager._episodes:
                            del self.recorder_manager._episodes[env_id]
                print("[INFO] Current recording discarded")
            except Exception as e:
                print(f"[WARNING] Failed to discard current recording: {e}")

        # Re-apply the current scheduled pose so the reset lands at the same position.
        if self._pose_schedule_data is not None:
            self._apply_scheduled_poses_to_default_state()

        self.reset(warm_up=False)
        self._recording_active = False
        print("[INFO] Task reset - recording stopped")
        return True

    def close_data_collector(self):
        """Close the data collector, discarding any in-progress recording.

        Does not export the current episode to avoid creating partial demos.
        """
        if self.recorder_manager is not None:
            try:
                if self._recording_active and hasattr(self.recorder_manager, '_episodes'):
                    env_ids = list(range(self.num_envs))
                    for env_id in env_ids:
                        if env_id in self.recorder_manager._episodes:
                            del self.recorder_manager._episodes[env_id]
                self._recording_active = False
            except Exception as e:
                print(f"[WARNING] Failed to discard in-progress recording on close: {e}")
        print("[INFO] Data collector closed (no export on close)")

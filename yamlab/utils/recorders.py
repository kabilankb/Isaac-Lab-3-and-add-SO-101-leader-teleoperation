"""HDF5 streaming recorders and the direct LeRobot v2 dataset writer."""
import enum
import copy
import json
import shutil
import h5py
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
import torch
import cv2

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData
from isaaclab.managers import RecorderManager, RecorderTerm, RecorderTermCfg, DatasetExportMode
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

from yamlab.robot.spec import get_robot

# Robot spec (IsaacLab-free; cached once) used to name the per-camera LeRobot RGB keys.
_ROBOT_SPEC = get_robot("yam")


class StreamWriteMode(enum.Enum):
    """Whether an episode write appends a partial chunk or finalizes the last chunk."""
    APPEND = 0  # Append a partial record (mid-episode flush)
    LAST = 1    # Write the final record of the episode

class StreamingHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """HDF5 file handler that appends episode data incrementally via a background writer.

    Datasets are created with an unbounded leading dimension and chunked along it, so episode
    chunks can be appended as they are recorded instead of buffering a whole episode in memory.
    """

    def __init__(self):
        """Initialize the handler with default chunking, no compression, and a background writer.

        Compression options (see the ``compression`` property):
            - "gzip": high compression ratio (50-80%), high latency (CPU-intensive).
            - "lzf": moderate compression ratio (30-50%), low latency (fast algorithm).
            - None: no compression, minimum latency but largest file size.
        """
        super().__init__()
        self._chunks_length = 100
        self._compression = None
        self._writer = self.SingleThreadHDF5DatasetWriter(self)

    class SingleThreadHDF5DatasetWriter:
        """Serializes episode writes onto a single background thread to keep HDF5 access ordered."""

        def __init__(self, file_handler):
            """Initialize the writer with a single-worker executor bound to ``file_handler``.

            Args:
                file_handler (StreamingHDF5DatasetFileHandler): Owning handler whose chunking and
                    compression settings the writes use.
            """
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.file_handler = file_handler

        def write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData, write_mode: StreamWriteMode):
            """Submit an episode write to the background thread on a deep copy of the data.

            Args:
                h5_episode_group (h5py.Group): Target group for this episode's datasets.
                episode (EpisodeData): Episode data to append.
                write_mode (StreamWriteMode): LAST blocks for the result (final chunk); APPEND
                    returns the pending future without waiting.

            Returns:
                The write result if ``write_mode`` is LAST, otherwise the pending Future.
            """
            episode_copy = copy.deepcopy(episode)
            future = self.executor.submit(self._do_write_episode, h5_episode_group, episode_copy)
            return future.result() if write_mode == StreamWriteMode.LAST else future

        def _do_write_episode(self, h5_episode_group: h5py.Group, episode: EpisodeData):
            """Append (or create) and extend the episode's datasets, then flush to disk.

            Args:
                h5_episode_group (h5py.Group): Target group for this episode's datasets.
                episode (EpisodeData): Episode data; nested dicts are written as nested groups.
            """
            def create_dataset_helper(group, key, value):
                """Helper method to create dataset that contains recursive dict objects."""
                if isinstance(value, dict):
                    if key not in group:
                        key_group = group.create_group(key)
                    else:
                        key_group = group[key]
                    for sub_key, sub_value in value.items():
                        create_dataset_helper(key_group, sub_key, sub_value)
                else:
                    if isinstance(value, list):
                        data = torch.stack(value).cpu().numpy()
                    else:
                        data = value.cpu().numpy()
                    if key not in group:
                        dataset = group.create_dataset(
                            key,
                            shape=data.shape,
                            maxshape=(None, *data.shape[1:]),
                            chunks=(self.file_handler.chunks_length, *data.shape[1:]),
                            dtype=data.dtype,
                            compression=self.file_handler.compression,
                        )
                        dataset[0: data.shape[0]] = data
                    else:
                        dataset = group[key]
                        dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
                        dataset[dataset.shape[0] - data.shape[0]:] = data

            for key, value in episode.data.items():
                create_dataset_helper(h5_episode_group, key, value)

            self.file_handler.flush()

        def shutdown(self):
            """Wait for all queued writes to finish and shut the executor down."""
            self.executor.shutdown(wait=True)

    @property
    def chunks_length(self) -> int:
        """HDF5 chunk size along the time axis, in samples."""
        return self._chunks_length

    @chunks_length.setter
    def chunks_length(self, chunks_length: int):
        self._chunks_length = chunks_length

    @property
    def compression(self) -> str | None:
        """HDF5 dataset compression filter ("gzip", "lzf", or None for uncompressed)."""
        return self._compression

    @compression.setter
    def compression(self, compression: str | None):
        self._compression = compression

    def write_episode(self, episode: EpisodeData, write_mode: StreamWriteMode):
        """Append an episode to its demo group, updating sample counts and metadata.

        Each episode is stored under a ``demo_<index>`` group. On the finalizing
        (StreamWriteMode.LAST) write, the global total-sample count and demo count are advanced.

        Args:
            episode (EpisodeData): Episode data to write; empty episodes are skipped.
            write_mode (StreamWriteMode): APPEND for a mid-episode chunk, LAST to finalize.
        """
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        group_name = f"demo_{self._demo_count}"
        if group_name not in self._hdf5_data_group:
            h5_episode_group = self._hdf5_data_group.create_group(group_name)
        else:
            h5_episode_group = self._hdf5_data_group[group_name]

        # store number of steps taken
        if "actions" in episode.data:
            if "num_samples" not in h5_episode_group.attrs:
                h5_episode_group.attrs["num_samples"] = 0
            h5_episode_group.attrs["num_samples"] += len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        self._writer.write_episode(h5_episode_group, episode, write_mode)

        if write_mode == StreamWriteMode.LAST:
            # increment total step counts
            self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

            # increment total demo counts
            self._demo_count += 1

    def close(self):
        """Flush the background writer and close the underlying HDF5 file."""
        self._writer.shutdown()
        super().close()


class StreamingRecorderManager(RecorderManager):
    """Recorder manager that streams episodes to HDF5 in chunks instead of buffering whole episodes.

    Installs the streaming HDF5 file handler and flushes each environment's buffered samples once
    they reach ``flush_steps``. Supports the EXPORT_ALL, EXPORT_NONE, and EXPORT_SUCCEEDED_ONLY
    export modes.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv) -> None:
        """Install the streaming file handler and configure chunking/compression.

        Args:
            cfg (object): Recorder manager configuration. Its ``dataset_file_handler_class_type``
                is overridden to use the streaming HDF5 handler.
            env (ManagerBasedEnv): Environment being recorded.

        Raises:
            AssertionError: If ``cfg.dataset_export_mode`` is not one of EXPORT_ALL, EXPORT_NONE,
                or EXPORT_SUCCEEDED_ONLY.
        """
        # use streaming_hdf5_dataset_file_handler
        cfg.dataset_file_handler_class_type = StreamingHDF5DatasetFileHandler

        super().__init__(cfg, env)

        assert self.cfg.dataset_export_mode in [
            DatasetExportMode.EXPORT_ALL, 
            DatasetExportMode.EXPORT_NONE,
            DatasetExportMode.EXPORT_SUCCEEDED_ONLY,  # For MimicGen data generation
        ], "only support EXPORT_NONE|EXPORT_ALL|EXPORT_SUCCEEDED_ONLY"

        self._env_steps_record = torch.zeros(self._env.num_envs)
        self._flush_steps = 100
        self._compression = None
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.chunks_length = self._flush_steps
            self._dataset_file_handler.compression = self._compression

    @property
    def flush_steps(self) -> int:
        """Number of buffered steps per environment before a chunk is flushed to disk."""
        return self._flush_steps

    @flush_steps.setter
    def flush_steps(self, flush_steps: int) -> None:
        self._flush_steps = flush_steps
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.chunks_length = self._flush_steps

    @property
    def compression(self) -> str | None:
        """HDF5 compression filter passed through to the streaming file handler."""
        return self._compression

    @compression.setter
    def compression(self, compression: str | None):
        self._compression = compression
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.compression = self._compression

    def __str__(self) -> str:
        """Return a human-readable summary of the recorder manager."""
        msg = "[Enhanced] StreamingRecorderManager. \n"
        msg += super().__str__()
        return msg

    def record_pre_step(self) -> None:
        """Record pre-step data and flush any environment that has reached ``flush_steps``."""
        self._env_steps_record += 1
        super().record_pre_step()
        self.export_episodes(from_step=True)

    def export_episodes(self, env_ids: Sequence[int] | None = None, from_step: bool = False) -> None:
        """Export buffered episode data per environment, honoring the configured export mode.

        For EXPORT_SUCCEEDED_ONLY, failed episodes are discarded without writing. When called
        from a step (``from_step=True``), only environments that have buffered at least
        ``flush_steps`` samples are flushed (as an APPEND chunk); otherwise all selected
        environments are finalized (LAST).

        Args:
            env_ids (Sequence[int] or None): Environment indices to export; None exports all.
            from_step (bool): True when called mid-episode from ``record_pre_step``.
        """
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Export episode data through dataset exporter
        for env_id in env_ids:
            if env_id in self._episodes and not self._episodes[env_id].is_empty() and (self._env_steps_record[env_id] >= self._flush_steps or not from_step):
                if self._env.cfg.seed is not None:
                    self._episodes[env_id].seed = self._env.cfg.seed
                episode_succeeded = self._episodes[env_id].success
                target_dataset_file_handler = None
                
                # Determine which file handler to use based on export mode
                if self.cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL:
                    target_dataset_file_handler = self._dataset_file_handler
                elif self.cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY:
                    # Only export successful episodes
                    if episode_succeeded:
                        target_dataset_file_handler = self._dataset_file_handler
                    else:
                        # Clear failed episode data without exporting
                        self._clear_episode_cache([env_id])
                        self._exported_failed_episode_count[env_id] = self._exported_failed_episode_count.get(env_id, 0) + 1
                        continue
                
                if target_dataset_file_handler is not None:
                    write_mode = StreamWriteMode.APPEND if from_step else StreamWriteMode.LAST
                    target_dataset_file_handler.write_episode(self._episodes[env_id], write_mode)
                    self._clear_episode_cache([env_id])
                    
                if episode_succeeded:
                    self._exported_successful_episode_count[env_id] = (
                        self._exported_successful_episode_count.get(env_id, 0) + 1
                    )
                else:
                    self._exported_failed_episode_count[env_id] = self._exported_failed_episode_count.get(env_id, 0) + 1

    def _clear_episode_cache(self, env_ids: Sequence[int] | None = None) -> None:
        """Drop the buffered episode data and reset the step counter for the given environments.

        Args:
            env_ids (Sequence[int] or None): Environment indices to clear; None clears all.
        """
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        for env_id in env_ids:
            del self._episodes[env_id]._data
            self._episodes[env_id].data = dict()
            self._env_steps_record[env_id] = 0

class ManualControlRecorderManager(StreamingRecorderManager):
    """HDF5 recorder that supports both manual teleoperation control and MimicGen modes.

    Manual mode: records only when _recording_active is True; call start_recording()
    to begin and save_trajectory() to export.
    MimicGen mode: records whenever datagen_config is present; MimicGen handles
    export via export_episodes().
    """

    def __init__(self, cfg, env):
        """Initialize the recorder and enable manual-control mode by default.

        Args:
            cfg (object): Recorder manager configuration.
            env (ManagerBasedEnv): Environment being recorded.
        """
        super().__init__(cfg, env)
        self._manual_control = True

    def _should_record(self) -> bool:
        """Return True if the current mode requires recording."""
        if hasattr(self._env, 'cfg') and hasattr(self._env.cfg, 'datagen_config') and self._env.cfg.datagen_config is not None:
            return True
        if hasattr(self._env, '_recording_active') and self._env._recording_active:
            return True
        return False

    def record_pre_step(self) -> None:
        """Increment step counter and record pre-step data when active."""
        self._env_steps_record += 1
        if self._should_record():
            super(StreamingRecorderManager, self).record_pre_step()

    def record_post_step(self) -> None:
        """Record post-step data when active."""
        if self._should_record():
            super(StreamingRecorderManager, self).record_post_step()

    def record_pre_reset(self, env_ids=None, force_export_or_skip=None) -> None:
        """Suppress auto-export on reset in MimicGen mode; MimicGen manages export manually."""
        if hasattr(self._env, 'cfg') and hasattr(self._env.cfg, 'datagen_config') and self._env.cfg.datagen_config is not None:
            force_export_or_skip = False
        super(StreamingRecorderManager, self).record_pre_reset(env_ids, force_export_or_skip)

    def record_post_reset(self, env_ids=None) -> None:
        """Record initial state after reset."""
        super(StreamingRecorderManager, self).record_post_reset(env_ids)

    def export_episodes_manual(self, env_ids=None):
        """Manually export episodes for teleoperation/replay workflows."""
        return self.export_episodes(env_ids, from_step=False)

    def export_episodes(self, env_ids=None, from_step=False):
        """Export episodes to LeRobot in MimicGen mode, or HDF5 otherwise.

        In MimicGen mode with a LeRobot recorder attached, routes successful
        episodes to lerobot_recorder.save_episode() and discards failed ones.

        Args:
            env_ids: Environment indices to export. None exports all environments.
            from_step: Whether called from a mid-step export path.
        """
        if (hasattr(self._env, 'lerobot_recorder') and
            self._env.lerobot_recorder is not None and
            hasattr(self._env, 'cfg') and
            hasattr(self._env.cfg, 'datagen_config') and
            self._env.cfg.datagen_config is not None):
            if env_ids is None:
                env_ids = list(range(self._env.num_envs))
            elif isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.tolist()

            # Cap the saved dataset at exactly generation_num_trials episodes. With parallel
            # environments, more than one trial can succeed within a single batch step, so the
            # number of successes (and saved episodes) can overshoot the requested count; the
            # extra successes are discarded once the target is reached.
            target = getattr(self._env.cfg.datagen_config, "generation_num_trials", None)
            recorder = self._env.lerobot_recorder
            for env_id in env_ids:
                if env_id in self._episodes and not self._episodes[env_id].is_empty():
                    episode_succeeded = self._episodes[env_id].success
                    target_reached = target is not None and recorder.episode_count >= target
                    if episode_succeeded and not target_reached:
                        recorder.save_episode(env_id=env_id, encode_videos=False)
                        print(f"[LeRobotRecorder] Saved successful MimicGen episode (env {env_id})")
                    elif episode_succeeded and target_reached:
                        recorder.discard_episode(env_id=env_id)
                        print(f"[LeRobotRecorder] Reached target of {target} episodes; "
                              f"discarding extra successful episode (env {env_id})")
                    else:
                        recorder.discard_episode(env_id=env_id)
                        print(f"[LeRobotRecorder] Discarded failed MimicGen episode (env {env_id})")

                    self._clear_episode_cache([env_id])
        else:
            super().export_episodes(env_ids, from_step)


class LeRobotRecorderManager:
    """Writes observations and actions directly to a LeRobot v2 dataset.

    Saves observations and actions frame-by-frame to LeRobot parquet + video format.
    Supports async episode saving via a single background worker thread (required by
    LeRobot's sequential episode ordering constraint).

    Usage:
        1. Initialize with features, fps, and output path.
        2. Call record_frame() each step with observation data.
        3. Call save_episode() when episode completes.
        4. Call consolidate() at the end to encode videos and compute stats.
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        fps: int = 30,
        features: dict = None,
        robot_type: str = "YAM",
        task_name: str = "PutPotOnCooktop-v0",
        use_videos: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 4,
        discard_first_n_frames: int = 0,
        discard_last_n_frames: int = 0,
        async_save: bool = True,
        max_pending_saves: int = 16,
        image_downsample_factor: int = 1,
        base_width: int = 320,
        base_height: int = 240,
    ):
        """Initialize LeRobot recorder manager.

        Args:
            repo_id: Repository ID for the dataset (e.g., "user/dataset_name").
            root: Root directory to save the dataset.
            fps: Frames per second.
            features: Feature definitions. None uses the default bimanual YAM features.
            robot_type: Robot type string embedded in dataset metadata.
            task_name: Task name for episode metadata.
            use_videos: Encode frames as video (True) or save raw images (False).
            image_writer_processes: Processes for async image writing.
            image_writer_threads: Threads for async image writing.
            discard_first_n_frames: Frames to drop from the start of each episode.
            discard_last_n_frames: Frames to drop from the end of each episode.
            async_save: If True, serialize episodes in a background thread.
            max_pending_saves: Maximum queued async saves before blocking.
            image_downsample_factor: Spatial downsampling factor applied at record time.
            base_width: Camera sensor render width; recorded width = base_width // factor.
            base_height: Camera sensor render height.
        """
        self._base_width = base_width
        self._base_height = base_height
        self.repo_id = repo_id
        self.root = Path(root)
        self.fps = fps
        self.task_name = task_name
        self.use_videos = use_videos
        self._episode_count = 0
        self._recording_active = False
        # Per-environment episode buffers, keyed by env_id, so parallel trials in
        # different environments are recorded as separate episodes.
        self._episode_frames = {}        # env_id -> list[frame dict]
        self._episode_frame_index = {}   # env_id -> frames seen this episode

        self._discard_first_n_frames = discard_first_n_frames
        self._discard_last_n_frames = discard_last_n_frames

        self._image_downsample_factor = image_downsample_factor

        self._async_save = async_save
        self._max_pending_saves = max_pending_saves
        self._pending_saves = []
        # Single worker ensures saves are sequential (LeRobot requires ordered episodes).
        self._save_executor = ThreadPoolExecutor(max_workers=1) if async_save else None
        self._save_lock = None
        if async_save:
            import threading
            self._save_lock = threading.Lock()

        if features is None:
            features = self._get_default_bimanual_features(image_downsample_factor)

        self.features = features

        if self.root.exists():
            print(f"[LeRobotRecorder] Removing existing dataset at {self.root}")
            shutil.rmtree(self.root)

        print(f"[LeRobotRecorder] Creating LeRobotDataset at {self.root}")
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=str(self.root),
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )

        print(f"[LeRobotRecorder] Initialized with {len(features)} features")

    def _get_default_bimanual_features(self, image_downsample_factor: int = 1) -> dict:
        """Return default feature definitions for the bimanual YAM robot.

        The recorded image resolution is the camera sensor resolution
        (``self._base_width`` x ``self._base_height``) divided by
        ``image_downsample_factor``.

        Args:
            image_downsample_factor: Spatial downsampling factor applied to the sensor
                resolution (1 = record at the sensor resolution).
        """
        base_width, base_height = self._base_width, self._base_height
        width = base_width // image_downsample_factor
        height = base_height // image_downsample_factor

        def make_image_feature(fps: float = 30.0):
            """Build video feature descriptor for a single RGB camera."""
            return {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }

        return {
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": [
                    "left_joint1.pos", "left_joint2.pos", "left_joint3.pos",
                    "left_joint4.pos", "left_joint5.pos", "left_joint6.pos",
                    "left_finger.pos",
                    "right_joint1.pos", "right_joint2.pos", "right_joint3.pos",
                    "right_joint4.pos", "right_joint5.pos", "right_joint6.pos",
                    "right_finger.pos",
                ]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": [
                    "left_joint1.pos", "left_joint2.pos", "left_joint3.pos",
                    "left_joint4.pos", "left_joint5.pos", "left_joint6.pos",
                    "left_finger.pos",
                    "right_joint1.pos", "right_joint2.pos", "right_joint3.pos",
                    "right_joint4.pos", "right_joint5.pos", "right_joint6.pos",
                    "right_finger.pos",
                ]
            },
            # One RGB video feature per configured camera (key from the robot config).
            **{f"observation.images.{_ROBOT_SPEC.camera_lerobot_key(name)}": make_image_feature(self.fps)
               for name in _ROBOT_SPEC.camera_names},
        }

    def start_recording(self):
        """Open a recording session and clear all per-environment frame buffers.

        Frames are buffered separately per environment so that parallel data
        generation (one independent trial per environment) can be de-interleaved
        into separate episodes. The session remains open across per-environment
        episode saves; calling this again resets every buffer.
        """
        self._recording_active = True
        self._episode_frames = {}        # env_id -> list[frame dict]
        self._episode_frame_index = {}   # env_id -> frames seen this episode
        print(f"[LeRobotRecorder] Started recording (next episode index {self._episode_count})")

    def is_recording(self) -> bool:
        """Return True while a recording session is open."""
        return self._recording_active

    def _ensure_env_buffer(self, env_id: int) -> None:
        """Create the frame buffer and frame counter for ``env_id`` if absent."""
        if env_id not in self._episode_frames:
            self._episode_frames[env_id] = []
            self._episode_frame_index[env_id] = 0

    def _reset_env_buffer(self, env_id: int) -> None:
        """Clear one environment's frame buffer and reset its frame counter."""
        self._episode_frames[env_id] = []
        self._episode_frame_index[env_id] = 0

    def record_frame(self, obs_dict: dict, action: torch.Tensor) -> None:
        """Buffer one frame per environment from a batched observation/action.

        Each environment's slice is appended to that environment's own episode
        buffer, so concurrent trials in different environments stay separate.

        Args:
            obs_dict: Observation dict from the environment. The consumed entries
                (``left_arm_joint_pos``, ``right_arm_joint_pos``, the gripper states,
                and the camera RGB tensors) are batched with a leading environment
                dimension, i.e. shape ``(num_envs, ...)``.
            action: Action tensor of shape ``(num_envs, 14)``. A 1-D ``(14,)`` action
                is treated as a single environment.
        """
        if not self._recording_active:
            return

        policy_obs = obs_dict.get("policy", obs_dict)

        if isinstance(action, torch.Tensor):
            actions_np = action.detach().cpu().numpy()
        else:
            actions_np = np.asarray(action)
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        num_envs = actions_np.shape[0]

        for env_id in range(num_envs):
            self._ensure_env_buffer(env_id)
            # Drop the first N frames of each episode.
            if self._episode_frame_index[env_id] < self._discard_first_n_frames:
                self._episode_frame_index[env_id] += 1
                continue
            self._episode_frames[env_id].append(self._build_frame(policy_obs, actions_np, env_id))
            self._episode_frame_index[env_id] += 1

    def _build_frame(self, policy_obs: dict, actions_np: np.ndarray, env_id: int) -> dict:
        """Build one LeRobot frame dict for a single environment.

        Args:
            policy_obs: The ``policy`` observation group, batched over environments.
            actions_np: Actions as a NumPy array of shape ``(num_envs, 14)``.
            env_id: Environment index whose slice is extracted.

        Returns:
            A dict of LeRobot feature keys for this environment: ``action`` (14,),
            ``observation.state`` (14,) float32, and each configured
            ``observation.images.*`` as an ``(H, W, 3)`` uint8 array.
        """
        frame = {"action": actions_np[env_id]}

        def _env_slice(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            else:
                value = np.asarray(value)
            return value[env_id] if value.ndim >= 1 else value

        left_joint_pos = _env_slice(policy_obs.get("left_arm_joint_pos"))
        right_joint_pos = _env_slice(policy_obs.get("right_arm_joint_pos"))
        left_gripper = _env_slice(policy_obs.get("left_gripper_state"))
        right_gripper = _env_slice(policy_obs.get("right_gripper_state"))

        if left_joint_pos is not None and right_joint_pos is not None:
            if left_gripper is not None and right_gripper is not None:
                state = np.concatenate([
                    left_joint_pos,
                    left_gripper if left_gripper.ndim >= 1 else [left_gripper],
                    right_joint_pos,
                    right_gripper if right_gripper.ndim >= 1 else [right_gripper],
                ])
            else:
                state = np.concatenate([left_joint_pos, right_joint_pos])
            if len(state) == 14:
                frame["observation.state"] = state.astype(np.float32)

        # Map each camera's policy RGB obs (<name>_camera_rgb) to its LeRobot key
        # (observation.images.<lerobot_key>), both from the robot config.
        image_keys = [
            (f"{name}_camera_rgb", f"observation.images.{_ROBOT_SPEC.camera_lerobot_key(name)}")
            for name in _ROBOT_SPEC.camera_names
        ]

        for src_key, dst_key in image_keys:
            if dst_key in self.features and src_key in policy_obs:
                img = policy_obs[src_key]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                else:
                    img = np.asarray(img)
                if img.ndim == 4:
                    img = img[env_id]
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                if self._image_downsample_factor > 1:
                    h, w = img.shape[:2]
                    new_h = h // self._image_downsample_factor
                    new_w = w // self._image_downsample_factor
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame[dst_key] = img

        return frame

    def save_episode(self, env_id: int = 0, encode_videos: bool = False) -> bool:
        """Save one environment's buffered episode to disk.

        Only the given environment's buffer is flushed and cleared; the recording
        session stays open so other environments keep accumulating their own
        episodes. The last ``discard_last_n_frames`` frames are dropped. With
        ``async_save=True`` the write runs in a background thread on a deep copy of
        the frames, and the episode count is incremented immediately.

        Args:
            env_id: Environment index whose buffered episode to save.
            encode_videos: Encode video inline (default False; done in consolidate).

        Returns:
            True if an episode was queued/saved, False if its buffer was empty or
            fully consumed by ``discard_last_n_frames``.
        """
        if not self._recording_active:
            print("[LeRobotRecorder] No active recording to save")
            return False

        try:
            frames = self._episode_frames.get(env_id, [])
            valid_end = len(frames) - self._discard_last_n_frames

            if valid_end <= 0:
                print(f"[LeRobotRecorder] env {env_id}: discard_last_n_frames ({self._discard_last_n_frames}) >= frames in buffer ({len(frames)}). Discarding episode.")
                self._reset_env_buffer(env_id)
                return False

            frames_to_save = frames[:valid_end]
            num_frames = len(frames_to_save)
            episode_index = self._episode_count

            self._episode_count += 1
            self._reset_env_buffer(env_id)

            if self._async_save and self._save_executor is not None:
                self._cleanup_completed_saves()
                if len(self._pending_saves) >= self._max_pending_saves:
                    self._wait_for_one_save()

                frames_copy = copy.deepcopy(frames_to_save)
                future = self._save_executor.submit(
                    self._do_save_episode_sync,
                    frames_copy,
                    encode_videos,
                    episode_index,
                    num_frames,
                )
                self._pending_saves.append(future)
                print(f"[LeRobotRecorder] Queued episode {episode_index} (env {env_id}) for async save ({num_frames} frames)")
            else:
                self._do_save_episode_sync(frames_to_save, encode_videos, episode_index, num_frames)

            return True
        except Exception as e:
            print(f"[LeRobotRecorder] Failed to save episode (env {env_id}): {e}")
            import traceback
            traceback.print_exc()
            self._reset_env_buffer(env_id)
            return False

    def _do_save_episode_sync(self, frames: list, encode_videos: bool, episode_index: int, num_frames: int):
        """Perform the actual episode save, serialized by a lock.

        Runs in the background thread (max_workers=1 ensures sequential execution).
        """
        try:
            if self._save_lock is not None:
                self._save_lock.acquire()

            try:
                for frame in frames:
                    self.dataset.add_frame(frame)

                self.dataset.save_episode(
                    task=self.task_name,
                    encode_videos=encode_videos,
                )
            finally:
                if self._save_lock is not None:
                    self._save_lock.release()

            print(f"[LeRobotRecorder] Saved episode {episode_index} ({num_frames} frames, skipped first {self._discard_first_n_frames} and last {self._discard_last_n_frames})")
        except Exception as e:
            print(f"[LeRobotRecorder] Failed to save episode {episode_index}: {e}")
            import traceback
            traceback.print_exc()

    def _cleanup_completed_saves(self):
        """Remove completed futures from the pending list."""
        self._pending_saves = [f for f in self._pending_saves if not f.done()]

    def _wait_for_one_save(self):
        """Block until at least one pending save completes."""
        if self._pending_saves:
            from concurrent.futures import wait, FIRST_COMPLETED
            done, _ = wait(self._pending_saves, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Exception as e:
                    print(f"[LeRobotRecorder] Async save error: {e}")
            self._cleanup_completed_saves()

    def wait_for_pending_saves(self):
        """Block until all pending async saves complete.

        Call this before consolidate() or whenever all episodes must be flushed.
        """
        if self._pending_saves:
            print(f"[LeRobotRecorder] Waiting for {len(self._pending_saves)} pending saves...")
            from concurrent.futures import wait
            done, _ = wait(self._pending_saves)
            for future in done:
                try:
                    future.result()
                except Exception as e:
                    print(f"[LeRobotRecorder] Async save error: {e}")
            self._pending_saves = []
            print("[LeRobotRecorder] All pending saves completed")

    def discard_episode(self, env_id: int = 0) -> None:
        """Discard one environment's buffered episode without saving.

        Only this environment's buffered frames are dropped; the recording session
        stays open and other environments are unaffected. Buffered frames live in
        this manager (not the LeRobotDataset), so nothing needs to be flushed from
        the dataset's own frame buffer here.

        Args:
            env_id: Environment index whose buffered episode to discard.
        """
        if self._recording_active:
            self._reset_env_buffer(env_id)
            print(f"[LeRobotRecorder] Discarded current episode (env {env_id})")

    def consolidate(
        self,
        run_compute_stats: bool = False,
    ):
        """Encode videos and compute statistics. Waits for pending async saves first.

        Args:
            run_compute_stats: Whether to compute dataset statistics.
        """
        self.wait_for_pending_saves()

        print(f"[LeRobotRecorder] Consolidating dataset...")
        self.dataset.consolidate(
            run_compute_stats=run_compute_stats,
            keep_image_files=False,
        )
        print(f"[LeRobotRecorder] Dataset consolidated: {self._episode_count} episodes")

        self._generate_custom_metadata()

    def _generate_custom_metadata(self):
        """Generate embodiment.json, modality.json, and metadata.json.

        These files are not part of standard LeRobot v2.0 but are required by the
        downstream training pipeline. Structure mirrors convert_hdf5_to_lerobot.py.
        """
        meta_dir = Path(self.dataset.root) / "meta"
        meta_dir.mkdir(exist_ok=True)

        robot_type = self.dataset.meta.info.get("robot_type", "YAM")

        embodiment = {
            "robot_name": robot_type,
            "robot_type": robot_type,
            "record_frequency": float(self.fps),
            "body_controller_frequency": float(self.fps),
            "hand_controller_frequency": float(self.fps),
            "embodiment_tag": robot_type.lower()
        }
        with open(meta_dir / "embodiment.json", "w") as f:
            json.dump(embodiment, f, indent=4)

        modality = {
            "state": {},
            "action": {},
            "video": {},
            # annotation.human.task_description maps task_index for language-conditioned training.
            "annotation": {
                "human.task_description": {
                    "original_key": "task_index"
                }
            }
        }

        if "observation.state" in self.features:
            modality["state"]["left_arm"] = {"original_key": "observation.state", "start": 0, "end": 6}
            modality["state"]["left_gripper"] = {"original_key": "observation.state", "start": 6, "end": 7}
            modality["state"]["right_arm"] = {"original_key": "observation.state", "start": 7, "end": 13}
            modality["state"]["right_gripper"] = {"original_key": "observation.state", "start": 13, "end": 14}

        if "action" in self.features:
            modality["action"]["left_arm"] = {"original_key": "action", "start": 0, "end": 6}
            modality["action"]["left_gripper"] = {"original_key": "action", "start": 6, "end": 7}
            modality["action"]["right_arm"] = {"original_key": "action", "start": 7, "end": 13}
            modality["action"]["right_gripper"] = {"original_key": "action", "start": 13, "end": 14}

        for key in self.dataset.meta.video_keys:
            if "rgb" in key:
                video_name = key.replace("observation.images.", "")
                modality["video"][video_name] = {"original_key": key}

        with open(meta_dir / "modality.json", "w") as f:
            json.dump(modality, f, indent=4)

        metadata = {
            "dataset_name": self.dataset.repo_id,
            "dataset_statistics": {},
            "modalities": {},
            "embodiment": embodiment,
            "processing": {"rgb_encoding": "h264_video"},
            "version": "1.0"
        }

        if self.dataset.meta.stats:
            for key, stats in self.dataset.meta.stats.items():
                def tensor_to_list(tensor):
                    """Convert a tensor or array to a JSON-serializable list."""
                    if hasattr(tensor, "tolist"):
                        return tensor.tolist()
                    elif hasattr(tensor, "numpy"):
                        return tensor.numpy().tolist()
                    else:
                        return tensor

                metadata["dataset_statistics"][key] = {
                    "mean": tensor_to_list(stats["mean"]),
                    "std": tensor_to_list(stats["std"]),
                    "min": tensor_to_list(stats["min"]),
                    "max": tensor_to_list(stats["max"]),
                }

        metadata["modalities"]["video"] = {}
        for key in self.dataset.meta.video_keys:
            if "rgb" in key:
                video_name = key.replace("observation.images.", "")
                shape = self.features[key]["shape"]
                metadata["modalities"]["video"][video_name] = {
                    "resolution": [shape[0], shape[1]],
                    "channels": shape[2] if len(shape) > 2 else 1,
                    "fps": float(self.fps),
                    "codec": "h264",
                    "pixel_format": "yuv420p"
                }

        if "observation.state" in self.features:
            metadata["modalities"]["state"] = {
                "left_arm": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [6],
                    "continuous": True,
                    "joint_names": ["left_joint1.pos", "left_joint2.pos", "left_joint3.pos",
                                   "left_joint4.pos", "left_joint5.pos", "left_joint6.pos"]
                },
                "left_gripper": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [1],
                    "continuous": True,
                    "joint_names": ["left_finger.pos"]
                },
                "right_arm": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [6],
                    "continuous": True,
                    "joint_names": ["right_joint1.pos", "right_joint2.pos", "right_joint3.pos",
                                   "right_joint4.pos", "right_joint5.pos", "right_joint6.pos"]
                },
                "right_gripper": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [1],
                    "continuous": True,
                    "joint_names": ["right_finger.pos"]
                }
            }

        if "action" in self.features:
            metadata["modalities"]["action"] = {
                "left_arm": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [6],
                    "continuous": True,
                    "joint_names": ["left_joint1.pos", "left_joint2.pos", "left_joint3.pos",
                                   "left_joint4.pos", "left_joint5.pos", "left_joint6.pos"]
                },
                "left_gripper": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [1],
                    "continuous": True,
                    "joint_names": ["left_finger.pos"]
                },
                "right_arm": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [6],
                    "continuous": True,
                    "joint_names": ["right_joint1.pos", "right_joint2.pos", "right_joint3.pos",
                                   "right_joint4.pos", "right_joint5.pos", "right_joint6.pos"]
                },
                "right_gripper": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [1],
                    "continuous": True,
                    "joint_names": ["right_finger.pos"]
                }
            }

        with open(meta_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[LeRobotRecorder] Generated custom metadata files in {meta_dir}")

    @property
    def episode_count(self) -> int:
        """Number of saved episodes."""
        return self._episode_count

    def close(self):
        """Wait for pending saves, shutdown the executor, and stop the image writer."""
        self.wait_for_pending_saves()

        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True)
            self._save_executor = None

        if hasattr(self.dataset, 'image_writer') and self.dataset.image_writer is not None:
            self.dataset.stop_image_writer()


# ============================================================================
# MimicGen recorder terms - capture datagen info during annotation replay
# ============================================================================

class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that captures MimicGen datagen info (object/eef/target poses) each step."""

    def record_pre_step(self):
        """Record per-step datagen info.

        Returns:
            tuple: (key, dict) where key is "obs/datagen_info" and dict holds the
                object poses, per-eef poses, and target eef poses for this step.
        """
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion signals each step."""

    def record_pre_step(self):
        """Record per-step subtask termination signals.

        Returns:
            tuple: (key, signals) where key is "obs/datagen_info/subtask_term_signals".
        """
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask terms observation recorder term."""
    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic-specific recorder terms."""
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def configure_annotation_recorder(env, output_dir: str, output_filename: str, auto_mode: bool):
    """Set up the annotation-specific recorder on an already-created environment.

    Replaces the default recorder_manager with one that records MimicGen
    datagen_info and (optionally) subtask termination signals.

    Args:
        env (ManagerBasedRLMimicEnv): The environment (already created via create_task_environment).
        output_dir (str): Directory for the output HDF5.
        output_filename (str): Base filename for the output HDF5 (without extension).
        auto_mode (bool): If True, record subtask_term_signals automatically.
    """
    recorder_cfg = MimicRecorderManagerCfg()
    if not auto_mode:
        # Disable automatic subtask signal recording in manual mode
        recorder_cfg.record_pre_step_subtask_term_signals = None

    recorder_cfg.dataset_export_dir_path = output_dir
    recorder_cfg.dataset_filename = output_filename
    recorder_cfg.export_in_record_pre_reset = False  # Manual export control

    env.recorder_manager = ManualControlRecorderManager(recorder_cfg, env)


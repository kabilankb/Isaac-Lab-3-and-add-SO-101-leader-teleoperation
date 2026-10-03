"""Sim-side teleoperation server: RPC control and demo recording for the bimanual YAM.

Wraps a :class:`ManagerBasedRLEnv` task with IsaacLab's RecorderManager for HDF5
demo collection (:class:`TeleopManagerWrapper`) and exposes a 14-dim bimanual
joint-position interface over RPC (:class:`BimanualRPCServer`) for the JoyLo leader.
"""

import torch
import json
from typing import Dict, Any, Tuple

from isaaclab.envs import ManagerBasedRLEnv
from yamlab.utils.recorders import StreamingRecorderManager
from yamlab.utils.io import serialize_dict
from yamlab.robot.yam import ROBOT, YamActionLayout

class TeleopRecorderManager(StreamingRecorderManager):
    """Recorder for teleoperation collection.

    Buffers every step and exports on demand (the operator triggers save via
    TeleopManagerWrapper), and tolerates missing episodes during reset.
    """

    def __init__(self, cfg, env):
        """Initialize the recorder and enable manual (operator-triggered) export.

        Args:
            cfg: RecorderManager config (``env.cfg.recorders``).
            env (ManagerBasedRLEnv): The wrapped task environment.
        """
        super().__init__(cfg, env)
        self._manual_control = True  # Flag to disable automatic export

    def record_pre_step(self) -> None:
        """Record the pre-step snapshot without auto-exporting the episode."""
        self._env_steps_record += 1
        super(StreamingRecorderManager, self).record_pre_step()  # Call parent's parent
        # Don't call self.export_episodes(from_step=True) - we want manual control

    def export_episodes_manual(self, env_ids=None):
        """Export buffered episodes on operator request.

        Args:
            env_ids (Sequence[int] | None): Environment indices to export; None exports all.

        Returns:
            The parent ``export_episodes`` return value.
        """
        return self.export_episodes(env_ids, from_step=False)

    def record_pre_reset(self, env_ids):
        """Record the pre-reset snapshot, tolerating envs without an open episode.

        Args:
            env_ids (Sequence[int]): Environment indices being reset.
        """
        # Call parent method with original env_ids - it will handle episode creation
        super().record_pre_reset(env_ids)

    def set_success_to_episodes(self, env_ids, success_values):
        """Tag episodes with their success flag, skipping envs with no open episode.

        Args:
            env_ids (Sequence[int]): Environment indices to tag.
            success_values (Sequence[bool]): Success flag per ``env_ids`` entry.
        """
        # Ensure episodes exist before setting success values
        if not hasattr(self, '_episodes') or self._episodes is None:
            return
            
        # Only process episodes that actually exist
        existing_env_ids = []
        success_values_subset = []
        
        for i, env_id in enumerate(env_ids):
            if env_id in self._episodes:
                existing_env_ids.append(env_id)
                success_values_subset.append(success_values[i])
        
        if not existing_env_ids:
            # No episodes exist, nothing to do
            return
            
        # Convert to tensor for parent method
        success_values_tensor = torch.tensor(success_values_subset, dtype=torch.bool, device=self._env.device)
        
        # Call parent method only for existing episodes
        super().set_success_to_episodes(existing_env_ids, success_values_tensor)


class TeleopManagerWrapper:
    """Wraps a task environment to collect teleoperation demos via IsaacLab's RecorderManager.

    Tracks per-asset demo counts and drives episode start/save/discard from operator
    commands, recording state and actions to HDF5.
    """

    def __init__(self,
                 task_env: ManagerBasedRLEnv,
                 output_dir: str = "./datasets",
                 output_filename: str = "teleop_demos",
                 demos_per_asset: int = 5,
                 flush_every_n_steps: int = 100):
        """Initialize the wrapper and configure the environment for data collection.

        Args:
            task_env (ManagerBasedRLEnv): The task environment to wrap.
            output_dir (str): Directory to store dataset files.
            output_filename (str): Base filename for datasets (without extension).
            demos_per_asset (int): Number of demonstrations to collect per asset.
            flush_every_n_steps (int): How often (in steps) to flush data to disk.
        """
        self.task_env = task_env
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.flush_every_n_steps = flush_every_n_steps
        
        # Teleoperation-specific demo tracking
        self.demos_per_asset = demos_per_asset
        self.current_demo_count = 0
        self.total_demos_collected = 0
        
        # Configure the environment for data collection
        self._setup_data_collection()
        
        print(f"[INFO] TeleopManagerWrapper initialized")
        print(f"[INFO] - Using IsaacLab's RecorderManager system")
        print(f"[INFO] - Output directory: {output_dir}")
        print(f"[INFO] - Demos per asset: {demos_per_asset}")
    
    def _setup_data_collection(self):
        """Configure the environment and install the streaming teleop recorder."""
        # Configure the environment for data collection
        self.task_env.configure_for_data_collection(
            output_dir=self.output_dir,
            filename=self.output_filename
        )
        
        # Replace with streaming recorder manager for better performance
        if self.task_env.recorder_manager is not None:
            del self.task_env.recorder_manager
            
        self.task_env.recorder_manager = TeleopRecorderManager(
            self.task_env.cfg.recorders,
            self.task_env,
        )
        self.task_env.recorder_manager.flush_steps = self.flush_every_n_steps
        self.task_env.recorder_manager.compression = 'lzf'
        
        print("[INFO] TeleopRecorderManager configured")
    
    def _add_datagroup_attr(self, attr_name: str, attr: Any):
        """Attach an attribute to the HDF5 data group, no-op if the file is not ready.

        Args:
            attr_name (str): Name of the attribute to add.
            attr (Any): Attribute value; dict/list values are JSON-serialized first.
        """
        # Check if recorder manager and dataset file handler are initialized
        if self.task_env.recorder_manager is None:
            print(f"[WARNING] Cannot add attribute '{attr_name}': recorder manager not initialized")
            return
        
        dataset_file_handler = self.task_env.recorder_manager._dataset_file_handler
        if dataset_file_handler is None:
            print(f"[WARNING] Cannot add attribute '{attr_name}': dataset file handler not initialized")
            return
        
        # Check if HDF5 file is initialized
        if not hasattr(dataset_file_handler, '_hdf5_data_group'):
            print(f"[WARNING] Cannot add attribute '{attr_name}': HDF5 data group not initialized")
            return
        
        try:
            # Get the data group
            data_group = dataset_file_handler._hdf5_data_group
            
            # Serialize the attribute value to JSON if it's a dict or list
            if isinstance(attr, (dict, list)):
                attr_value = json.dumps(attr)
            else:
                attr_value = attr
            
            # Save the attribute
            data_group.attrs[attr_name] = attr_value
            print(f"[INFO] Added attribute '{attr_name}' to HDF5 data group")
        except Exception as e:
            print(f"[WARNING] Failed to add attribute '{attr_name}' to HDF5 data group: {e}")
    
    def step(self, action: torch.Tensor, metadata: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the wrapped environment (recording happens automatically) and return its outputs.

        Args:
            action (torch.Tensor): Action to apply, float tensor of shape (num_envs, 14).
            metadata (Dict[str, Any] | None): Extra key/values merged into the returned info dict.

        Returns:
            Tuple: ``(obs_dict, reward, terminated, truncated, info)`` from the env step.
        """
        # Step the environment (recording happens automatically)
        obs_dict, reward, terminated, truncated, info = self.task_env.step(action)
        
        # Add metadata to info if provided
        if metadata is not None:
            info.update(metadata)
        
        return obs_dict, reward, terminated, truncated, info
    
    def start_recording(self) -> bool:
        """Reset the success check and start recording a new episode.

        Returns:
            bool: True if recording started.
        """
        self.task_env.reset_success_check()
        return self.task_env.start_recording()

    def save_trajectory(self) -> bool:
        """Save the current trajectory and increment the demo counters.

        If a pose schedule is enabled, the environment also advances to the next
        pose for the following demo.

        Returns:
            bool: True if a trajectory was saved.
        """
        success = self.task_env.save_trajectory()
        if success:
            # Increment demo counters only when trajectory is actually saved
            self.current_demo_count += 1
            self.total_demos_collected += 1
            print(f"[INFO] Trajectory saved (Demo {self.current_demo_count}/{self.demos_per_asset})")
        return success
    
    def reset_task(self) -> Tuple[Dict[str, Any], bool]:
        """Reset the task, discarding any in-progress episode without saving it.

        Unlike :meth:`save_trajectory`, this does NOT advance the pose schedule:
        objects reset to their current scheduled pose, not the next one.

        Returns:
            Tuple: ``(demo_info, is_complete)`` where ``demo_info`` is the task info
            plus demo-tracking fields and ``is_complete`` flags that the target demo
            count has been reached.
        """
        # Use the task environment's reset_task method (does NOT advance pose schedule)
        self.task_env.reset_task()
        
        # Get updated task info
        task_info = self.task_env.get_current_task_info()
        
        # Add teleoperation-specific demo tracking info
        demo_info = {
            **serialize_dict(task_info),  # Include task environment info (serialized)
            "demo_count": self.current_demo_count,
            "demos_per_asset": self.demos_per_asset,
            "total_demos_collected": self.total_demos_collected,
            "progress_percent": (self.total_demos_collected / (serialize_dict(task_info).get("total_objects", 1) * self.demos_per_asset)) * 100,
            "demos_remaining": self.demos_per_asset - self.current_demo_count,
        }
        
        # Check if data collection is complete
        is_complete = self.current_demo_count >= self.demos_per_asset
        
        return demo_info, is_complete
    
    def is_data_collection_complete(self) -> bool:
        """Whether the target demo count for the current asset has been reached.

        Returns:
            bool: True if collection is complete.
        """
        return self.current_demo_count >= self.demos_per_asset

    def is_recording(self) -> bool:
        """Whether an episode is currently being recorded.

        Returns:
            bool: True if recording is active.
        """
        return self.task_env.is_recording()

    def close_data_collector(self):
        """Flush remaining data and close the underlying data collector."""
        self.task_env.close_data_collector()

    def save_and_close(self):
        """Save any remaining data and close the data collector."""
        self.close_data_collector()

    def close(self):
        """Close the wrapper, saving any remaining data."""
        self.save_and_close()
    def get_current_task_info(self) -> Tuple[Dict[str, Any], bool]:
        """Return the current task info plus teleop demo-tracking fields.

        Returns:
            Tuple: ``(demo_info, is_complete)`` where ``is_complete`` reflects both the
            demo count and (if enabled) exhaustion of the pose schedule.
        """
        task_info = self.task_env.get_current_task_info()
        
        # Add teleoperation-specific demo tracking info
        demo_info = {
            **serialize_dict(task_info),  # Include task environment info (serialized)
            "demo_count": self.current_demo_count,
            "demos_per_asset": self.demos_per_asset,
            "total_demos_collected": self.total_demos_collected,
            "progress_percent": (self.total_demos_collected / (serialize_dict(task_info).get("total_objects", 1) * self.demos_per_asset)) * 100,
            "demos_remaining": self.demos_per_asset - self.current_demo_count,
        }
        
        # Check if data collection is complete
        # Consider both demo count AND pose schedule exhaustion
        is_complete = self.current_demo_count >= self.demos_per_asset
        
        # Also check if pose schedule is exhausted (if enabled)
        if hasattr(self.task_env, 'get_pose_schedule_info'):
            pose_info = self.task_env.get_pose_schedule_info()
            if pose_info.get("enabled", False) and pose_info.get("is_exhausted", False):
                is_complete = True
        
        return demo_info, is_complete
    
    def __getattr__(self, name):
        """Delegate unknown attribute access to the wrapped task environment.

        Args:
            name (str): Attribute name.

        Returns:
            The attribute resolved on ``self.task_env``.
        """
        return getattr(self.task_env, name)


class BimanualRPCServer:
    """RPC-facing controller exposing 14-dim bimanual joint commands over a task environment.

    The leader pushes a 14-dim command ``[left_arm(6), left_gripper(1),
    right_arm(6), right_gripper(1)]``; the server caches it and assembles the
    action tensor that drives the next env step. Optionally records via a
    :class:`TeleopManagerWrapper`.
    """

    def __init__(self, task_env: ManagerBasedRLEnv, teleop_wrapper: TeleopManagerWrapper = None):
        """Initialize the RPC server.

        Args:
            task_env (ManagerBasedRLEnv): The task environment to control.
            teleop_wrapper (TeleopManagerWrapper | None): Recorder wrapper; if None,
                steps go straight to the env and recording calls are no-ops.
        """
        self.task_env = task_env
        self.teleop_wrapper = teleop_wrapper
        
        # Command storage for bimanual control
        self.left_arm_command = None
        self.right_arm_command = None
        self.left_gripper_command = None
        self.right_gripper_command = None
        
    def get_left_joint_pos(self):
        """Read the left arm's six joint positions (arm DOF only, gripper excluded).

        Returns:
            np.ndarray: Joint positions, float32 array of shape (6,).
        """
        left_arm = self.task_env.scene["left_arm"]
        return left_arm.data.joint_pos.torch[0, :6].cpu().numpy()  # First 6 joints (arm only)

    def get_right_joint_pos(self):
        """Read the right arm's six joint positions (arm DOF only, gripper excluded).

        Returns:
            np.ndarray: Joint positions, float32 array of shape (6,).
        """
        right_arm = self.task_env.scene["right_arm"]
        return right_arm.data.joint_pos.torch[0, :6].cpu().numpy()  # First 6 joints (arm only)

    def get_left_gripper_pos(self):
        """Read the left gripper state as a binary value (0=open, 1=close).

        Derives the state by thresholding the raw finger joint position, since
        IsaacLab's BinaryJointPositionAction affects action OUTPUT only, not the
        joint position INPUT read here.

        Returns:
            float: 1.0 if closed, 0.0 if open.
        """
        left_arm = self.task_env.scene["left_arm"]
        # Get left finger position to determine gripper state
        left_finger_pos = left_arm.data.joint_pos.torch[0, YamActionLayout.ARM_GRIPPER_JOINT_INDEX].cpu().numpy()
        
        # Use YAM constants to determine open/close state
        # Binary action convention: 0=open, 1=close
        threshold = (ROBOT.finger_closed('left') + ROBOT.finger_open('left')) / 2  # 0.02
        return 1.0 if left_finger_pos < threshold else 0.0
    
    def get_right_gripper_pos(self):
        """Read the right gripper state as a binary value (0=open, 1=close).

        Derives the state by thresholding the raw finger joint position, since
        IsaacLab's BinaryJointPositionAction affects action OUTPUT only, not the
        joint position INPUT read here.

        Returns:
            float: 1.0 if closed, 0.0 if open.
        """
        right_arm = self.task_env.scene["right_arm"]
        # Get the finger position to determine gripper state
        right_finger_pos = right_arm.data.joint_pos.torch[0, YamActionLayout.ARM_GRIPPER_JOINT_INDEX].cpu().numpy()
        
        # Use YAM constants to determine open/close state
        # Binary action convention: 0=open, 1=close
        threshold = (ROBOT.finger_closed('right') + ROBOT.finger_open('right')) / 2  # 0.02
        return 1.0 if right_finger_pos < threshold else 0.0
    
    def command_bimanual_joint_pos(self, joint_pos):
        """Cache a 14-dim bimanual joint command for the next step.

        Args:
            joint_pos (Sequence[float]): Length-14 command
                ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.
                Gripper entries are absolute finger joint positions (continuous).

        Raises:
            ValueError: if ``joint_pos`` does not have length 14.
        """
        if len(joint_pos) != YamActionLayout.DIM:
            raise ValueError(f"Expected {YamActionLayout.DIM} DOF for bimanual control, got {len(joint_pos)}")

        # Set individual commands
        self.left_arm_command = joint_pos[YamActionLayout.LEFT_ARM]
        self.right_arm_command = joint_pos[YamActionLayout.RIGHT_ARM]
        # Gripper values are absolute finger joint positions (continuous control)
        self.left_gripper_command = float(joint_pos[YamActionLayout.LEFT_GRIPPER])
        self.right_gripper_command = float(joint_pos[YamActionLayout.RIGHT_GRIPPER])

    def get_combined_action(self):
        """Assemble the 14-dim bimanual action tensor from cached commands.

        Any command not yet set falls back to the arm/gripper's current measured state.

        Returns:
            torch.Tensor: Action of shape (1, 14),
            ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.
        """
        # Get current positions if no commands set
        if self.left_arm_command is None:
            self.left_arm_command = self.get_left_joint_pos()
        if self.right_arm_command is None:
            self.right_arm_command = self.get_right_joint_pos()
        if not hasattr(self, 'left_gripper_command') or self.left_gripper_command is None:
            self.left_gripper_command = self.get_left_gripper_pos()
        if not hasattr(self, 'right_gripper_command') or self.right_gripper_command is None:
            self.right_gripper_command = self.get_right_gripper_pos()

        # Assemble [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
        combined_action = torch.zeros(1, YamActionLayout.DIM, device=self.task_env.device)
        combined_action[0, YamActionLayout.LEFT_ARM] = torch.from_numpy(self.left_arm_command).float()
        combined_action[0, YamActionLayout.LEFT_GRIPPER] = self.left_gripper_command
        combined_action[0, YamActionLayout.RIGHT_ARM] = torch.from_numpy(self.right_arm_command).float()
        combined_action[0, YamActionLayout.RIGHT_GRIPPER] = self.right_gripper_command

        return combined_action
    
    def step_with_commands(self, metadata: Dict[str, Any] = None):
        """Step the environment using the currently cached commands.

        Args:
            metadata (Dict[str, Any] | None): Extra metadata to record with this step.

        Returns:
            Tuple: ``(obs_dict, reward, terminated, truncated, info)`` from the env step.
        """
        action = self.get_combined_action()

        if self.teleop_wrapper:
            return self.teleop_wrapper.step(action, metadata)
        else:
            return self.task_env.step(action)

    # Expose teleop wrapper methods if available; no-op fallbacks when unwrapped.
    def reset_task(self):
        """Discard the in-progress episode and reset the task.

        Returns:
            Tuple: ``(info, is_complete)``; when no recorder is attached, ``info`` is
            the env reset info and ``is_complete`` is False.
        """
        if self.teleop_wrapper:
            return self.teleop_wrapper.reset_task()
        else:
            obs_dict, info = self.task_env.reset()
            return info, False
    
    def start_recording(self):
        """Start a new recording episode, or no-op if unwrapped.

        Returns:
            bool: True if recording started, else False.
        """
        if self.teleop_wrapper:
            return self.teleop_wrapper.start_recording()
        return False

    def save_trajectory(self):
        """Save the current trajectory, or no-op if unwrapped.

        Returns:
            bool: True if a trajectory was saved, else False.
        """
        if self.teleop_wrapper:
            return self.teleop_wrapper.save_trajectory()
        return False

    def is_recording(self):
        """Whether an episode is currently being recorded.

        Returns:
            bool: True if recording is active (always False when unwrapped).
        """
        if self.teleop_wrapper:
            return self.teleop_wrapper.is_recording()
        return False

    def is_data_collection_complete(self):
        """Whether the target demo count has been reached.

        Returns:
            bool: True if collection is complete (always False when unwrapped).
        """
        if self.teleop_wrapper:
            return self.teleop_wrapper.is_data_collection_complete()
        return False

    def close_data_collector(self):
        """Flush and close the data collector, or no-op if unwrapped."""
        if self.teleop_wrapper:
            self.teleop_wrapper.close_data_collector()

    def get_current_task_info(self):
        """Return current task info as an ``(info, is_complete)`` tuple.

        Returns:
            Tuple: ``(task_info, is_complete)``; ``is_complete`` is always False when
            no recorder is attached.
        """
        if self.teleop_wrapper:
            return self.teleop_wrapper.get_current_task_info()
        else:
            # Return tuple format for consistency
            task_info = self.task_env.get_current_task_info()
            return task_info, False

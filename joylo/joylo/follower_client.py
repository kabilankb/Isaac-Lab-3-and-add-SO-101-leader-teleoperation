"""RPC client for the bimanual follower (IsaacLab sim) used by JoyLo teleoperation."""

import numpy as np
import portal


DEFAULT_PORT = 11333


class BimanualFollowerClient:
    """RPC client for connecting to bimanual follower (IsaacLab sim)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self._client = portal.Client(f"{host}:{port}")
        print(f"[INFO] Connecting to bimanual follower at {host}:{port}...")

    def num_dofs(self) -> int:
        """Get the number of DOFs reported by the follower.

        Returns:
            int: Number of degrees of freedom (14 for the bimanual setup).
        """
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions from the follower.

        Returns:
            np.ndarray: float, shape (14,) ordered as
                [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
        """
        return self._client.get_joint_pos().result()

    def command_bimanual_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Send a joint position command to the follower.

        Args:
            joint_pos (np.ndarray): float, shape (14,) ordered as
                [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
        """
        self._client.command_bimanual_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: dict) -> None:
        """Send a joint state command (dict with a "pos" key) to the follower."""
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> dict:
        """Get observations from follower."""
        return self._client.get_observations().result()

    # Data collection methods
    def start_recording(self) -> bool:
        """Start recording trajectory."""
        return self._client.start_recording().result()

    def save_trajectory(self) -> bool:
        """Save current trajectory."""
        return self._client.save_trajectory().result()

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._client.is_recording().result()

    def reset_task(self) -> tuple:
        """Reset task."""
        return self._client.reset_task().result()

    def get_task_info(self) -> tuple:
        """Get current task information."""
        return self._client.get_task_info().result()

    def close_data_collector(self):
        """Close data collector."""
        self._client.close_data_collector()

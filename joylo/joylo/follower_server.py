"""RPC server wrapper and render optimizations for the bimanual YAM follower; import only after AppLauncher has launched the IsaacLab app."""

import threading
from typing import Optional

import numpy as np
import portal

from yamlab.teleoperation.teleop_server import BimanualRPCServer
from yamlab.teleoperation import TeleopManagerWrapper


class PortalSimServer:
    """Binds the BimanualRPCServer API over RPC (single server for both arms).

    RPC handlers only set flags for state-modifying operations. The main simulation
    loop checks these flags and performs the actual operations synchronously. This
    prevents race conditions from concurrent PhysX access.
    """

    def __init__(self, rpc_server: BimanualRPCServer, port: int,
                 teleop_wrapper: Optional[TeleopManagerWrapper] = None):
        self._rpc_server = rpc_server
        self._port = port
        self._teleop_wrapper = teleop_wrapper
        self._server = portal.Server(port)

        # Request flags for state-modifying operations (to be processed by main loop)
        # Using threading.Event for thread-safe flag communication
        self._reset_task_requested = threading.Event()
        self._save_trajectory_requested = threading.Event()
        self._start_recording_requested = threading.Event()
        self._close_data_collector_requested = threading.Event()

        # Results from operations (set by main loop, read by RPC handlers)
        self._last_reset_result = (({}, {}), False)  # (task_info, success)
        self._last_save_result = False
        self._last_task_info = ({},)

        # Lock for result access
        self._result_lock = threading.Lock()

        # Event to signal operation completion
        self._operation_complete = threading.Event()

        # Bind bimanual methods (14 DOF: 7 per arm)
        self._server.bind("num_dofs", lambda: 14)
        self._server.bind("get_joint_pos", self._get_bimanual_joint_pos)
        self._server.bind("command_bimanual_joint_pos", self._command_bimanual_joint_pos)
        self._server.bind("command_joint_state", self._command_joint_state)
        self._server.bind("get_observations", self._get_observations)

        # Bind teleop wrapper methods if available
        # State-modifying operations use request/response pattern
        if self._teleop_wrapper is not None:
            self._server.bind("reset_task", self._request_reset_task)
            self._server.bind("get_task_info", self._request_get_task_info)
            self._server.bind("start_recording", self._request_start_recording)
            self._server.bind("save_trajectory", self._request_save_trajectory)
            self._server.bind("is_recording", self._teleop_wrapper.is_recording)  # Read-only, safe
            self._server.bind("close_data_collector", self._request_close_data_collector)

    def _request_reset_task(self, *args, **kwargs):
        """Request reset task - sets flag for main loop to process."""
        self._operation_complete.clear()
        self._reset_task_requested.set()
        # Wait for main loop to complete the operation (with timeout)
        if not self._operation_complete.wait(timeout=10.0):
            print("[WARNING] reset_task request timed out")
            return ({}, {}), False
        with self._result_lock:
            return self._last_reset_result

    def _request_get_task_info(self, *args, **kwargs):
        """Get task info - safe to call directly as it's read-only during step pause."""
        # Task info is read-only, but we still need to ensure we're not mid-step
        # For simplicity, we can call it directly since it just reads state
        try:
            return self._teleop_wrapper.get_current_task_info(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] Error in get_task_info: {e}")
            return ({},)

    def _request_start_recording(self, *args, **kwargs):
        """Request start recording - safe to call directly as it doesn't modify physics."""
        try:
            return self._teleop_wrapper.start_recording(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] Error in start_recording: {e}")
            return False

    def _request_save_trajectory(self, *args, **kwargs):
        """Request save trajectory - sets flag for main loop to process."""
        self._operation_complete.clear()
        self._save_trajectory_requested.set()
        # Wait for main loop to complete the operation (with timeout)
        if not self._operation_complete.wait(timeout=10.0):
            print("[WARNING] save_trajectory request timed out")
            return False
        with self._result_lock:
            return self._last_save_result

    def _request_close_data_collector(self, *args, **kwargs):
        """Request close data collector - sets flag for main loop to process."""
        self._operation_complete.clear()
        self._close_data_collector_requested.set()
        # Wait for main loop to complete the operation (with timeout)
        self._operation_complete.wait(timeout=10.0)
        return None

    def process_pending_requests(self) -> bool:
        """Process any pending state-modifying RPC requests, called from the main loop.

        Returns:
            bool: True if any request was processed (the caller should skip stepping
                the simulation this iteration).
        """
        processed = False

        # Process reset_task request
        if self._reset_task_requested.is_set():
            self._reset_task_requested.clear()
            try:
                result = self._teleop_wrapper.reset_task()
                with self._result_lock:
                    self._last_reset_result = result
            except Exception as e:
                print(f"[ERROR] Error in reset_task: {e}")
                import traceback
                traceback.print_exc()
                with self._result_lock:
                    self._last_reset_result = (({}, {}), False)
            self._operation_complete.set()
            processed = True

        # Process save_trajectory request
        if self._save_trajectory_requested.is_set():
            self._save_trajectory_requested.clear()
            try:
                result = self._teleop_wrapper.save_trajectory()
                with self._result_lock:
                    self._last_save_result = result
            except Exception as e:
                print(f"[ERROR] Error in save_trajectory: {e}")
                import traceback
                traceback.print_exc()
                with self._result_lock:
                    self._last_save_result = False
            self._operation_complete.set()
            processed = True

        # Process close_data_collector request
        if self._close_data_collector_requested.is_set():
            self._close_data_collector_requested.clear()
            try:
                self._teleop_wrapper.close_data_collector()
            except Exception as e:
                print(f"[ERROR] Error in close_data_collector: {e}")
                import traceback
                traceback.print_exc()
            self._operation_complete.set()
            processed = True

        return processed

    def _get_bimanual_joint_pos(self) -> np.ndarray:
        """Get bimanual joint positions over RPC.

        Returns:
            np.ndarray: float, shape (14,) ordered as
                [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
        """
        left_arm_pos = self._rpc_server.get_left_joint_pos()  # 6 DOF
        right_arm_pos = self._rpc_server.get_right_joint_pos()  # 6 DOF
        left_gripper_pos = self._rpc_server.get_left_gripper_pos()  # 1 DOF
        right_gripper_pos = self._rpc_server.get_right_gripper_pos()  # 1 DOF

        # Combine into 14 DOF array: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        return np.concatenate([
            left_arm_pos,      # 6 DOF
            [left_gripper_pos], # 1 DOF
            right_arm_pos,     # 6 DOF
            [right_gripper_pos] # 1 DOF
        ])

    def _command_bimanual_joint_pos(self, joint_pos: np.ndarray):
        """Command bimanual joint positions over RPC.

        Args:
            joint_pos (np.ndarray): float, shape (14,) ordered as
                [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
                Gripper entries are absolute finger joint positions.
        """
        assert len(joint_pos) == 14, f"Expected 14 DOF, got {len(joint_pos)}"
        self._rpc_server.command_bimanual_joint_pos(joint_pos) # gripper: absolute finger joint position

    def _command_joint_state(self, joint_state: dict):
        """Command joint state over RPC, accepting a dict with a "pos" key or a raw array.

        Args:
            joint_state (dict): Either {"pos": np.ndarray of shape (14,)} or a
                bare np.ndarray of shape (14,).
        """
        if "pos" in joint_state:
            self._command_bimanual_joint_pos(joint_state["pos"])
        elif isinstance(joint_state, np.ndarray):
            self._command_bimanual_joint_pos(joint_state)

    def _get_observations(self) -> dict:
        """Get proprioceptive observations for both arms over RPC.

        Returns:
            dict: Per-arm and combined joint positions/velocities. "joint_pos" and
                "joint_vel" are float arrays of shape (14,); the per-arm entries are
                shape (7,).
        """
        left_pos = self._rpc_server.get_left_joint_pos()
        right_pos = self._rpc_server.get_right_joint_pos()

        # Get velocities
        left_arm = self._rpc_server.task_env.scene["left_arm"]
        right_arm = self._rpc_server.task_env.scene["right_arm"]
        left_vel = left_arm.data.joint_vel[0, :7].cpu().numpy()
        right_vel = right_arm.data.joint_vel[0, :7].cpu().numpy()

        return {
            "joint_pos": np.concatenate([left_pos, right_pos]),
            "joint_vel": np.concatenate([left_vel, right_vel]),
            "left_joint_pos": left_pos,
            "right_joint_pos": right_pos,
            "left_joint_vel": left_vel,
            "right_joint_vel": right_vel,
        }

    def start_server_thread(self) -> threading.Thread:
        """Start the RPC server in a background daemon thread.

        Returns:
            threading.Thread: The started server thread.
        """
        def server_thread():
            print(f"[INFO] BimanualPortalSimServer starting on port {self._port}...")
            self._server.start()

        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        return thread


def apply_teleop_render_optimizations(task_env):
    """Configure the Omniverse renderer for low-latency teleoperation.

    Teleoperation records only simulation state (joint/object poses), never rendered
    images, so the GPU render does not need to complete synchronously within the
    control loop. This applies Kit/RTX settings so that physics stepping is not
    blocked by the viewport render. Each setting is applied independently; if a
    settings path is unavailable on the running Kit version it is skipped without
    aborting the rest. The settings only have a visible effect when a GUI/viewport
    is present.

    Args:
        task_env: The teleoperation environment (a ``ManagerBasedRLEnv`` subclass).
            Its ``sim`` is used to issue the intermediate render required by the
            async-rendering enable sequence.

    Returns:
        None.
    """
    try:
        import carb
    except ImportError:
        return

    settings = carb.settings.get_settings()

    def _set_bool(path, value):
        try:
            settings.set_bool(path, value)
        except Exception:
            pass

    def _set_int(path, value):
        try:
            settings.set_int(path, value)
        except Exception:
            pass

    # Remove the per-loop frame-rate cap.
    for run_loop in ("present", "main", "rendering_0"):
        _set_bool(f"/app/runLoops/{run_loop}/rateLimitEnabled", False)

    # Select DLSS performance mode.
    _set_int("/rtx/post/dlss/execMode", 0)

    # Render the viewport asynchronously to the control loop. The flags are toggled
    # (on, one render, off, on) because enabling them in a single step can leave the
    # viewport frozen until the next render.
    try:
        settings.set_bool("/app/asyncRendering", True)
        settings.set_bool("/app/asyncRenderingLowLatency", True)
        task_env.sim.render()
        settings.set_bool("/app/asyncRendering", False)
        settings.set_bool("/app/asyncRenderingLowLatency", False)
        settings.set_bool("/app/asyncRendering", True)
        settings.set_bool("/app/asyncRenderingLowLatency", True)
    except Exception:
        pass

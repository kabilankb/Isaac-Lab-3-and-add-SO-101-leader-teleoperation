"""
JoyLo agent for YAM bimanual teleoperation.

This module provides the agent classes for reading JoyLo hardware (Dynamixel motors)
and mapping joint positions from hardware to simulation coordinates.
"""

import numpy as np
import yaml
import time
from typing import Dict, Any, Optional

from joylo.dynamixel.driver import DynamixelDriver, OperatingMode
from pyjoycon import get_L_id, get_R_id
from joylo.joycon.rumble import RumbleJoyCon
from joylo.agents import GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, GRIPPER_DELTA_PER_STEP

class YAMGelloAgent:
    """
    Agent for controlling YAM robot arms via Dynamixel motors (JoyLo hardware).
    
    This agent:
    - Reads joint positions from Dynamixel motors
    - Applies calibration (offsets and signs)
    - Maps between hardware joint space and simulation joint space
    - Handles gripper control
    """
    
    def __init__(
        self,
        port: str,
        joint_config_file: str,
        num_motors: int = 12,  # YAM bimanual: 6 joints per arm * 2 arms
        start_joints: Optional[np.ndarray] = None,
        damping_motor_kp: float = 0.3,
    ):
        """Initialize the YAM GELLO agent and open the Dynamixel connection.

        Args:
            port (str): Serial port for the JoyLo connection (e.g., "/dev/ttyUSB0").
            joint_config_file (str): Path to the joint calibration YAML file.
            num_motors (int): Total number of motors (12 for bimanual YAM).
            start_joints (np.ndarray): Initial joint positions in radians, shape
                (num_motors,). Defaults to zeros.
            damping_motor_kp (float): Motor damping gain.
        """
        self.port = port
        self.num_motors = num_motors
        self.num_joints_per_arm = num_motors // 2  # 6 joints per arm
        self.damping_motor_kp = damping_motor_kp
        
        # Load joint configuration
        self.joint_config = self._load_joint_config(joint_config_file)
        self.joint_offsets = np.deg2rad(np.array(self.joint_config['joints']['offsets']))
        self.joint_signs = np.array(self.joint_config['joints']['signs'])
        
        # Initialize Dynamixel driver
        joint_ids = list(range(num_motors))
        self.driver = DynamixelDriver(ids=joint_ids, port=port, baudrate=2000000)
        
        # Set start joints
        if start_joints is None:
            start_joints = np.zeros(num_motors)
        self.start_joints = start_joints
        
        # Warm up driver
        for _ in range(10):
            self.driver.get_joints()
        
        print(f"[INFO] YAMGelloAgent initialized:")
        print(f"[INFO] - Port: {port}")
        print(f"[INFO] - Motors: {num_motors} ({self.num_joints_per_arm} per arm)")
        print(f"[INFO] - Joint config: {joint_config_file}")
    
    def _load_joint_config(self, config_file: str) -> Dict[str, Any]:
        """Load the joint calibration configuration from a YAML file.

        Args:
            config_file (str): Path to the joint calibration YAML file.

        Returns:
            Dict[str, Any]: Parsed configuration (joint offsets and signs).
        """
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config
    
    def get_joints(self) -> np.ndarray:
        """Read calibrated joint positions from the Dynamixel hardware.

        Applies the per-joint sign/offset calibration and wraps angles to [-pi, pi].

        Returns:
            np.ndarray: float, shape (num_motors,) joint positions in radians
                (12 for bimanual YAM).
        """
        # Read raw joint positions from Dynamixel motors
        raw_joints = self.driver.get_joints()
        
        # Apply calibration: joint_pos = sign * (raw_pos - offset)
        calibrated_joints = self.joint_signs * (raw_joints - self.joint_offsets)
        
        # Apply angle wrapping to handle 360-degree offsets properly
        calibrated_joints = self._wrap_angles(calibrated_joints)
        
        return calibrated_joints
    
    def _wrap_angles(self, angles: np.ndarray) -> np.ndarray:
        """Wrap angles to the [-pi, pi] range, accounting for 360-degree offsets.

        This prevents issues where 359° and 361° are treated as 358° apart
        instead of 2° apart when a joint uses a 360° calibration offset.

        Args:
            angles (np.ndarray): float, shape (num_motors,) angles in radians.

        Returns:
            np.ndarray: float, shape (num_motors,) angles wrapped to [-pi, pi].
        """
        # For joints with 360° (2π) offsets, we need special handling
        # Check which joints have 360° offsets
        offset_360_mask = np.abs(self.joint_offsets - 2*np.pi) < 0.1  # 360° = 2π radians
        
        if np.any(offset_360_mask):
            # For joints with 360° offsets, use a different wrapping strategy
            wrapped_angles = angles.copy()
            
            # For 360° offset joints, wrap to [0, 2π] first, then to [-π, π]
            for i, has_360_offset in enumerate(offset_360_mask):
                if has_360_offset:
                    # Wrap to [0, 2π] range first
                    angle = angles[i]
                    angle = angle % (2 * np.pi)
                    if angle > np.pi:
                        angle -= 2 * np.pi
                    wrapped_angles[i] = angle
                else:
                    # Standard wrapping for other joints
                    wrapped_angles[i] = np.arctan2(np.sin(angles[i]), np.cos(angles[i]))
        else:
            # Standard wrapping for all joints
            wrapped_angles = np.arctan2(np.sin(angles), np.cos(angles))
        
        return wrapped_angles
    
    def get_left_arm_joints(self) -> np.ndarray:
        """Get left arm joint positions.

        Returns:
            np.ndarray: float, shape (num_joints_per_arm,) in radians (6 for YAM).
        """
        joints = self.get_joints()
        return joints[:self.num_joints_per_arm]

    def get_right_arm_joints(self) -> np.ndarray:
        """Get right arm joint positions.

        Returns:
            np.ndarray: float, shape (num_joints_per_arm,) in radians (6 for YAM).
        """
        joints = self.get_joints()
        return joints[self.num_joints_per_arm:]
    
    def set_joints(self, joint_angles: np.ndarray) -> None:
        """Command joint angles on the physical JoyLo hardware.

        Converts from simulation joint space to hardware space before sending.

        Args:
            joint_angles (np.ndarray): float, shape (num_motors,) angles in radians,
                ordered as left_arm(6) + right_arm(6).
        """
        if len(joint_angles) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} joint angles, got {len(joint_angles)}")
        
        # Apply calibration (convert from simulation space to hardware space)
        # Hardware angles = simulation_angles * signs + offsets
        hardware_angles = joint_angles * self.joint_signs + self.joint_offsets
        
        # Send to physical hardware (hardware_angles are in radians)
        self.driver.set_joints(hardware_angles.tolist())
    
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Get the action (joint positions) from the current hardware state.

        Args:
            obs (Dict[str, Any]): Observation dictionary (unused in open-loop mode).

        Returns:
            np.ndarray: float, shape (num_motors,) joint positions for both arms.
        """
        return self.get_joints()
    
    def reset(self):
        """Reset agent to initial state."""
        # Warm up driver
        for _ in range(5):
            self.driver.get_joints()
    
    def start(self):
        """Start agent (initialization)."""
        # Set operating mode to POSITION for all joints
        self.driver.set_operating_mode(OperatingMode.POSITION)
        
        # Start with torque disabled for free movement (teleoperation mode)
        self.driver.set_torque_mode(False)
        
        # Warm up driver
        for _ in range(5):
            self.get_joints()
            time.sleep(0.1)
        print(f"[INFO] YAMGelloAgent started on port {self.port}")
        print(f"[INFO] Motors have torque DISABLED - arms can move freely")


class JoyConAgent:
    """
    Agent for controlling base movement, trunk, and grippers via Nintendo JoyCon controllers.
    
    Note: For YAM bimanual setup, we don't have base/trunk movement,
    so this agent primarily handles gripper control via JoyCon triggers.
    """
    
    def __init__(
        self,
        max_gripper_cooldown: int = 10,  # Frames to wait before allowing another toggle
        max_button_cooldown: int = 20,   # Frames to wait before allowing another button press
    ):
        """Initialize the JoyCon agent and connect to the left/right controllers.

        Args:
            max_gripper_cooldown (int): Frames to wait between gripper toggles.
            max_button_cooldown (int): Frames to wait between button presses.
        """
        self.max_gripper_cooldown = max_gripper_cooldown
        self.max_button_cooldown = max_button_cooldown
        
        # Gripper state tracking (absolute finger joint positions, start fully open)
        self.gripper_info = {
            "left": {"pos": -0.0475},   # -0.0475 = open, 0.0 = closed
            "right": {"pos": -0.0475},
        }
        
        # Button cooldown tracking
        self.button_cooldowns = {
            "x": 0,
            "home": 0,
            "l": 0,
            "r": 0,
            "y": 0,
            "b": 0,
            "a": 0,
            "capture": 0,
            "left_arrow": 0,
            "right_arrow": 0,
            "up": 0,
            "down": 0,
            "minus": 0,
            "plus": 0,
        }
        
        # Button state tracking
        self.button_states = {
            "x": False,
            "home": False,
            "l": False,
            "r": False,
            "y": False,
            "b": False,
            "a": False,
            "capture": False,
            "left_arrow": False,
            "right_arrow": False,
            "up": False,
            "down": False,
            "minus": False,
            "plus": False,
        }
        
        # Try to connect to JoyCons with retry
        jc_id_left = get_L_id()
        jc_id_right = get_R_id()
        assert jc_id_left[0] is not None, "Failed to connect to Left JoyCon!"
        assert jc_id_right[0] is not None, "Failed to connect to Right JoyCon!"
        
        self.jc_left = self._create_joycon_with_retry(jc_id_left, "Left")
        self.jc_right = self._create_joycon_with_retry(jc_id_right, "Right")
        print(f"[INFO] JoyCons connected successfully")
    
    def _create_joycon_with_retry(
        self, 
        jc_id: tuple, 
        name: str, 
        max_retries: int = 10, 
        retry_delay: float = 0.5
    ) -> RumbleJoyCon:
        """Create a RumbleJoyCon, retrying to absorb transient Bluetooth failures.

        The JoyCon Bluetooth connection can fail if a previous connection was not
        closed cleanly, the device needs time to reset, or SPI flash reads get out
        of sync; retrying with a delay typically recovers.

        Args:
            jc_id (tuple): (vendor_id, product_id, serial) from get_L_id/get_R_id.
            name (str): Name used for logging (e.g., "Left" or "Right").
            max_retries (int): Maximum number of connection attempts.
            retry_delay (float): Delay in seconds between attempts.

        Returns:
            RumbleJoyCon: The connected JoyCon instance.

        Raises:
            RuntimeError: If the connection fails after all retries.
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Connecting to {name} JoyCon (attempt {attempt + 1}/{max_retries})...")
                joycon = RumbleJoyCon(*jc_id)
                print(f"[INFO] {name} JoyCon connected successfully")
                
                # Small delay to stabilize connection
                time.sleep(0.1)
                return joycon
                
            except (AssertionError, OSError, Exception) as e:
                last_error = e
                print(f"[WARN] {name} JoyCon connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Wait before retrying to let the device reset
                if attempt < max_retries - 1:
                    print(f"[INFO] Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
        
        # All retries failed
        raise RuntimeError(
            f"Failed to connect to {name} JoyCon after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def get_gripper_commands(self) -> tuple[float, float]:
        """Compute gripper commands from the JoyCon trigger buttons.

        Continuous control: hold ZL/ZR to close, hold L/R to open; with no button
        held the current position is retained. The internal position is updated by
        a fixed delta per call and clamped to the open/closed range.

        Returns:
            tuple: (left_gripper, right_gripper) absolute finger joint positions
                in radians.
        """
        # Read buttons: ZL/ZR = close, L/R = open
        left_close = self.jc_left.get_button_zl()
        right_close = self.jc_right.get_button_zr()
        left_open = self.jc_left.get_button_l()
        right_open = self.jc_right.get_button_r()

        # Update left gripper position (only when button held)
        if left_close:
            self.gripper_info["left"]["pos"] += GRIPPER_DELTA_PER_STEP  # toward 0 (closed)
        elif left_open:
            self.gripper_info["left"]["pos"] -= GRIPPER_DELTA_PER_STEP  # toward -0.0475 (open)

        # Update right gripper position (only when button held)
        if right_close:
            self.gripper_info["right"]["pos"] += GRIPPER_DELTA_PER_STEP
        elif right_open:
            self.gripper_info["right"]["pos"] -= GRIPPER_DELTA_PER_STEP

        # Clamp to valid range
        self.gripper_info["left"]["pos"] = float(np.clip(
            self.gripper_info["left"]["pos"], GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS
        ))
        self.gripper_info["right"]["pos"] = float(np.clip(
            self.gripper_info["right"]["pos"], GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS
        ))

        return self.gripper_info["left"]["pos"], self.gripper_info["right"]["pos"]
    
    def get_button_states(self) -> dict:
        """Read the pressed/released state of all tracked JoyCon buttons.

        Returns:
            dict: Mapping from button name to bool (True if currently pressed).
        """
        # Read all button states
        states = {
            "x": self.jc_right.get_button_x(),
            "y": self.jc_right.get_button_y(),
            "b": self.jc_right.get_button_b(),
            "a": self.jc_right.get_button_a(),
            "home": self.jc_right.get_button_home(),
            "capture": self.jc_left.get_button_capture(),
            "l": self.jc_left.get_button_l(),
            "r": self.jc_right.get_button_r(),
            "minus": self.jc_left.get_button_minus(),
            "plus": self.jc_right.get_button_plus(),
            "left_arrow": self.jc_left.get_button_left(),
            "right_arrow": self.jc_left.get_button_right(),
            "up": self.jc_left.get_button_up(),
            "down": self.jc_left.get_button_down(),
        }
        return states
    
    def is_button_ready(self, button_name: str) -> bool:
        """Check whether a button is ready to be pressed (not in cooldown).

        Args:
            button_name (str): Name of the button to check.

        Returns:
            bool: True if the button is ready, False if still in cooldown.
        """
        return self.button_cooldowns.get(button_name, 0) == 0
    
    def set_button_cooldown(self, button_name: str):
        """Start the cooldown for a button after it has been pressed.

        Args:
            button_name (str): Name of the button to set the cooldown for.
        """
        if button_name in self.button_cooldowns:
            self.button_cooldowns[button_name] = self.max_button_cooldown
    
    def update_button_cooldowns(self):
        """Update all button cooldowns (decrement by 1)."""
        for button in self.button_cooldowns:
            self.button_cooldowns[button] = max(self.button_cooldowns[button] - 1, 0)
    
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Get the gripper commands as an action.

        Args:
            obs (Dict[str, Any]): Observation dictionary (unused).

        Returns:
            np.ndarray: float, shape (2,) gripper commands [left, right].
        """
        left_grip, right_grip = self.get_gripper_commands()
        return np.array([left_grip, right_grip])
    
    def reset(self):
        """Reset agent state."""
        self.gripper_info["left"]["pos"] = GRIPPER_OPEN_POS
        self.gripper_info["right"]["pos"] = GRIPPER_OPEN_POS

        # Reset all button cooldowns
        for button in self.button_cooldowns:
            self.button_cooldowns[button] = 0

    def reset_grippers(self):
        """Reset grippers to fully open position."""
        self.gripper_info["left"]["pos"] = GRIPPER_OPEN_POS
        self.gripper_info["right"]["pos"] = GRIPPER_OPEN_POS
    
    def start(self):
        """Start agent."""
        pass


class MultiControllerAgent:
    """
    Combines multiple agents (arm control + gripper control).
    
    For YAM bimanual setup:
    - YAMGelloAgent: Reads arm joint positions from JoyLo hardware
    - JoyConAgent: Reads gripper commands from JoyCon controllers
    """
    
    def __init__(self, agents: list):
        """Initialize the multi-controller agent.

        Args:
            agents (list): List of agent instances. By convention, the first is the
                arm-control agent and the second is the gripper-control agent.
        """
        self.agents = agents
    
    def act(self, obs: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Get the combined action from all agents.

        Args:
            obs (Dict[str, Any]): Observation dictionary passed to each agent.

        Returns:
            tuple: (arm_joints, gripper_commands) where arm_joints is float, shape
                (12,) ordered [left_arm(6), right_arm(6)] and gripper_commands is
                float, shape (2,) [left_gripper, right_gripper].
        """
        actions = [agent.act(obs) for agent in self.agents]
        
        # Assume first agent is arm control, second is gripper control
        if len(actions) == 2:
            arm_joints = actions[0]  # 12 DOF from YAMGelloAgent
            gripper_commands = actions[1]  # 2 values from JoyConAgent
            return arm_joints, gripper_commands
        else:
            # Only arm control available
            arm_joints = actions[0]
            gripper_commands = np.array([0.0, 0.0])
            return arm_joints, gripper_commands
    
    def reset(self):
        """Reset all agents."""
        for agent in self.agents:
            agent.reset()
    
    def start(self):
        """Start all agents."""
        for agent in self.agents:
            agent.start()


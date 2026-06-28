"""
JoyLo Teleoperation for YAM Bimanual Setup.

This script launches JoyLo hardware (Dynamixel motors + JoyCons) and connects to
the IsaacLab follower simulation via RPC.

JoyLo Control:
- Arm joint positions from Dynamixel motors (12 DOF: 6 per arm)
- Gripper commands from Nintendo JoyCon controllers (2 DOF: 1 per gripper)

Button Controls:
- ZL/ZR: Hold to close left/right gripper (continuous)
- L/R: Hold to open left/right gripper (continuous)
- X (1st press): Start recording trajectory
- X (2nd press): Save trajectory and stop recording
- HOME: Reset task and align JoyLo to randomized sim pose
- Other buttons: Print notification (no functionality yet)

Run:
  python launch_joylo.py --joylo_port /dev/ttyUSB0 --joint_config ../configs/joint_config_yam_gello.yaml --enable_recording
"""

import argparse
import sys
import time
from pathlib import Path

# Add the repo root to the path so the joylo and simulation packages import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from joylo.agents import YAMGelloAgent, JoyConAgent
from joylo.follower_client import DEFAULT_PORT, BimanualFollowerClient
from joylo.teleop_controller import JoyLoTeleopController
from joylo.terminal_ui import print_controls


def main():
    """Initialize JoyLo hardware, connect to the follower, and run the teleoperation loop."""
    parser = argparse.ArgumentParser(description="JoyLo Teleoperation for YAM Bimanual Setup")

    # JoyLo hardware configuration
    parser.add_argument("--joylo_port", type=str, required=True,
                        help="Serial port for JoyLo connection (e.g., /dev/ttyUSB0)")
    parser.add_argument("--joint_config", type=str, required=True,
                        help="Path to joint configuration YAML file")

    # Follower connection configuration
    parser.add_argument("--server_host", type=str, default="127.0.0.1",
                        help="Host address for follower RPC server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port for bimanual follower RPC server")

    # Control configuration
    parser.add_argument("--control_frequency", type=float, default=30.0,
                        help="Control loop frequency in Hz")
    parser.add_argument("--enable_recording", action="store_true", default=False,
                        help="Enable trajectory recording")

    args = parser.parse_args()

    # Validate joint config file
    joint_config_path = Path(args.joint_config)
    if not joint_config_path.exists():
        print(f"[ERROR] Joint config file not found: {args.joint_config}")
        print(f"[INFO] Please run the JoyLo calibration script first to generate this file")
        sys.exit(1)

    print(f"[INFO] Starting JoyLo teleoperation for YAM bimanual setup")
    print(f"[INFO] - JoyLo port: {args.joylo_port}")
    print(f"[INFO] - Joint config: {args.joint_config}")

    # Initialize JoyLo hardware agents
    print(f"\n[INFO] Initializing JoyLo hardware agents...")

    # Create arm control agent (YAMGelloAgent)
    arm_agent = YAMGelloAgent(
        port=args.joylo_port,
        joint_config_file=args.joint_config,
        num_motors=12,  # 6 per arm
    )

    # Create gripper/button control agent (JoyConAgent)
    joycon_agent = JoyConAgent()

    # Start agents
    arm_agent.start()
    joycon_agent.start()

    # Connect to follower RPC server
    print(f"\n[INFO] Connecting to bimanual follower RPC server...")
    follower = BimanualFollowerClient(host=args.server_host, port=args.port)
    dofs = follower.num_dofs()
    print(f"[INFO] Connected to bimanual follower. DOFs: {dofs}")

    # Get initial state
    follower_pos = follower.get_joint_pos()

    print(f"[INFO] Initial follower state:")
    print(f"[INFO] - Bimanual follower (14-DoF): {follower_pos}")
    print(f"[INFO]   - Left arm (7-DoF):  {follower_pos[:7]}")
    print(f"[INFO]   - Right arm (7-DoF): {follower_pos[7:14]}")

    # Create teleoperation controller
    controller = JoyLoTeleopController(
        arm_agent=arm_agent,
        joycon_agent=joycon_agent,
        follower=follower,
        enable_recording=args.enable_recording,
        control_frequency=args.control_frequency,
    )

    # Print control instructions
    print_controls()

    print(f"\n[INFO] Starting teleoperation loop...")
    print(f"[INFO] Press Ctrl+C to exit\n")

    try:
        last_time = time.time()

        while not controller.exit_requested:
            # Handle button presses
            controller.handle_button_presses()

            # Execute teleoperation step
            controller.run_teleoperation_step()

            # Maintain control frequency
            elapsed = time.time() - last_time
            sleep_time = max(0, controller.control_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed - shutting down teleoperation...")

    finally:
        # Cleanup
        controller.cleanup()
        print("[INFO] JoyLo teleoperation shutdown complete")


if __name__ == "__main__":
    main()

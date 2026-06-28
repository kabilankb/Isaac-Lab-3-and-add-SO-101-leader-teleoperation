"""
Script for automatically calibrating a JoyLo set for YAM bimanual robot.

This script calibrates the joint signs and offsets for a YAM bimanual setup.
The YAM robot has 6 DOF per arm (joint1-6) plus gripper fingers, for a total of 12 arm joints.

Reference positions are defined as global variables at the top of the file.
"""

import sys
import numpy as np
import yaml
import tyro
from dataclasses import dataclass
from pathlib import Path

# Add the project root to the path to import gello modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from joylo.dynamixel.driver import DynamixelDriver

# Global reference positions for YAM bimanual calibration
# YAM has 6 DOF per arm: joint1-6 for each arm (left and right)
# Total: 12 arm joints (excluding gripper fingers)

# Left arm reference positions (6 DOF)
LEFT_REFERENCE_JOINT_POS_1 = np.array([
    0.0,      # joint1
    0.0,      # joint2 
    0.0,      # joint3
    0.0,      # joint4
    0.0,      # joint5
    90.0,     # joint6
])

LEFT_REFERENCE_JOINT_POS_2 = np.array([
    90.0,     # joint1
    90.0,     # joint2
    90.0,     # joint3
    90.0,     # joint4
    90.0,     # joint5
    0.0,      # joint6
])

# Right arm reference positions (same as left arm)
RIGHT_REFERENCE_JOINT_POS_1 = LEFT_REFERENCE_JOINT_POS_1.copy()
RIGHT_REFERENCE_JOINT_POS_2 = LEFT_REFERENCE_JOINT_POS_2.copy()

# Combined reference positions for the full bimanual system (12 DOF)
REFERENCE_JOINT_POS_1 = np.concatenate([LEFT_REFERENCE_JOINT_POS_1, RIGHT_REFERENCE_JOINT_POS_1])
REFERENCE_JOINT_POS_2 = np.concatenate([LEFT_REFERENCE_JOINT_POS_2, RIGHT_REFERENCE_JOINT_POS_2])

CONFIG_DIR = project_root / "configs"

@dataclass
class Args:
    gello_name: str = "yam_gello"
    """The name of the gello (used to determine which file to write to)"""

    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    baudrate: int = 2000000 # 2M bps on Dynamixel Wizard
    """The baudrate of the connected GELLO's dynamixel board."""


def pretty_print_list(items: list[float], title: str = "") -> None:
    """Pretty print a list of numbers with optional title."""
    if title:
        print(f"{title}:")
    for i, x in enumerate(items):
        print(f"  Joint {i+1:2d}: {x:>8.2f}°", end='')
        if (i + 1) % 6 == 0:  # New line every 6 joints (one arm)
            print()
    if len(items) % 6 != 0:
        print()


def get_joint_angles_left(driver: DynamixelDriver, num_joints_per_arm: int = 6) -> np.ndarray:
    """Get joint angles for the left arm."""
    return np.rad2deg(driver.get_joints()[:num_joints_per_arm])


def get_joint_angles_right(driver: DynamixelDriver, num_joints_per_arm: int = 6) -> np.ndarray:
    """Get joint angles for the right arm."""
    return np.rad2deg(driver.get_joints()[num_joints_per_arm:2*num_joints_per_arm])


def compute_joint_offsets_and_signs(
    joints_1: np.ndarray,
    joints_2: np.ndarray,
    expected_pos_1: np.ndarray,
    expected_pos_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the joint signs and offsets given the measured angles at two known positions.

    Args:
        joints_1: Measured joint angles at position 1 (degrees)
        joints_2: Measured joint angles at position 2 (degrees)
        expected_pos_1: Expected joint angles at position 1 (degrees)
        expected_pos_2: Expected joint angles at position 2 (degrees)

    Returns:
        signs (np.ndarray): Joint direction signs
        offsets (np.ndarray): Joint offsets in degrees
    """
    # Compute signs by comparing measured and expected delta between positions
    delta = joints_2 - joints_1
    expected_delta = expected_pos_2 - expected_pos_1
    
    # Avoid division by zero
    signs = np.ones_like(delta)
    non_zero_mask = expected_delta != 0
    signs[non_zero_mask] = np.sign(delta[non_zero_mask] / expected_delta[non_zero_mask])
    
    # Compute offsets by subtracting expected position and accounting for sign
    def round_to_90(x: np.ndarray) -> np.ndarray:
        """Round all elements to the nearest multiple of 90 degrees."""
        return 90 * np.rint(x / 90)
    
    offsets_1 = round_to_90(joints_1 - signs * expected_pos_1)
    offsets_2 = round_to_90(joints_2 - signs * expected_pos_2)
    
    # Check if offsets are consistent between the two positions
    if not np.allclose(offsets_1, offsets_2, atol=1.0):  # 1 degree tolerance
        print("WARNING: The joint offsets at the two positions don't seem to match!")
        print("         Are you sure that you positioned the robot correctly?")
        print("")
        print("         Try re-running this script and placing the arms at the correct")
        print("         positions for calibration!")
        print("")
        print("Expected offsets at position 1:")
        pretty_print_list(offsets_1)
        print("Expected offsets at position 2:")
        pretty_print_list(offsets_2)
        sys.exit(1)
    
    return signs, offsets_1


def main(args: Args) -> None:
    """Main calibration function."""
    # YAM bimanual has 12 arm joints (6 per arm)
    num_joints = 12
    num_joints_per_arm = 6

    # Create joint IDs (assuming consecutive IDs starting from 0)
    joint_ids = list(range(num_joints))
    driver = DynamixelDriver(ids=joint_ids, port=args.port, baudrate=args.baudrate)

    # Initialize arrays to store joint angles
    joint_angles_pos_1 = np.zeros(num_joints)
    joint_angles_pos_2 = np.zeros(num_joints)

    # Warm up dynamixel driver
    print("Warming up dynamixel driver...")
    for _ in range(10):
        driver.get_joints()

    print("=" * 50)
    print("YAM Bimanual JoyLo Joint Calibration")
    print("=" * 50)
    print('')

    # Gather data for the left arm
    print("Now calibrating LEFT arm...")
    print("")
    print("LEFT arm - Position 1 target:")
    pretty_print_list(LEFT_REFERENCE_JOINT_POS_1)
    print("Place the LEFT arm in Position 1 and press enter!", end='')
    input()
    joint_angles_pos_1[:num_joints_per_arm] = get_joint_angles_left(driver, num_joints_per_arm)
    print(f"Measured left arm angles: {joint_angles_pos_1[:num_joints_per_arm]}")
    print("")

    print("LEFT arm - Position 2 target:")
    pretty_print_list(LEFT_REFERENCE_JOINT_POS_2)
    print("Place the LEFT arm in Position 2 and press enter!", end='')
    input()
    joint_angles_pos_2[:num_joints_per_arm] = get_joint_angles_left(driver, num_joints_per_arm)
    print(f"Measured left arm angles: {joint_angles_pos_2[:num_joints_per_arm]}")
    print("")
    
    # Gather data for the right arm
    print("Now calibrating RIGHT arm...")
    print("")
    print("RIGHT arm - Position 1 (Zero/Neutral) target:")
    pretty_print_list(RIGHT_REFERENCE_JOINT_POS_1)
    print("Place the RIGHT arm in Position 1 (Zero/Neutral) and press enter...", end='')
    input()
    joint_angles_pos_1[num_joints_per_arm:] = get_joint_angles_right(driver, num_joints_per_arm)
    print(f"Measured right arm angles: {joint_angles_pos_1[num_joints_per_arm:]}")
    print("")

    print("RIGHT arm - Position 2 target:")
    pretty_print_list(RIGHT_REFERENCE_JOINT_POS_2)
    print("Place the RIGHT arm in Position 2 and press enter!", end='')
    input()
    joint_angles_pos_2[num_joints_per_arm:] = get_joint_angles_right(driver, num_joints_per_arm)
    print(f"Measured right arm angles: {joint_angles_pos_2[num_joints_per_arm:]}")
    print("")

    # Compute offsets and signs
    signs, offsets = compute_joint_offsets_and_signs(
        joint_angles_pos_1, 
        joint_angles_pos_2, 
        REFERENCE_JOINT_POS_1, 
        REFERENCE_JOINT_POS_2
    )

    print("")
    print("Successfully calibrated your YAM bimanual JoyLo:")
    print("")
    print("Left arm signs:")
    pretty_print_list(signs[:num_joints_per_arm])
    print("Left arm offsets:")
    pretty_print_list(offsets[:num_joints_per_arm])
    print("")
    print("Right arm signs:")
    pretty_print_list(signs[num_joints_per_arm:])
    print("Right arm offsets:")
    pretty_print_list(offsets[num_joints_per_arm:])
    
    # Write to output file
    joint_data = {
        "joints": {
            "offsets": [float(x) for x in offsets],  # Convert to float for YAML serialization
            "signs": [float(x) for x in signs],      # Convert to float for YAML serialization
        }
    }
    
    # Ensure config directory exists
    CONFIG_DIR.mkdir(exist_ok=True)
    output_filename = CONFIG_DIR / f"joint_config_{args.gello_name}.yaml"
    
    with open(output_filename, "w") as file:
        yaml.dump(joint_data, file, default_flow_style=False)
    
    print("")
    print(f"Joint offsets and signs have been saved to {output_filename}!")
    print("")
    print("You can now use this configuration file with your YAM bimanual JoyLo setup.")
    print("")


if __name__ == "__main__":
    main(tyro.cli(Args))

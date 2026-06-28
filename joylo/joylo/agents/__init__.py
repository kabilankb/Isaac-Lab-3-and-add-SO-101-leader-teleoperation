"""JoyLo agents and shared gripper constants for bimanual YAM teleoperation."""

# Gripper constants shared between the leader (joylo) and the sim package.
# Defined here because joylo runs without Isaac Sim, so it cannot import
# them from the sim robot module (which pulls in isaaclab → carb).
GRIPPER_OPEN_POS = -0.0475    # fully open finger joint position (rad)
GRIPPER_CLOSED_POS = 0.0      # fully closed finger joint position (rad)
GRIPPER_RANGE = abs(GRIPPER_OPEN_POS - GRIPPER_CLOSED_POS)
GRIPPER_DELTA_PER_STEP = GRIPPER_RANGE / 20

from .agent import *
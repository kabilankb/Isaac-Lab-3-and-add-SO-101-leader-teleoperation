"""Small math/transform helpers shared across the package."""
import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz
import isaaclab.utils.math as PoseUtils


def euler2quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> torch.Tensor:
    """Convert XYZ Euler angles in degrees to a quaternion in (x, y, z, w) order (Isaac Lab 3).

    Args:
        roll_deg (float): Roll angle in degrees.
        pitch_deg (float): Pitch angle in degrees.
        yaw_deg (float): Yaw angle in degrees.

    Returns:
        torch.Tensor: float quaternion, shape (4,), in (x, y, z, w) order.
    """
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    quat_tensor = quat_from_euler_xyz(
        torch.tensor([roll_rad]),
        torch.tensor([pitch_rad]),
        torch.tensor([yaw_rad])
    )[0]  # Get first (and only) element
    return quat_tensor

def get_delta_object_pose(
    cur_obj_pose: torch.Tensor,
    src_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """Compute the relative pose mapping the source pose onto the current pose.

    Computes ``delta = cur_obj_pose @ inv(src_obj_pose)``, the homogeneous
    transform that takes ``src_obj_pose`` to ``cur_obj_pose``. Used by the
    MimicGen coordination-transform scheme.

    Args:
        cur_obj_pose (torch.Tensor): float homogeneous pose matrix, shape (..., 4, 4).
        src_obj_pose (torch.Tensor): float homogeneous pose matrix, shape (..., 4, 4).

    Returns:
        torch.Tensor: float homogeneous delta pose matrix, shape (..., 4, 4).
    """
    return cur_obj_pose @ PoseUtils.pose_inv(src_obj_pose)

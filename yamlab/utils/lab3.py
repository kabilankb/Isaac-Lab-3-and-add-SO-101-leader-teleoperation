"""Isaac Lab 3 compatibility helpers for the Isaac Sim 6 port.

Isaac Lab 3 stores quaternions as (x, y, z, w); Isaac Lab 2.x (and therefore the YAMLab YAML
configs) use (w, x, y, z). Runtime code in this package works in XYZW; legacy WXYZ values are
converted at the input boundary with these helpers. Legacy 2.x HDF5 demos need no helper here:
Isaac Lab 3's ``HDF5DatasetFileHandler.load_episode`` detects them (no ``format_version``
attribute) and converts every ``root_pose`` to XYZW on load.
"""

import numpy as np
import torch

def quat_wxyz_to_xyzw(q):
    """Reorder quaternion(s) from (w, x, y, z) to (x, y, z, w) along the last axis.

    Args:
        q (torch.Tensor or np.ndarray or sequence): quaternion(s), shape (..., 4).

    Returns:
        Same type as ``q`` (sequences come back as tuples), shape (..., 4).
    """
    if isinstance(q, torch.Tensor):
        return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)
    if isinstance(q, np.ndarray):
        return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
    return (q[1], q[2], q[3], q[0])


def quat_xyzw_to_wxyz(q):
    """Reorder quaternion(s) from (x, y, z, w) to (w, x, y, z) along the last axis.

    Args:
        q (torch.Tensor or np.ndarray or sequence): quaternion(s), shape (..., 4).

    Returns:
        Same type as ``q`` (sequences come back as tuples), shape (..., 4).
    """
    if isinstance(q, torch.Tensor):
        return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)
    if isinstance(q, np.ndarray):
        return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)
    return (q[3], q[0], q[1], q[2])


def as_torch(x):
    """Return the ``torch.Tensor`` view of an Isaac Lab 3 ``ProxyArray`` (identity for tensors/None).

    Isaac Lab 3 asset/sensor ``.data.*`` properties return ``ProxyArray``; implicit tensor use
    goes through a deprecation bridge (and bypasses the quaternion-order detector), so reads go
    through this helper or ``.torch`` explicitly.

    Args:
        x (ProxyArray or torch.Tensor or None): The data property value.

    Returns:
        torch.Tensor or None: Zero-copy tensor view.
    """
    return x.torch if hasattr(x, "torch") and not isinstance(x, torch.Tensor) else x

"""J-PARSE IK solver for MimicGen trajectory generation.

Replaces IsaacLab's Damped Least Squares (DLS) with J-PARSE
(Jacobian-based Projection Algorithm for Resolving Singularities Effectively).

DLS adds constant damping to ALL Jacobian directions, which slows motion even
when the robot is far from singularities. J-PARSE uses SVD to identify
near-singular directions and only dampens those, giving full-speed motion in
well-conditioned directions.

Reference:
    Guptasarma, Strong, Zhen, Kennedy.
    "J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities
     Effectively in Inverse Kinematic Control of Serial Manipulators"
    https://github.com/armlabstanford/jparse

Adapted from BEHAVIOR-1K/OmniGibson (MIT License).
Original numba/numpy implementation converted to pure PyTorch for GPU compatibility.
"""

import torch


def jparse_pseudo_inverse(
    jacobian: torch.Tensor,
    gamma: float = 0.1,
    sg_gain_pos: float = 1.0,
    sg_gain_ang: float = 1.0,
) -> torch.Tensor:
    """Compute the J-PARSE pseudo-inverse of a Jacobian matrix.

    Args:
        jacobian: (m, n) Jacobian matrix (typically 6 x num_joints).
        gamma: Singularity threshold relative to max singular value (0 < gamma < 1).
            Singular values below ``gamma * sigma_max`` are treated as near-singular.
        sg_gain_pos: Gain for position-related singular directions.
        sg_gain_ang: Gain for orientation-related singular directions.

    Returns:
        (n, m) J-PARSE pseudo-inverse matrix.
    """
    J = jacobian.double()
    m, n = J.shape

    U, S, Vh = torch.linalg.svd(J, full_matrices=True)
    k = S.shape[0]  # min(m, n)
    sigma_max = S[0]
    threshold = gamma * sigma_max

    non_singular_mask = S > threshold  # (k,)

    # J_safety: singular values clamped to threshold (never zero denominator).
    S_safety = torch.where(non_singular_mask, S, threshold * torch.ones_like(S))
    J_safety = U[:, :k] @ torch.diag(S_safety) @ Vh[:k, :]

    # J_proj: retain only non-singular directions (zero out near-singular ones).
    S_proj = torch.where(non_singular_mask, S, torch.zeros_like(S))
    J_proj = U[:, :k] @ torch.diag(S_proj) @ Vh[:k, :]

    J_safety_pinv = torch.linalg.pinv(J_safety)
    J_proj_pinv = torch.linalg.pinv(J_proj)

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv

    # Singular-direction feedback term (Phi_singular): proportionally fades
    # near-singular directions back in, weighted by position/orientation gains.
    n_sing = int((~non_singular_mask).sum().item())
    if n_sing > 0:
        sing_indices = torch.where(~non_singular_mask)[0]
        U_sing = U[:, sing_indices]  # (m, n_sing)
        phi_vals = S[sing_indices] / (sigma_max * gamma)  # proportional fade
        Phi_mat = torch.diag(phi_vals)  # (n_sing, n_sing)

        gains = torch.empty(m, device=J.device, dtype=J.dtype)
        gains[:3] = sg_gain_pos
        gains[3:] = sg_gain_ang
        Kp = torch.diag(gains)

        Phi_singular = U_sing @ Phi_mat @ U_sing.T @ Kp  # (m, m)
        J_parse = J_parse + J_safety_pinv @ Phi_singular

    return J_parse


def compute_ik_jparse(
    ee_pos_b: torch.Tensor,
    ee_quat_b: torch.Tensor,
    ee_pos_des_b: torch.Tensor,
    ee_quat_des_b: torch.Tensor,
    jacobian: torch.Tensor,
    joint_pos: torch.Tensor,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Compute desired joint positions using J-PARSE IK.

    Same interface as ``IsaacLab DifferentialIKController.compute()``, but
    replaces the DLS pseudo-inverse with J-PARSE. Assumes a single environment
    (batch dim is squeezed internally).

    Args:
        ee_pos_b: Current EEF position in base frame, shape ``(1, 3)``.
        ee_quat_b: Current EEF quaternion in base frame, shape ``(1, 4)``.
        ee_pos_des_b: Desired EEF position in base frame, shape ``(1, 3)``.
        ee_quat_des_b: Desired EEF quaternion in base frame, shape ``(1, 4)``.
        jacobian: Geometric Jacobian, shape ``(1, 6, num_joints)``.
        joint_pos: Current joint positions, shape ``(1, num_joints)``.
        gamma: J-PARSE singularity threshold.

    Returns:
        Desired joint positions of shape ``(1, num_joints)``.
    """
    from isaaclab.utils.math import compute_pose_error

    pos_error, axis_angle_error = compute_pose_error(
        ee_pos_b, ee_quat_b, ee_pos_des_b, ee_quat_des_b, rot_error_type="axis_angle"
    )
    delta_pose = torch.cat((pos_error, axis_angle_error), dim=1)  # (1, 6)

    # All math in float64 for numerical stability; cast back at the end.
    J = jacobian[0]  # (6, num_joints)
    J_parse = jparse_pseudo_inverse(J, gamma=gamma)  # (num_joints, 6) float64

    delta_joint_pos = J_parse @ delta_pose[0].double()  # (num_joints,)

    return joint_pos + delta_joint_pos.unsqueeze(0).to(joint_pos.dtype)  # (1, num_joints)

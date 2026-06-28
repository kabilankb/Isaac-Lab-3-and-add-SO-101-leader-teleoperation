"""Task success checks and reward functions for the YAM bimanual tasks."""
import torch

from isaaclab.managers import SceneEntityCfg


def check_pick_success(
    env,
    asset_cfg: SceneEntityCfg,
    init_obj_height: float,
    pick_threshold: float = 0.20,
    check_contact: bool = False,
    pick_by_two_hands: bool = False,
    pick_arm: str | None = None,
    grasp=None
) -> torch.Tensor:
    """Check if the pick subtask is successful.

    Success requires the object to be above init_obj_height + pick_threshold.
    Optionally also requires gripper contact, as reported by the robot's grasp detection.

    Args:
        env: Environment instance.
        asset_cfg: Asset configuration for the object.
        init_obj_height: Initial stable Z of the object center (table_z + height/2 + offset).
        pick_threshold: Minimum height above initial position to count as picked (m).
        check_contact: Whether to require gripper contact in addition to height.
        pick_by_two_hands: Require both arms in contact (ignored when pick_arm is set).
        pick_arm: Require only "left" or "right" arm. Overrides pick_by_two_hands.
        grasp: grasp handle exposing is_grasping() -- the robot or one of its arms (required when check_contact=True).

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    obj = env.scene[asset_cfg.name]
    obj_height = obj.data.root_pos_w[:, 2]
    height_success = (obj_height - init_obj_height) > pick_threshold

    if not check_contact:
        return height_success

    if grasp is None:
        print("[WARNING] check_contact=True but grasp not provided. Falling back to height check only.")
        return height_success

    left_grasping, right_grasping = grasp.is_grasping()

    if pick_arm == "left":
        contact_success = left_grasping
    elif pick_arm == "right":
        contact_success = right_grasping
    elif pick_by_two_hands:
        contact_success = left_grasping & right_grasping
    else:
        contact_success = (left_grasping & ~right_grasping) | (right_grasping & ~left_grasping)

    return height_success & contact_success


def check_ontop_success(
    env,
    pot_cfg: SceneEntityCfg,
    cooktop_cfg: SceneEntityCfg,
    pot_height: float,
    cooktop_height: float,
    height_tolerance: float = 0.02,
    orientation_tolerance: float | None = 0.2,
    xy_tolerance: float = 0.20,
    check_gripper_release: bool = False,
    gripper_release_mode: str = "both",
    grasp=None
) -> torch.Tensor:
    """Check if pot is successfully placed on top of cooktop.

    Success requires: pot XY within xy_tolerance of cooktop; pot Z within
    height_tolerance of cooktop_center_z + cooktop_height/2 + pot_height/2; and
    pot upright. Optionally also checks gripper release.

    Args:
        env: Environment instance.
        pot_cfg: Asset configuration for pot (object placed on top).
        cooktop_cfg: Asset configuration for cooktop (base object).
        pot_height: Full height of pot (m).
        cooktop_height: Full height of cooktop (m).
        height_tolerance: Max deviation from expected stacking height (m).
        orientation_tolerance: Max angle from vertical (rad). None skips check.
        xy_tolerance: Max horizontal distance between obj centers (m).
        check_gripper_release: Whether to check gripper release.
        gripper_release_mode: "both", "left_only", or "right_only".
        grasp: grasp handle exposing is_grasping() -- the robot or one of its arms (required when check_gripper_release=True).

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    pot = env.scene[pot_cfg.name]
    pot_pos = pot.data.root_pos_w
    pot_z = pot_pos[:, 2]
    pot_quat = pot.data.root_quat_w

    cooktop = env.scene[cooktop_cfg.name]
    cooktop_pos = cooktop.data.root_pos_w
    cooktop_z = cooktop_pos[:, 2]

    xy_distance = torch.norm(pot_pos[:, :2] - cooktop_pos[:, :2], dim=1)
    xy_aligned = xy_distance < xy_tolerance

    expected_pot_z = cooktop_z + cooktop_height / 2.0 + pot_height / 2.0
    at_ontop_height = torch.abs(pot_z - expected_pot_z) < height_tolerance

    if orientation_tolerance is not None:
        qx, qy = pot_quat[:, 1], pot_quat[:, 2]
        z_axis_z = 1 - 2 * (qx**2 + qy**2)
        orientation_error = torch.acos(torch.clamp(torch.abs(z_axis_z), 0, 1))
        ontop_success = xy_aligned & at_ontop_height & (orientation_error < orientation_tolerance)
    else:
        ontop_success = xy_aligned & at_ontop_height

    if check_gripper_release:
        if grasp is None:
            print("[WARNING] check_gripper_release=True but grasp not provided. Skipping gripper release check.")
            return ontop_success

        left_grasping, right_grasping = grasp.is_grasping()

        if gripper_release_mode == "both":
            grippers_released = (~left_grasping) & (~right_grasping)
        elif gripper_release_mode == "left_only":
            grippers_released = ~left_grasping
        elif gripper_release_mode == "right_only":
            grippers_released = ~right_grasping
        else:
            print(f"[WARNING] Unknown gripper_release_mode '{gripper_release_mode}'. Defaulting to 'both'.")
            grippers_released = (~left_grasping) & (~right_grasping)

        ontop_success = ontop_success & grippers_released
    return ontop_success


def check_hang_success(
    env,
    mug_cfg: SceneEntityCfg,
    tree_cfg: SceneEntityCfg,
    mug_height: float,
    table_z: float,
    xy_tolerance: float = 0.15,
    min_hang_height: float = 0.05,
    check_gripper_release: bool = False,
    gripper_release_mode: str = "both",
    grasp=None
) -> torch.Tensor:
    """Check if the mug is successfully hung on the mug tree.

    Success requires: mug XY within xy_tolerance of tree; mug Z above
    table_z + mug_height/2 + min_hang_height. The consecutive-steps mechanism
    in the environment confirms the mug stays elevated after gripper release.

    Args:
        env: Environment instance.
        mug_cfg: Asset configuration for the mug (mug).
        tree_cfg: Asset configuration for the mug tree (mug tree).
        mug_height: Full height of the mug (m).
        table_z: Z position of the table surface.
        xy_tolerance: Max horizontal distance between mug and tree centers (m).
        min_hang_height: Min elevation above mug's table-resting position (m).
        check_gripper_release: Whether to require both grippers released.
        gripper_release_mode: "both", "left_only", or "right_only".
        grasp: grasp handle exposing is_grasping() -- the robot or one of its arms (required when check_gripper_release=True).

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    mug = env.scene[mug_cfg.name]
    mug_pos = mug.data.root_pos_w
    mug_z = mug_pos[:, 2]

    tree = env.scene[tree_cfg.name]
    tree_pos = tree.data.root_pos_w

    xy_distance = torch.norm(mug_pos[:, :2] - tree_pos[:, :2], dim=1)
    xy_aligned = xy_distance < xy_tolerance

    # When resting on table: mug center Z ~ table_z + mug_height/2.
    min_z = table_z + mug_height / 2.0 + min_hang_height
    hang_success = xy_aligned & (mug_z > min_z)

    if check_gripper_release:
        if grasp is None:
            print("[WARNING] check_gripper_release=True but grasp not provided. Skipping gripper release check.")
            return hang_success

        left_grasping, right_grasping = grasp.is_grasping()

        if gripper_release_mode == "both":
            grippers_released = (~left_grasping) & (~right_grasping)
        elif gripper_release_mode == "left_only":
            grippers_released = ~left_grasping
        elif gripper_release_mode == "right_only":
            grippers_released = ~right_grasping
        else:
            print(f"[WARNING] Unknown gripper_release_mode '{gripper_release_mode}'. Defaulting to 'both'.")
            grippers_released = (~left_grasping) & (~right_grasping)

        hang_success = hang_success & grippers_released

    return hang_success


def check_obj_below_table(env, asset_cfg: SceneEntityCfg, table_z: float, margin: float = 0.01):
    """Return True for environments where the object has fallen below the table.

    Args:
        env: Environment instance.
        asset_cfg: Scene entity configuration for the object.
        table_z: Z position of the table surface.
        margin: Distance below table_z that triggers termination (m).

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < (table_z - margin)


def compute_sparse_hang_reward(env) -> torch.Tensor:
    """Sparse, episode-bounded success reward for HangMugOnTree.

    Per-episode totals (summed across steps):
      - 0.0 if stage2 (handover) never latches
      - 0.5 if stage2 latches but stage3 (hang) does not
      - 1.0 if stage3 latches (which implies stage2 also latched)

    One-shot guards (_stage2_reward_given, _stage3_reward_given) prevent the latched
    flags from re-crediting on subsequent steps. Both are cleared in
    HangMugOnTreeManager.reset_success_check.

    Args:
        env: HangMugOnTree environment instance with stage2_success and stage3_success tensors.

    Returns:
        torch.Tensor of shape (num_envs,), per-env per-step reward in {0, 0.5}.
    """
    if not hasattr(env, "_stage2_reward_given"):
        env._stage2_reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_stage3_reward_given"):
        env._stage3_reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    stage2_new = env.stage2_success & ~env._stage2_reward_given
    reward[stage2_new] += 0.5
    env._stage2_reward_given |= env.stage2_success

    stage3_new = env.stage3_success & ~env._stage3_reward_given
    reward[stage3_new] += 0.5
    env._stage3_reward_given |= env.stage3_success

    return reward


def compute_sparse_success_reward(env) -> torch.Tensor:
    """Sparse binary reward: +1.0 once per episode when all stages succeed.

    Cumulative episode return equals 1.0 on success, 0.0 on failure.
    Uses a one-shot _success_reward_given guard cleared in reset_success_check().

    Args:
        env: Environment with get_task_success() returning (num_envs,) bool.

    Returns:
        torch.Tensor of shape (num_envs,), per-env per-step reward in {0, 1}.
    """
    if not hasattr(env, "_success_reward_given"):
        env._success_reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    task_success = env.get_task_success()
    new_success = task_success & ~env._success_reward_given
    reward[new_success] = 1.0
    env._success_reward_given |= task_success
    return reward

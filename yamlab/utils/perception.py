"""Observation extraction (masks, joint/gripper state) and camera/calibration utilities."""
from typing import Dict, List

import torch

from isaaclab.managers import SceneEntityCfg

GROUND_ID = None


def extract_background_mask(env, sensor_cfg):
    """Compute a per-pixel background mask from a camera's segmentation and depth outputs.

    Background is the union of floor pixels (matched against the "ground" class in the
    semantic-segmentation label map) and sky pixels (infinite depth). The ground class id is
    resolved once and cached in the module-level ``GROUND_ID``.

    Args:
        env (ManagerBasedRLEnv): Task environment exposing the camera sensor in its scene.
        sensor_cfg (SceneEntityCfg): Configuration identifying the camera sensor.

    Returns:
        torch.Tensor: bool, shape (1, H, W); True at background pixels.
    """
    global GROUND_ID
    camera = env.scene[sensor_cfg.name]
    depth = camera.data.output["distance_to_image_plane"][0]

    background_mask = torch.zeros_like(depth, dtype=torch.bool)

    if GROUND_ID is None:
        class_id_to_label = camera.data.info[0]['semantic_segmentation']['idToLabels']
        for id, label_dict in class_id_to_label.items():
            if label_dict.get("class") == "ground":
                GROUND_ID = int(id)
                break

    if GROUND_ID is not None:
        floor_mask = (camera.data.output["semantic_segmentation"][0] == GROUND_ID)
        if floor_mask.shape != depth.shape:
            floor_mask = floor_mask.squeeze()
        background_mask |= floor_mask

    background_mask |= torch.isinf(depth)

    return background_mask.unsqueeze(0)


def get_arm_joint_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the 6-DOF arm joint positions, excluding the gripper joints.

    Args:
        env (ManagerBasedRLEnv): Task environment exposing the arm articulation in its scene.
        asset_cfg (SceneEntityCfg): Configuration identifying the arm articulation.

    Returns:
        torch.Tensor: float, shape (num_envs, 6); the first six joint positions.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos.torch[:, :6]


def get_gripper_continuous_state(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the continuous gripper state as the leading finger joint position.

    Both fingers are mirrored to the same value, so the joint at index 6 is read as a proxy for
    the gripper opening. Its range is approximately -0.0475 (open) to 0.0 (closed).

    Args:
        env (ManagerBasedRLEnv): Task environment exposing the arm articulation in its scene.
        asset_cfg (SceneEntityCfg): Configuration identifying the arm articulation.

    Returns:
        torch.Tensor: float, shape (num_envs, 1); the finger joint position.
    """
    asset = env.scene[asset_cfg.name]
    finger_pos = asset.data.joint_pos.torch[:, 6]
    return finger_pos.unsqueeze(-1)


# ===========================================================================
# Camera / calibration utilities (merged from former yam_utils.py)
# ===========================================================================


def intrinsics_to_matrix(intrinsics: Dict[str, float]) -> List[float]:
    """Convert a camera intrinsics dict to a flat, row-major 3x3 matrix.

    Args:
        intrinsics (dict): Pinhole intrinsics with keys ``fx``, ``fy``, ``cx``, ``cy``.

    Returns:
        list[float]: The 9 entries of the row-major intrinsic matrix
            ``[fx, 0, cx, 0, fy, cy, 0, 0, 1]``.
    """
    return [
        intrinsics["fx"], 0.0, intrinsics["cx"],
        0.0, intrinsics["fy"], intrinsics["cy"],
        0.0, 0.0, 1.0
    ]

def setup_camera_viewports_fallback():
    """Create viewport windows for the top and left/right wrist cameras from hardcoded prim paths.

    Fallback used when ``setup_real_camera_viewports_from_scene`` cannot resolve cameras from the
    scene object. Delegates to the shared ``_setup_camera_viewport`` helper.

    Returns:
        tuple: ``(top_viewport, left_wrist_viewport, right_wrist_viewport)`` viewport windows.
    """
    top_viewport = _setup_camera_viewport(
        "/World/envs/env_0/TopCamera",
        "Top Camera View",
        (720, 540),
        position_type="top",
    )
    left_wrist_viewport = _setup_camera_viewport(
        "/World/envs/env_0/LeftArm/arm/link_6/wrist_camera",
        "Left Wrist Camera",
        (720, 540),
        position_type="left_wrist",
    )
    right_wrist_viewport = _setup_camera_viewport(
        "/World/envs/env_0/RightArm/arm/link_6/wrist_camera",
        "Right Wrist Camera",
        (720, 540),
        position_type="right_wrist",
    )
    return top_viewport, left_wrist_viewport, right_wrist_viewport


def setup_real_camera_viewports_from_scene(scene):
    """Create viewport windows bound to the top and wrist cameras resolved from the scene.

    Args:
        scene (InteractiveScene): Scene containing the ``top_camera``, ``left_wrist_camera``, and
            ``right_wrist_camera`` sensors.

    Returns:
        tuple: ``(top_viewport, left_wrist_viewport, right_wrist_viewport)`` viewport windows;
            each entry is None if its camera is unavailable, and all three are None when the
            scene is None.
    """
    if scene is None:
        print("[WARNING] No scene available for real camera viewports")
        return None, None, None
    
    cameras = {}
    cameras["top_camera"] = scene["top_camera"]
    cameras["left_wrist_camera"] = scene["left_wrist_camera"]
    cameras["right_wrist_camera"] = scene["right_wrist_camera"]
    
    if not cameras:
        print("[WARNING] No cameras found in scene for real camera viewports")
        return None, None, None
    
    # Setup viewports for each available camera
    top_viewport = None
    left_wrist_viewport = None
    right_wrist_viewport = None
    
    if "top_camera" in cameras:
        top_viewport = _setup_camera_viewport(
            "/World/envs/env_0/TopCamera",
            "Top Camera (Real)",
            (480, 360),
            position_type="top"
        )

    if "left_wrist_camera" in cameras:
        left_wrist_viewport = _setup_camera_viewport(
            "/World/envs/env_0/LeftArm/arm/link_6/wrist_camera",
            "Left Wrist Camera (Real)",
            (480, 360),
            position_type="left_wrist"
        )

    if "right_wrist_camera" in cameras:
        right_wrist_viewport = _setup_camera_viewport(
            "/World/envs/env_0/RightArm/arm/link_6/wrist_camera",
            "Right Wrist Camera (Real)",
            (480, 360),
            position_type="right_wrist"
        )
    
    return top_viewport, left_wrist_viewport, right_wrist_viewport


def _setup_camera_viewport(cam_prim_path, window_title, size, position_type="top"):
    """Create a viewport window, bind it to a camera prim, and dock it in a corner.

    Args:
        cam_prim_path (str): USD prim path of the camera to bind to the viewport.
        window_title (str): Title of the viewport window.
        size (tuple): ``(width, height)`` of the window in pixels.
        position_type (str): Docking corner: ``"top"``, ``"left_wrist"``, or ``"right_wrist"``.

    Returns:
        ViewportWindow: The created viewport window.
    """
    import omni.kit.app
    import omni.ui as ui
    from omni.kit.viewport.window import ViewportWindow

    app = omni.kit.app.get_app()
    window_width, window_height = size

    # 1) Create the window
    vp_window = ViewportWindow(window_title, width=window_width, height=window_height, visible=True)

    # 2) Let the UI realize the viewport API
    # Without this, get_viewport_from_window_name(window_title) may be None
    for _ in range(10):
        app.update()

    # 3) Get the specific viewport API for this window (not the active one)
    vp_api = vp_window.viewport_api

    # 4) Bind the camera to this window.
    # Use the window API so it creates/attaches a render product for you.
    try:
        vp_api.set_active_camera(cam_prim_path)
        vp_api.set_texture_resolution((window_width, window_height))

    except Exception as exc:
        print(f"[WARNING] Failed to bind camera '{cam_prim_path}' to '{window_title}': {exc}")

    # 5) Docking/positioning
    try:
        main_window = ui.Workspace.get_window("Viewport")
        dock_win = ui.Workspace.get_window(window_title)
        if main_window and dock_win:
            margin = 10
            if position_type == "top":
                # Top-right corner of main viewport
                dock_win.position_x = main_window.position_x + max(0, main_window.width - window_width - margin)
                dock_win.position_y = main_window.position_y + margin
            elif position_type == "left_wrist":
                # Bottom-left area (under left arm in the scene)
                dock_win.position_x = main_window.position_x + margin
                dock_win.position_y = main_window.position_y + max(0, main_window.height - window_height - margin)
            elif position_type == "right_wrist":
                # Bottom-right area (under right arm in the scene)
                dock_win.position_x = main_window.position_x + max(0, main_window.width - window_width - margin)
                dock_win.position_y = main_window.position_y + max(0, main_window.height - window_height - margin)
    except Exception:
        pass

    print(f"[INFO] Camera viewport '{window_title}' bound to {cam_prim_path} at {window_width}x{window_height}")
    return vp_window

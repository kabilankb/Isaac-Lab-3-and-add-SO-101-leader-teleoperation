"""
Teleoperation follower for the YAM bimanual robot.

Runs the IsaacLab (ManagerBasedRLEnv) simulation and serves the two arms over an RPC server
so a JoyLo leader can drive them. Streams the top and wrist cameras for operator feedback and
records state-action demonstrations to HDF5 for the downstream replay/MimicGen pipeline.

Run (one --asset per object instance dir):
  python launch_follower.py \
    --task PutPotOnCooktop-v0 \
    --asset pot=/path/to/pot \
    --asset cooktop=/path/to/cooktop \
    --port 11333 \
    --demos_per_asset 25 \
    --data_output_dir /path/to/datasets \
    --data_output_filename teleoperation_data \
    --enable_cameras
"""

import argparse
import os
import threading
import time
from typing import Optional

import numpy as np
from isaaclab.app import AppLauncher

from yamlab import ROOT_DIR
from yamlab.configs import get_objects_randomization, get_task_kwargs, resolve_assets, parse_asset_args

parser = argparse.ArgumentParser(description="Bimanual Follower: IsaacLab YAM arms RPC server (ManagerBasedRLEnv).")
parser.add_argument("--task", type=str, default="PutPotOnCooktop-v0", help="Task environment to use")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                    help="Object asset instance dir, e.g. --asset pot=/path/to/pot "
                         "(repeatable, one per object). Overrides objects.<name>.asset in the task YAML.")
parser.add_argument("--port", type=int, default=11333, help="Port for bimanual RPC server")

# Task environment arguments
parser.add_argument("--demos_per_asset", type=int, default=5,
                    help="Number of demonstrations to collect per asset")
parser.add_argument("--init_joint_pos_randomization", type=float, default=0.0,
                    help="Per-reset arm joint randomization range (+/- radians). Default 0 (no randomization).")
parser.add_argument("--data_output_dir", type=str, 
                    default=os.path.join(ROOT_DIR, "recorded_data"),
                    help="Directory to save collected data")
parser.add_argument("--data_output_filename", type=str, default="teleoperation_data",
                    help="Base filename for recorded data (without extension)")
parser.add_argument("--enable_pose_schedule", action="store_true", default=False,
                    help="Enable pose schedule for teleoperation (uses schedule defined in task config)")
parser.add_argument("--enable_gripper_clamp", action="store_true", default=False,
                    help="Enable gripper grasp protection: clamp gripper command when grasp detected to prevent over-closing")
parser.add_argument("--disable_grasp_ray_viz", action="store_true", default=False,
                    help="Disable the green/red grasp-ray visualization drawn between the gripper "
                         "fingertips during teleoperation (an on-screen beam hinting that closing "
                         "the gripper is likely to grasp an object; on by default)")

# IsaacLab arg passthrough (adds --device, whose built-in default is "cuda:0").
AppLauncher.add_app_launcher_args(parser)
# Override that default to None so an omitted --device is distinguishable from an
# explicit one and can fall back to the per-mode YAML device, resolved below.
parser.set_defaults(device=None)
args = parser.parse_args()

# Pick the device: an explicit --device wins, otherwise the teleoperation YAML
# default (cpu). Assigned back to args.device so AppLauncher (below) and
# create_task_environment use the same device.
from yamlab.configs import resolve_device
args.device = resolve_device(args.task, "teleoperation", args.device)

# Launch Kit/IsaacLab
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from yamlab.teleoperation import TeleopManagerWrapper, BimanualRPCServer
from yamlab.utils.perception import setup_real_camera_viewports_from_scene
from yamlab.utils.task_creation import create_task_environment
from joylo.follower_server import PortalSimServer, apply_teleop_render_optimizations


def main():
    """Create the teleoperation environment, start the RPC server, and run the control loop."""
    # Get task-specific configs for the selected task
    task_kwargs = get_task_kwargs(args.task, mode="teleop")
    objects_randomization = get_objects_randomization(args.task)
    print(f"[INFO] Using task-specific kwargs for {args.task}: {task_kwargs}")

    # Resolve object asset instances from --asset
    assets_instance_paths = resolve_assets(
        args.task,
        cli_assets=parse_asset_args(args.asset),
    )
    if not assets_instance_paths:
        raise SystemExit(
            "No object assets specified. Pass --asset pot=/path/to/instance "
            "(one per object), or set objects.<name>.asset in the task YAML."
        )

    # Create the task environment.  Task-specific parameters are passed
    # via kwargs.  If enable_pose_schedule is True, it overrides
    # randomization for scheduled objects.
    task_env = create_task_environment(
        task_name=args.task,
        assets_instance_paths=assets_instance_paths,
        objects_randomization=objects_randomization,
        init_joint_pos_randomization=args.init_joint_pos_randomization,
        mode="teleoperation",  # Teleoperation mode
        device=args.device,  # matches the launched app device (cpu by default for teleop)
        enable_pose_schedule=args.enable_pose_schedule,  # Enable pose schedule if flag is set
        enable_self_collisions=False,
        enable_gripper_grasp_clamp=args.enable_gripper_clamp,
        enable_grasp_ray_viz=not args.disable_grasp_ray_viz,
        **task_kwargs  # Unpack task-specific kwargs
    )

    # Override demos_per_asset with number of scheduled poses if pose schedule is enabled
    demos_per_asset = args.demos_per_asset
    if args.enable_pose_schedule:
        pose_info = task_env.get_pose_schedule_info()
        total_poses = pose_info.get("total_poses", 0)
        if total_poses > 0:
            demos_per_asset = total_poses
            print(f"[INFO] Pose schedule enabled: overriding demos_per_asset to {demos_per_asset} (number of scheduled poses)")
        else:
            print(f"[WARNING] Pose schedule enabled but no poses found, using demos_per_asset={args.demos_per_asset}")
    
    # Setup camera viewports for streaming observations
    print(f"[INFO] Setting up camera viewports...")
    # Wait a bit for the scene to be fully initialized
    time.sleep(1.0)
    
    # Setup real camera viewports directly from scene (always use real cameras, never virtual)
    top_viewport, left_wrist_viewport, right_wrist_viewport = setup_real_camera_viewports_from_scene(task_env.scene)
    print(f"[INFO] Using built-in D405 cameras for viewports")

    # Fall back to virtual cameras if real cameras not available
    if top_viewport is None and left_wrist_viewport is None and right_wrist_viewport is None:
        print(f"[WARNING] No real cameras available, falling back to virtual viewports")
        from yamlab.utils.perception import setup_camera_viewports_fallback
        top_viewport, left_wrist_viewport, right_wrist_viewport = setup_camera_viewports_fallback()
    
    # Wrap with teleop wrapper for data collection
    print(f"[INFO] Creating TeleopManagerWrapper for data collection...")
    teleop_wrapper = TeleopManagerWrapper(
        task_env=task_env,
        output_dir=args.data_output_dir,
        output_filename=args.data_output_filename,
        demos_per_asset=demos_per_asset,  # Use overridden value if pose schedule is enabled
        flush_every_n_steps=100
    )

    # Add ASSETS_INSTANCE_PATHS as an attribute to the HDF5 data group
    # Save category/instance (last two path components) so replay/mimicgen can
    # reconstruct full paths by joining with --assets_root_path.
    # E.g. "/data/assets/PotFactory/PotFactory_000" → "PotFactory/PotFactory_000"
    assets_relative_paths = {
        obj_name: os.path.join(
            os.path.basename(os.path.dirname(os.path.normpath(path))),
            os.path.basename(os.path.normpath(path))
        )
        for obj_name, path in assets_instance_paths.items()
    }
    teleop_wrapper._add_datagroup_attr("ASSETS_INSTANCE_PATHS", assets_relative_paths)
    
    # Create RPC server for bimanual control
    print(f"[INFO] Creating BimanualRPCServer...")
    rpc_server = BimanualRPCServer(task_env, teleop_wrapper)
    
    # Create single bimanual portal server
    bimanual_server = PortalSimServer(
        rpc_server, 
        args.port, 
        teleop_wrapper
    )
    
    # Start server in background thread
    bimanual_server.start_server_thread()

    obs_dict, info = task_env.reset()

    # Decouple rendering from the control loop + remove frame-rate caps for teleop.
    apply_teleop_render_optimizations(task_env)

    print(f"[INFO] Setup complete. Task environment loaded:")
    print(f"[INFO] - Environment: {task_env.__class__.__name__}")
    if hasattr(task_env.scene, 'left_arm'):
        print(f"[INFO] - Left arm: {task_env.scene['left_arm']}")
        print(f"[INFO] - Right arm: {task_env.scene['right_arm']}")
    if hasattr(task_env, 'observation_manager'):
        print(f"[INFO] - Observation manager: {len(task_env.observation_manager.group_obs_term_dim)} terms")
    print(f"[INFO] - Recording: {task_env.recorder_manager is not None}")
    
    print(f"[INFO] Simulation dt: {task_env.physics_dt}")
    print(f"[INFO] Environment step dt: {task_env.step_dt}")
    print(f"[INFO] Bimanual follower running:")
    print(f"[INFO] - Bimanual RPC server on port {args.port} (num_dofs=14)")
    
    print(f"[INFO] - Data collection: {args.data_output_dir}/{args.data_output_filename}")
    print(f"[INFO] Ready for leader connections...")

    # Pace the step+record loop to the control rate so teleop runs realtime, default to 30 Hz.
    control_dt = float(task_env.step_dt)
    last_time = time.time()

    # Main simulation loop: all state-modifying operations happen in the main thread.
    while simulation_app.is_running():
        # Run pending RPC requests (reset_task, save_trajectory, ...) in the main thread.
        if bimanual_server.process_pending_requests():
            last_time = time.time()  # resync pacing after a request/reset
            continue

        # Get current joint commands as metadata for recording
        left_action = rpc_server.get_left_joint_pos()
        right_action = rpc_server.get_right_joint_pos()

        # Prepare metadata
        metadata = {
            "timestamp": time.time(),
            "physics_dt": float(task_env.physics_dt),
            "step_dt": float(task_env.step_dt),
            "left_arm_joints": left_action.tolist(),
            "right_arm_joints": right_action.tolist()
        }

        # Step with current commands
        obs_dict, reward, terminated, truncated, info = rpc_server.step_with_commands(metadata)

        # Handle environment reset if needed
        if terminated.any() or truncated.any():
            obs_dict, info = task_env.reset()
            last_time = time.time()  # resync pacing after a reset
            continue

        # Sleep the rest of the control period so env.step() runs at most once per period.
        sleep_time = max(0, control_dt - (time.time() - last_time))
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_time = time.time()

    # Save any remaining data before closing
    teleop_wrapper.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

"""
State-based replay with modality selection for IsaacLab teleoperation datasets.

- Restores simulator to recorded states using ManagerBasedRLEnv.reset_to for each frame.
- Saves ONLY requested observation modalities during replay using the RecorderManager.
- Records actions as absolute joint positions.

Usage:
  python scripts/replay_data.py \
    --task PutPotOnCooktop-v0 \
    --dataset_file /path/to/teleoperation_data.hdf5 \
    --assets_root_path /path/to/assets \
    --output_root /path/to/replay_dataset \
    --observation_modalities="rgb,proprioception" \
    --camera_width 320 \
    --camera_height 240 \
    --image_downsample_factor 1 \
    --task_description "Put the pot on the induction cooktop" \
    --enable_cameras \
    --enable_gripper_clamp \
    --headless
"""

from isaaclab.app import AppLauncher
import argparse
import os

import yamlab
from yamlab import ROOT_DIR

# Parse arguments
parser = argparse.ArgumentParser(description="Replay teleoperation data using ManagerBasedRLEnv")
parser.add_argument("--task", type=str, default="PutPotOnCooktop-v0", help="Name of the task environment")
parser.add_argument("--dataset_file", type=str, default=os.path.join(ROOT_DIR, "recorded_data", "teleoperation_data.hdf5"), 
                    help="File path to load recorded demos")
parser.add_argument("--episode_ids", type=int, nargs="+", default=None, 
                    help="List of episode indices to replay (default: all episodes)")

# Observation collection arguments (replay always saves observations to LeRobot format)
parser.add_argument("--output_root", type=str, default=os.path.join(ROOT_DIR, "recorded_data", "replay_dataset"),
                    help="Root directory to save LeRobot dataset (e.g., ./datasets/replay_dataset)")
parser.add_argument("--observation_modalities", type=str, default="rgb,proprioception",
                    help="Comma-separated observation modalities (default: rgb,proprioception for LeRobot format)")
parser.add_argument("--camera_width", type=int, default=640,
                    help="Camera sensor render width (default 640, matching the real camera). "
                         "Recorded resolution is this // image_downsample_factor.")
parser.add_argument("--camera_height", type=int, default=480,
                    help="Camera sensor render height (default 480; see --camera_width).")
parser.add_argument("--image_downsample_factor", type=int, default=2,
                    help="Recorded image resolution is the sensor resolution // this factor "
                         "(default 2: render 640x480, record 320x240 via cv2.INTER_AREA, "
                         "matching the training data and real-camera downscaling).")
parser.add_argument("--fps", type=int, default=30,
                    help="FPS for dataset (should match simulation decimation rate)")

# Language/task description for multi-task training.
parser.add_argument("--task_description", type=str, default=None,
                    help="Natural-language task instruction saved as each episode's language "
                         "label in the LeRobot dataset (for language-conditioned policy training). "
                         "If omitted, the recorder falls back to the env class name.")

parser.add_argument("--assets_root_path", type=str, default=None,
                    help="Root path containing asset categories. Will be joined with category/instance from ASSETS_INSTANCE_PATHS attribute in dataset.")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                    help="Override an object's asset instance dir, e.g. --asset obj_0=/path (repeatable).")

parser.add_argument("--discard_first_n_frames", type=int, default=4,
                    help="Number of frames to discard from the beginning of each episode when saving")
parser.add_argument("--discard_last_n_frames", type=int, default=1,
                    help="Number of frames to discard from the end of each episode when saving")

# Domain randomization arguments
parser.add_argument("--enable_domain_randomization", action="store_true",
                    help="Enable visual/appearance and/or lighting randomization during replay. "
                         "Requires at least one of --materials_path or --hdris_path.")
parser.add_argument("--hdris_path", type=str, default=None,
                    help="Path to folder containing .hdr files for HDRI lighting randomization")
parser.add_argument("--materials_path", type=str, default=None,
                    help="Path to local materials folder for texture randomization. "
                         "If not set, auto-detects from assets/materials/.")

parser.add_argument("--enable_gripper_clamp", action="store_true", default=False,
                    help="Enable gripper grasp protection: clamp gripper command when grasp detected to prevent over-closing")

# IsaacLab args
AppLauncher.add_app_launcher_args(parser)
# Let an omitted --device fall through to the per-mode YAML default instead of
# AppLauncher's own "cuda:0"; resolve_device fills args.device below.
parser.set_defaults(device=None)
args = parser.parse_args()

# Resolve the sim/app device: an explicit --device wins, else the replay YAML
# default (cpu). Assign back so AppLauncher launches on it, and pass the same
# value to create_task_environment so the sim device matches.
from yamlab.configs import resolve_device, resolve_assets, parse_asset_args
args.device = resolve_device(args.task, "replay", args.device)

# Validate observation modalities before launching the simulator.
from yamlab.utils.observation_modalities import validate_observation_modalities
validate_observation_modalities([m.strip() for m in args.observation_modalities.split(",")])

# Launch IsaacLab
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import contextlib

from isaaclab.utils.datasets import HDF5DatasetFileHandler

# Import our task environment cfg/classes so we can filter modalities at construction time
from yamlab.utils.task_creation import create_task_environment
from yamlab.utils.assets import load_assets_instance_paths_from_dataset


def reset_scene_to_state_safe(scene, state, env_ids, is_relative=False):
    """Reset a scene to a recorded state, restoring poses but skipping velocities.

    This is a wrapper around scene.reset_to() that manually restores articulation,
    rigid, deformable, and gripper poses/positions while skipping velocity setting,
    which can fail for bodies that are kinematic (e.g. while grasped).

    Args:
        scene: The interactive scene to write the state onto.
        state (dict): Recorded state grouped by asset type ("articulation",
            "rigid_object", "deformable_object", "gripper").
        env_ids: Environment indices to write. None applies to all environments.
        is_relative (bool): If True, positions are offset by the per-env origins.
    """
    import torch
    from collections.abc import Sequence
    
    # Resolve env_ids - convert to tensor if it's a list, since PhysX API expects tensor
    # The write methods use env_ids for both indexing (supports list/slice) and PhysX API (needs tensor)
    # Get device from state tensors or scene
    device = None
    if "articulation" in state and len(state["articulation"]) > 0:
        first_art_state = next(iter(state["articulation"].values()))
        if "root_pose" in first_art_state:
            device = first_art_state["root_pose"].device
    elif "rigid_object" in state and len(state["rigid_object"]) > 0:
        first_obj_state = next(iter(state["rigid_object"].values()))
        if "root_pose" in first_obj_state:
            device = first_obj_state["root_pose"].device
    if device is None:
        # Fallback: try to get device from scene or use CPU
        try:
            device = scene.device
        except AttributeError:
            device = torch.device("cpu")
    
    env_ids_for_indexing = env_ids
    env_ids_for_physx = env_ids
    
    if env_ids is None:
        env_ids_for_indexing = scene._ALL_INDICES
        env_ids_for_physx = scene._ALL_INDICES
    elif isinstance(env_ids, torch.Tensor):
        # Keep as tensor for PhysX, convert to list for indexing if needed
        if env_ids.dim() == 1 and len(env_ids) > 0:
            env_ids_for_indexing = env_ids.tolist()
        else:
            env_ids_for_indexing = env_ids
        env_ids_for_physx = env_ids
    elif isinstance(env_ids, Sequence):
        # Convert list/sequence to tensor for PhysX API
        env_ids_for_physx = torch.tensor(env_ids, dtype=torch.int64, device=device)
        env_ids_for_indexing = env_ids
    else:
        # Single integer
        env_ids_for_physx = torch.tensor([env_ids], dtype=torch.int64, device=device)
        env_ids_for_indexing = [env_ids]
    
    # Restore articulations (poses and joint positions, skip velocities)
    if "articulation" in state:
        for asset_name, articulation in scene.articulations.items():
            if asset_name not in state["articulation"]:
                continue
            asset_state = state["articulation"][asset_name]
            # Root pose
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += scene.env_origins[env_ids_for_indexing]
            # Pass tensor version for PhysX API compatibility
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids_for_physx)
            # Joint positions (skip velocities)
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = torch.zeros_like(joint_position)  # Zero velocity
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids_for_physx)
            articulation.set_joint_position_target(joint_position, env_ids=env_ids_for_physx)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids_for_physx)
    
    # Restore deformable objects (positions only, skip velocities)
    if "deformable_object" in state:
        for asset_name, deformable_object in scene.deformable_objects.items():
            if asset_name not in state["deformable_object"]:
                continue
            asset_state = state["deformable_object"][asset_name]
            nodal_position = asset_state["nodal_position"].clone()
            if is_relative:
                nodal_position[:, :3] += scene.env_origins[env_ids_for_indexing]
            deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids_for_physx)
            # Skip velocity setting for deformable objects
    
    # Restore rigid objects (poses only, skip velocities to avoid kinematic body errors)
    if "rigid_object" in state:
        for asset_name, rigid_object in scene.rigid_objects.items():
            if asset_name not in state["rigid_object"]:
                continue
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += scene.env_origins[env_ids_for_indexing]
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids_for_physx)
            # Skip velocity setting for rigid objects (they may be kinematic when grasped)
    
    # Restore surface grippers
    if "gripper" in state:
        for asset_name, surface_gripper in scene.surface_grippers.items():
            if asset_name not in state["gripper"]:
                continue
            asset_state = state["gripper"][asset_name]
            surface_gripper.set_grippers_command(asset_state)
    
    # Write data to simulation
    scene.write_data_to_sim()


def main():
    """Replay episodes loaded from a file and optionally save observations."""
    print("="*80)
    print("ISAACLAB TELEOPERATION DATA REPLAY - MANAGERBASEDRLENV")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Episodes: {args.episode_ids}")
    print(f"Observation modalities: {args.observation_modalities}")
    print(f"Output format: LeRobot dataset")
    print(f"Output: {args.output_root}/")
    print("="*80)
    
    # Check if dataset file exists
    if not os.path.exists(args.dataset_file):
        print(f"[ERROR] Dataset file not found: {args.dataset_file}")
        return
    
    # Asset paths: --asset overrides the dataset-recorded paths (joined with
    # --assets_root_path) for each object.
    dataset_assets = load_assets_instance_paths_from_dataset(
        args.dataset_file, args.assets_root_path
    ) or {}
    assets_instance_paths = resolve_assets(
        args.task,
        cli_assets=parse_asset_args(args.asset),
        dataset_paths=dataset_assets,
    )
    if not assets_instance_paths:
        print("[ERROR] No asset paths (dataset has none and no --asset given). Cannot proceed.")
        return
    
    observation_modalities = [modality.strip() for modality in args.observation_modalities.split(',')]
    # Create the task environment with proper configuration for replay
    # Create environment in replay mode
    task_env = create_task_environment(
        task_name=args.task,
        assets_instance_paths=assets_instance_paths,
        objects_randomization=None,
        init_joint_pos_randomization=0.0,
        mode="replay",  # Replay mode
        device=args.device,  # matches the launched app device (cpu by default for replay)
        observation_modalities=observation_modalities,
        enable_self_collisions=False,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        image_downsample_factor=args.image_downsample_factor,
        task_description=args.task_description,
        enable_domain_randomization=args.enable_domain_randomization,
        hdris_path=args.hdris_path,
        materials_path=args.materials_path,
        discard_first_n_frames=args.discard_first_n_frames,
        discard_last_n_frames=args.discard_last_n_frames,
        enable_gripper_grasp_clamp=args.enable_gripper_clamp,
    )

    obs_dict, info = task_env.reset()

    # Load dataset
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()
    
    if episode_count == 0:
        print("[ERROR] No episodes found in the dataset.")
        return
    
    print(f"[INFO] Found {episode_count} episodes in dataset")
    
    # Determine which episodes to replay
    if args.episode_ids is None:
        # Replay all episodes
        episode_ids_to_replay = list(range(episode_count))
        print(f"[INFO] Replaying all {episode_count} episodes")
    else:
        # Replay specified episodes
        episode_ids_to_replay = args.episode_ids
        available_episodes = list(range(episode_count))
        invalid_episodes = [ep_id for ep_id in episode_ids_to_replay if ep_id not in available_episodes]
        if invalid_episodes:
            print(f"[ERROR] Invalid episode IDs: {invalid_episodes}")
            print(f"[INFO] Available episodes: {available_episodes}")
            return
        print(f"[INFO] Replaying specified episodes: {episode_ids_to_replay}")
    
    # Create task environment
    print(f"[INFO] Creating task environment: {args.task}")
    
    # Configure the LeRobot recorder (replay always saves observations).
    output_name = os.path.basename(args.output_root)
    task_env.configure_for_lerobot_recording(
        output_dir=args.output_root,
        repo_id=f"local/{output_name}",
        fps=args.fps,
        task_name=args.task,
    )
    print(f"[INFO] LeRobot recorder configured")
    
    # Reset environment
    task_env.reset()
    
    # Get episode names
    episode_names = list(dataset_file_handler.get_episode_names())
    

    replayed_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for episode_id in episode_ids_to_replay:
            print(f"\n[INFO] Replaying episode {episode_id}...")
            
            # Load episode data
            episode_data = dataset_file_handler.load_episode(episode_names[episode_id], task_env.device)
            
            # Reset to initial state (fallback to first recorded state if missing)
            initial_state = episode_data.get_initial_state()
            has_initial_state = True
            if initial_state is None:
                has_initial_state = False
                print("[WARNING] No initial state found, using next state")
                initial_state = episode_data.get_next_state()
            
            # Apply initial state directly to the scene using safe reset (skips velocities)
            # This avoids kinematic body errors when objects are grasped
            env_ids = torch.tensor([0], device=task_env.device, dtype=torch.int64)
            reset_scene_to_state_safe(task_env.scene, initial_state, env_ids, is_relative=False)
            task_env.sim.forward()

            task_env.start_lerobot_recording()

            # Iterate over recorded states; for each timestamp:
            # - set state on scene directly (no recorder reset hooks)
            # - step once with action to trigger recorder hooks
            # Frame skipping is handled by LeRobotRecorderManager based on discard_first_n_frames
            # and discard_last_n_frames parameters passed to create_task_environment
            step_count = 0
            if has_initial_state:
                next_state = episode_data.get_next_state()
            else:
                next_state = initial_state
            
            while next_state is not None and simulation_app.is_running() and not simulation_app.is_exiting():
                # Restore state directly on scene using safe reset (skips velocities)
                # This avoids kinematic body errors when objects are grasped
                reset_scene_to_state_safe(task_env.scene, next_state, env_ids, is_relative=False)
                task_env.sim.forward()
                
                # Use recorded action if available; otherwise fall back to dummy actions
                next_action = episode_data.get_next_action()
                if isinstance(next_action, torch.Tensor):
                    if next_action.dim() == 1:
                        next_action = next_action.unsqueeze(0)
                    next_action = next_action.to(task_env.device)
                else:
                    next_action = torch.zeros(task_env.action_space.shape, device=task_env.device)
                
                obs_dict, reward, terminated, truncated, info = task_env.step(next_action)
                # Note: frame recording happens automatically inside step() via
                # base_manager.py's lerobot_recorder.record_frame() — no need to call it here
                
                step_count += 1
                next_state = episode_data.get_next_state()
                
                # Print progress
                if step_count % 100 == 0:  # Print every 100 steps
                    print(f"[INFO] Episode {episode_id} - Step {step_count}")
              
            print(f"[INFO] Episode {episode_id} completed - {step_count} steps")

            # Export this episode now (videos encoded later at consolidation).
            task_env.save_lerobot_episode(encode_videos=False)
            replayed_count += 1
            # task_env.reset_task()
    
    
    # Consolidate LeRobot dataset
    print(f"\n[INFO] Consolidating LeRobot dataset (encoding videos, computing stats)...")
    task_env.consolidate_lerobot_dataset(run_compute_stats=True)
    task_env.close_lerobot_recorder()
    
    # Clean up
    dataset_file_handler.close()
    task_env.close()
    
    print(f"\n[INFO] ✅ Replay completed successfully!")
    print(f"[INFO] Replayed {replayed_count} episodes")
    
    print(f"[INFO] LeRobot dataset saved to: {args.output_root}/")
    print(f"[INFO] Dataset ready for training (no conversion needed)")


if __name__ == "__main__":
    main()
    simulation_app.close()

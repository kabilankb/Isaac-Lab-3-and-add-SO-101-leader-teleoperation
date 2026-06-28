"""
Script to generate additional demonstrations using MimicGen from annotated source demos.

This script:
1. Loads annotated demonstrations (with subtask boundaries)
2. Randomizes object positions to create new scenarios
3. Transforms and stitches subtask segments to generate new trajectories
4. Saves generated demonstrations with full observations

Key Design Decision: Save Full Observations During Generation
- Unlike teleoperation (save states only, replay for observations), data generation
  saves full observations directly because:
  1. Generated trajectories are unique - no way to exactly replay them
  2. MimicGen transforms trajectories based on new object poses
  3. Observations must match the actual generated trajectory

Usage:
  python scripts/mimic/generate_dataset.py \
    --task PutPotOnCooktop-Mimic-v0 \
    --input_file /path/to/annotated_dataset.hdf5 \
    --output_root /path/to/generated_dataset \
    --generation_num_trials 100 \
    --num_envs 10 \
    --assets_root_path /path/to/assets \
    --enable_cameras \
    --observation_modalities="rgb,proprioception" \
    --camera_width 320 \
    --camera_height 240 \
    --image_downsample_factor 1 \
    --task_description "Put the pot on the induction cooktop" \
    --enable_domain_randomization \
    --hdris_path /path/to/HDRIs/indoor/train \
    --materials_path /path/to/materials/train \
    --enable_gripper_clamp \
    --headless
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

# Object randomization configuration: read from configs/tasks/<task>.yaml.
# Edit ranges THERE; all entry scripts (launch_follower.py, generate_dataset.py,
# visualize_*) and the task cfgs read from the same source of truth.
# Per-task object randomization + runtime kwargs are read from the layered
# YAML config (configs/tasks/<task>.yaml) via the loader-backed accessors.
from yamlab.configs import get_objects_randomization, get_task_kwargs, resolve_assets, parse_asset_args

# Parse arguments before launching IsaacLab
parser = argparse.ArgumentParser(description="Generate demonstrations using MimicGen.")
parser.add_argument("--task", type=str, required=True,
                    help="Name of the Mimic task (e.g., PutPotOnCooktop-Mimic-v0)")
parser.add_argument("--input_file", type=str, required=True,
                    help="Path to annotated HDF5 dataset (source demos)")
parser.add_argument("--output_root", type=str, required=True,
                    help="Root directory to save LeRobot dataset (e.g., ./datasets/generated_dataset)")
parser.add_argument("--generation_num_trials", type=int, default=100,
                    help="Number of demonstrations to generate")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of parallel environments to generate with. If omitted, "
                         "uses the mimicgen per-mode default from the YAML config "
                         "(configs/defaults.yaml modes.mimicgen.num_envs). >1 runs "
                         "independent trials in parallel; episodes are written to the "
                         "same LeRobot dataset format (order is not preserved).")
parser.add_argument("--assets_root_path", type=str, default=None,
                    help="Root path containing asset categories. Joined with category/instance from dataset.")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                    help="Override an object's asset instance dir, e.g. --asset obj_0=/path "
                         "(repeatable). Overrides the dataset-recorded path for that object.")
parser.add_argument("--pause_subtask", action="store_true", default=False,
                    help="Pause after each subtask for debugging")
parser.add_argument("--observation_modalities", type=str, default="rgb,proprioception",
                    help="Comma-separated list of observation modalities to record (default: rgb,proprioception)")

parser.add_argument("--camera_width", type=int, default=640,
                    help="Camera sensor render width (default 640, matching the real camera). "
                         "Recorded resolution is this // image_downsample_factor.")
parser.add_argument("--camera_height", type=int, default=480,
                    help="Camera sensor render height (default 480; see --camera_width).")
parser.add_argument("--image_downsample_factor", type=int, default=2,
                    help="Recorded image resolution is the sensor resolution // this factor "
                         "(default 2: render 640x480, record 320x240 via cv2.INTER_AREA, "
                         "matching the training data and real-camera downscaling).")
parser.add_argument("--discard_first_n_frames", type=int, default=4,
                    help="Number of frames to discard from the beginning of each episode when saving")
parser.add_argument("--discard_last_n_frames", type=int, default=1,
                    help="Number of frames to discard from the end of each episode when saving")

# Domain randomization arguments
parser.add_argument("--enable_domain_randomization", action="store_true",
                    help="Enable visual/appearance and/or lighting randomization during data generation. "
                         "Requires at least one of --materials_path or --hdris_path.")
parser.add_argument("--hdris_path", type=str, default=None,
                    help="Path to folder containing .hdr files for HDRI lighting randomization")
parser.add_argument("--materials_path", type=str, default=None,
                    help="Path to local materials folder for texture randomization. "
                         "If not set, auto-detects from assets/materials/.")

# Statistics arguments
parser.add_argument("--remove_stats", action="store_true",
                    help="If True, rename meta/stats.json to meta/_stats.json to force recomputation by training code")

# Language/task description for multi-task training
parser.add_argument("--task_description", type=str, default=None,
                    help="Task description string for language-conditioned multi-task policy training. "
                         "If provided, this will be saved with each episode for RGB+language training. "
                         "Example: 'Put the pot on top of the pan'")
parser.add_argument("--enable_gripper_clamp", action="store_true", default=False,
                    help="Enable gripper grasp protection: clamp gripper command when grasp detected to prevent over-closing")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for generation. Defaults to a time-based seed so "
                         "consecutive runs produce different demos.")

# IsaacLab arguments
AppLauncher.add_app_launcher_args(parser)
# Let an omitted --device fall through to the per-mode YAML default instead of
# AppLauncher's own "cuda:0"; resolve_device fills args.device below.
parser.set_defaults(device=None)
args = parser.parse_args()

# Resolve the sim/app device: an explicit --device wins, else the mimicgen YAML
# default (cpu). Assign back so AppLauncher launches on it, and pass the same
# value to create_task_environment so the sim device matches the app device.
from yamlab.configs import resolve_device, get_task_config
args.device = resolve_device(args.task, "mimicgen", args.device)
# --num_envs falls through to the mimicgen YAML default when omitted; resolve to
# a concrete int here since this script uses args.num_envs directly downstream.
if args.num_envs is None:
    args.num_envs = get_task_config(args.task, "mimicgen")["sim"]["num_envs"]

# Validate observation modalities before launching the simulator.
from yamlab.utils.observation_modalities import validate_observation_modalities
validate_observation_modalities([m.strip() for m in args.observation_modalities.split(",")])

# Launch IsaacLab
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest of the imports after IsaacLab launch."""

import asyncio
import numpy as np
import random
import time
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv

# Import task environments to trigger registration
import yamlab.envs.tasks
import yamlab.envs.mimic

from yamlab.utils.task_creation import create_task_environment
from yamlab.utils.assets import load_assets_instance_paths_from_dataset
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths


def run_mimicgen_generation(env, args):
    """Run MimicGen data generation using the isaaclab_mimic async pipeline.

    Sets the random seed, resets the environment, and drives the async generation
    loop until generation_num_trials trials complete. The environment's task_success
    termination is disabled during generation so episodes can continue after task
    completion (e.g., to record the arms moving away); the success_term function is
    still passed so MimicGen can flag success for statistics.

    Args:
        env (ManagerBasedRLMimicEnv): The Mimic-compatible environment to generate in.
        args (argparse.Namespace): Parsed CLI arguments (seed, num_envs, input_file,
            pause_subtask, etc.).
    """
    from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
    
    # Get success termination term for MimicGen to check success (for statistics)
    # Note: The termination itself is disabled by configure_for_data_generation() so episodes
    # don't end early, but we still need the success_term function for MimicGen
    success_term = None
    if hasattr(env.cfg, 'datagen_config') and hasattr(env.cfg.datagen_config, 'success_term'):
        # Success term was stored in datagen_config by configure_for_data_generation()
        success_term = env.cfg.datagen_config.success_term
    elif hasattr(env.cfg, 'terminations') and hasattr(env.cfg.terminations, 'task_success'):
        # Fallback: try to get from terminations (shouldn't happen if configure_for_data_generation was called)
        success_term = env.cfg.terminations.task_success
    
    if success_term is None:
        print("[WARNING] Could not find success_term for MimicGen. Success checking may not work.")
    
    # Set seed — use CLI flag if provided, otherwise generate a time-based seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(time.time() * 1000) % (2**31)
    print(f"[INFO] Generation seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Reset before starting
    env.reset()
    
    # Setup and run async data generation (one async generator per environment).
    async_components = setup_async_generation(
        env=env,
        num_envs=args.num_envs,
        input_file=args.input_file,
        success_term=success_term,
        pause_subtask=args.pause_subtask,
        motion_planners=None,
    )
    
    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled.")
        except Exception as e:
            print(f"Error cancelling tasks: {e}")


def main():
    """Main function for MimicGen data generation."""
    print("=" * 80)
    print("MIMICGEN DATA GENERATION")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_root}/")
    print(f"Trials: {args.generation_num_trials}")
    if args.task_description:
        print(f"Task Description: {args.task_description}")
    else:
        print(f"Task Description: (not set - single-task policy mode)")
    print(f"Parallel environments: {args.num_envs}")
    print("=" * 80)
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}")
        return
    
    # Asset paths: dataset-recorded (joined with --assets_root_path) by default,
    # overridable per object by --asset.
    dataset_assets = load_assets_instance_paths_from_dataset(args.input_file, args.assets_root_path) or {}
    assets_instance_paths = resolve_assets(
        args.task,
        cli_assets=parse_asset_args(args.asset),
        dataset_paths=dataset_assets,
    )
    if not assets_instance_paths:
        print("[ERROR] No asset paths (dataset has none and no --asset given)")
        return
    
    # Create output directory for LeRobot dataset
    lerobot_output_dir = args.output_root
    output_parent_dir = os.path.dirname(lerobot_output_dir)
    if output_parent_dir and not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)
    
    # Parse observation modalities
    observation_modalities = [m.strip() for m in args.observation_modalities.split(',')]
    
    # Look up task-specific randomization config
    objects_randomization = get_objects_randomization(args.task)

    if objects_randomization:
        print(f"[INFO] Object randomization (region-based):")
        for obj_name, rand_params in objects_randomization.items():
            print(f"  {obj_name}: region_size={rand_params['region_size']}, "
                  f"orientation={math.degrees(rand_params['orientation_range']):.1f}°")
    else:
        print(f"[WARNING] No object randomization config for task '{args.task}'")
    
    # Create environment
    print(f"[INFO] Creating environment: {args.task}")
    print(f"[INFO] Domain randomization: {args.enable_domain_randomization}")
    env = create_task_environment(
        task_name=args.task,
        assets_instance_paths=assets_instance_paths,
        objects_randomization=objects_randomization,
        init_joint_pos_randomization=0.0,
        mode="mimicgen",  # MimicGen data generation mode
        observation_modalities=observation_modalities,
        enable_self_collisions=False,
        num_envs=args.num_envs,
        device=args.device,
        lerobot_output_root=lerobot_output_dir,
        enable_domain_randomization=args.enable_domain_randomization,
        hdris_path=args.hdris_path,
        materials_path=args.materials_path,
        discard_first_n_frames=args.discard_first_n_frames,
        discard_last_n_frames=args.discard_last_n_frames,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        image_downsample_factor=args.image_downsample_factor,
        task_description=args.task_description,  # For multi-task language-conditioned training
        enable_gripper_grasp_clamp=args.enable_gripper_clamp,
        **get_task_kwargs(args.task, mode="mimic"),
    )
    
    # Verify environment is Mimic-compatible
    if not isinstance(env, ManagerBasedRLMimicEnv):
        print(f"[ERROR] Environment is not MimicGen-compatible. Got {type(env)}")
        return
    
    # Update data generation config
    if hasattr(env.cfg, 'datagen_config'):
        env.cfg.datagen_config.generation_num_trials = args.generation_num_trials
        env.cfg.datagen_config.source_dataset_path = args.input_file
        env.cfg.datagen_config.generation_path = args.output_root
    
    # LeRobot recorder is already configured in base_manager.py __init__ when lerobot_output_root/name are provided
    
    # Run generation
    run_mimicgen_generation(env, args)
    
    # Consolidate LeRobot dataset
    print(f"\n[INFO] Consolidating LeRobot dataset (encoding videos, computing stats)...")
    env.consolidate_lerobot_dataset(run_compute_stats=True)
    env.close_lerobot_recorder()
    
    # Optionally remove stats files to force recomputation by training code
    if args.remove_stats:
        meta_dir = os.path.join(lerobot_output_dir, "meta")
        stats_file = os.path.join(meta_dir, "stats.json")

        if os.path.exists(stats_file):
            backup_stats_file = os.path.join(meta_dir, "_stats.json")
            os.rename(stats_file, backup_stats_file)
            print(f"[INFO] Renamed {stats_file} to {backup_stats_file} to force recomputation")
    
    # Cleanup
    env.close()
    
    print(f"\n[INFO] Data generation complete!")
    print(f"[INFO] LeRobot dataset saved to: {lerobot_output_dir}")
    print(f"[INFO] Dataset ready for training (no conversion needed)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    
    simulation_app.close()

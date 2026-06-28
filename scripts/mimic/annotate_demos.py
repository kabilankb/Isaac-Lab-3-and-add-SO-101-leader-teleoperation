"""
Script to add mimic annotations to demos for MimicGen data generation.

This script:
1. Loads recorded teleoperation demos (states + actions)
2. Replays them using STATE-BASED replay (setting state at each step)
3. Records subtask termination signals automatically (--auto) or manually
4. Saves annotated demos for MimicGen generation

IMPORTANT: Uses state-based replay to ensure exact trajectory reproduction.
Physics simulation is non-deterministic, so action-only replay causes drift.

Usage:
  python annotate_demos.py \
    --task PutPotOnCooktop-Mimic-v0 \
    --input_file /path/to/teleoperation_data.hdf5 \
    --output_file /path/to/annotated_dataset.hdf5 \
    --assets_root_path /path/to/assets \
    --enable_cameras \
    --auto \
    --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

# Parse arguments before launching IsaacLab
parser = argparse.ArgumentParser(description="Annotate demonstrations for MimicGen data generation.")
parser.add_argument("--task", type=str, required=True, 
                    help="Name of the Mimic task (e.g., PutPotOnCooktop-Mimic-v0)")
parser.add_argument("--input_file", type=str, required=True,
                    help="Path to input HDF5 dataset with teleoperation recordings")
parser.add_argument("--output_file", type=str, required=True,
                    help="Path to output HDF5 dataset with annotations")
parser.add_argument("--auto", action="store_true", default=False,
                    help="Automatically annotate subtasks using get_subtask_term_signals()")
parser.add_argument("--assets_root_path", type=str, default=None,
                    help="Root path containing asset categories. Joined with category/instance from dataset.")

# IsaacLab arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch IsaacLab
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest of the imports after IsaacLab launch."""

import contextlib
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

# Import task environments to trigger registration
import yamlab.envs.tasks
import yamlab.envs.mimic

from yamlab.utils.recorders import configure_annotation_recorder
from yamlab.utils.task_creation import create_task_environment
from yamlab.utils.assets import load_assets_instance_paths_from_dataset
from yamlab.utils.io import copy_dataset_metadata

# Global state for manual annotation
is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False


def play_cb():
    """Resume replay (keyboard callback for manual annotation)."""
    global is_paused
    is_paused = False


def pause_cb():
    """Pause replay (keyboard callback for manual annotation)."""
    global is_paused
    is_paused = True


def skip_episode_cb():
    """Skip the current episode (keyboard callback for manual annotation)."""
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    """Mark a subtask boundary at the current action index (keyboard callback)."""
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked subtask signal at action index: {current_action_index}")


# ============================================================================
# Replay Functions - STATE-BASED replay for exact trajectory reproduction
# ============================================================================

def replay_episode(env, episode: EpisodeData, success_term=None) -> bool:
    """Replays an episode using STATE-BASED replay.
    
    IMPORTANT: This sets the full simulation state at each step rather than
    relying on physics to reproduce trajectories from actions. This is necessary
    because physics simulation is non-deterministic and action-only replay causes
    objects to slip or trajectories to diverge.
    
    Args:
        env (ManagerBasedRLMimicEnv): The environment to replay the episode in.
        episode (EpisodeData): The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        bool: True if the episode replayed and the success condition was met.
    """
    global current_action_index, skip_episode, is_paused
    
    # Get initial state
    initial_state = episode.get_initial_state()
    if initial_state is None:
        initial_state = episode.data.get("initial_state")
    
    if initial_state is None:
        print("[WARNING] No initial state found in episode")
        return False
    
    # Check if states are available for state-based replay
    has_states = "states" in episode.data
    if not has_states:
        print("[WARNING] No states found in episode - falling back to action-only replay")
        print("[WARNING] This may cause trajectory divergence due to physics non-determinism")
    
    # Reset simulation and recorder
    env.sim.reset()
    if env.recorder_manager is not None:
        env.recorder_manager.reset()
    
    # Set initial state
    env_ids = torch.tensor([0], device=env.device, dtype=torch.int64)
    env.scene.reset_to(initial_state, env_ids, is_relative=False)
    env.sim.forward()
    
    # Reset subtask latch states (CRITICAL for MimicGen)
    # The latch states must be cleared at the start of each episode replay
    # Otherwise signals from previous episode will carry over
    if hasattr(env, 'reset_subtask_latch_states'):
        env.reset_subtask_latch_states(env_ids=None)
    
    # Reset episode data iterators
    episode._next_state_index = 0
    episode._next_action_index = 0
    
    # Skip first state (it's the initial state we already set)
    if has_states:
        _ = episode.get_next_state()
    
    first_action = True
    action_index = 0
    
    while True:
        # Get next state and action
        if has_states:
            next_state = episode.get_next_state()
        else:
            next_state = None
        next_action = episode.get_next_action()
        
        # Check if we've reached the end
        if next_action is None:
            break
        
        current_action_index = action_index
        
        # Handle pause for manual annotation
        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
        
        # STATE-BASED REPLAY: Set full state at each step
        # This ensures object positions match exactly what was recorded
        if next_state is not None:
            env.scene.reset_to(next_state, env_ids, is_relative=False)
            env.sim.forward()
        
        # Prepare action tensor
        if isinstance(next_action, torch.Tensor):
            if next_action.dim() == 1:
                next_action = next_action.unsqueeze(0)
            action_tensor = next_action.to(env.device)
        else:
            action_tensor = torch.tensor(next_action, device=env.device, dtype=torch.float32).unsqueeze(0)
        
        # Step the environment (triggers recorder hooks to capture datagen info)
        env.step(action_tensor)
        
        action_index += 1
    
    # Check success if term provided
    if success_term is not None:
        try:
            result = success_term.func(env, **success_term.params)
            if not bool(result[0]):
                return False
        except Exception as e:
            print(f"[WARNING] Failed to check success term: {e}")
            # Continue anyway - success check is optional for annotation
    
    return True


def annotate_episode_auto(env, episode: EpisodeData, success_term=None) -> bool:
    """Annotates an episode in automatic mode.
    
    Uses get_subtask_term_signals() to automatically detect subtask boundaries.

    Args:
        env (ManagerBasedRLMimicEnv): The environment to replay the episode in.
        episode (EpisodeData): The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        bool: True if the episode was successfully annotated.
    """
    global skip_episode
    skip_episode = False
    
    is_episode_annotated_successfully = replay_episode(env, episode, success_term)
    
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
        return False
    
    # Check if all subtask term signals were detected during replay
    if env.recorder_manager is None:
        print("\t[ERROR] Recorder manager is None")
        return False
    
    try:
        annotated_episode = env.recorder_manager.get_episode(0)
        
        # Check if datagen_info was recorded
        if "obs" not in annotated_episode.data:
            print("\t[ERROR] No observations recorded")
            return False
        
        if "datagen_info" not in annotated_episode.data["obs"]:
            print("\t[ERROR] No datagen_info recorded - recorder may not be set up correctly")
            return False
        
        if "subtask_term_signals" not in annotated_episode.data["obs"]["datagen_info"]:
            print("\t[ERROR] No subtask_term_signals recorded")
            return False
        
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        
        # Verify each signal was detected at least once
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            signal_flags = torch.tensor(signal_flags, device=env.device)
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for subtask "{signal_name}".')
    
    except Exception as e:
        print(f"\t[ERROR] Failed to check annotations: {e}")
        return False
    
    return is_episode_annotated_successfully


def annotate_episode_manual(env, episode: EpisodeData, success_term, subtask_term_signal_names: dict) -> bool:
    """Annotates an episode in manual mode.
    
    The user presses 'S' to mark subtask boundaries during replay.

    Args:
        env (ManagerBasedRLMimicEnv): The environment to replay the episode in.
        episode (EpisodeData): The recorded episode data to replay.
        success_term: Termination term to check for task success.
        subtask_term_signal_names (dict): Mapping from eef name to the list of
            subtask signal names to annotate for that eef.

    Returns:
        bool: True if the episode was successfully annotated.
    """
    global is_paused, marked_subtask_action_indices, skip_episode
    
    subtask_term_signal_action_indices = {}
    
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        if len(eef_subtask_term_signal_names) == 0:
            continue
        
        while True:
            is_paused = True
            skip_episode = False
            
            print(f'\tPlaying episode for subtask annotations for eef "{eef_name}".')
            print(f"\tSubtask signals to annotate: {eef_subtask_term_signal_names}")
            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            
            marked_subtask_action_indices = []
            task_success_result = replay_episode(env, episode, success_term)
            
            if skip_episode:
                print("\tSkipping the episode.")
                return False
            
            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            
            if task_success_result and len(eef_subtask_term_signal_names) == len(marked_subtask_action_indices):
                print(f'\tAll {len(eef_subtask_term_signal_names)} subtask signals for eef "{eef_name}" were annotated.')
                for i, signal_name in enumerate(eef_subtask_term_signal_names):
                    subtask_term_signal_action_indices[signal_name] = marked_subtask_action_indices[i]
                break
            
            if not task_success_result:
                print("\tThe final task was not completed.")
                return False
            
            print(f"\tOnly {len(marked_subtask_action_indices)} out of {len(eef_subtask_term_signal_names)} signals annotated.")
            print(f'\tThe episode will be replayed for re-marking.\n')
    
    # Add manual annotations to episode data
    if env.recorder_manager is None:
        print("\t[ERROR] Recorder manager is None")
        return False
    
    try:
        annotated_episode = env.recorder_manager.get_episode(0)
        num_actions = len(episode.data.get("actions", []))
        
        for signal_name, signal_action_index in subtask_term_signal_action_indices.items():
            # Signal is False before completion, True after
            subtask_signals = torch.ones(num_actions, dtype=torch.bool)
            subtask_signals[:signal_action_index] = False
            annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{signal_name}", subtask_signals)
    
    except Exception as e:
        print(f"\t[ERROR] Failed to add annotations: {e}")
        return False
    
    return True


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to annotate demonstrations."""
    global is_paused
    
    print("=" * 80)
    print("MIMICGEN DEMONSTRATION ANNOTATION")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Auto mode: {args.auto}")
    print("=" * 80)
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}")
        return 0
    
    # Load dataset
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args.input_file)
    episode_count = dataset_file_handler.get_num_episodes()
    
    if episode_count == 0:
        print("[ERROR] No episodes found in dataset.")
        return 0
    
    print(f"[INFO] Found {episode_count} episodes in dataset")
    
    # Load asset paths from dataset
    assets_instance_paths = load_assets_instance_paths_from_dataset(args.input_file, args.assets_root_path)
    if assets_instance_paths is None:
        print("[ERROR] Could not load asset paths from dataset")
        return 0
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if not output_dir:
        output_dir = "."
    output_filename = os.path.splitext(os.path.basename(args.output_file))[0]
    if output_dir != "." and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create environment via the standard path (handles contact sensors, asset
    # paths, termination disabling, etc. generically for any task).
    print(f"[INFO] Creating environment: {args.task}")
    env = create_task_environment(
        task_name=args.task,
        assets_instance_paths=assets_instance_paths,
        mode="replay",  # replay mode: num_envs=1, terminations disabled
    )

    # Replace the default recorder with annotation-specific one
    configure_annotation_recorder(env, output_dir, output_filename, args.auto)
    
    # Verify environment is Mimic-compatible
    if not isinstance(env, ManagerBasedRLMimicEnv):
        print(f"[ERROR] Environment is not MimicGen-compatible. Got {type(env)}")
        return 0
    
    # Verify get_subtask_term_signals is implemented for auto mode
    if args.auto:
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            print("[ERROR] Environment does not implement get_subtask_term_signals()")
            print("[ERROR] Automatic annotation requires this method. Use manual mode instead.")
            return 0
    
    # Get subtask term signal names for manual annotation
    subtask_term_signal_names = {}
    if not args.auto:
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal 
                for subtask_config in eef_subtask_configs
                if subtask_config.subtask_term_signal is not None
            ]
        print(f"[INFO] Manual annotation mode - signals to annotate: {subtask_term_signal_names}")
    
    # Setup keyboard controls for manual annotation
    if not args.auto and not args.headless:
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        keyboard.add_callback("N", play_cb)
        keyboard.add_callback("B", pause_cb)
        keyboard.add_callback("Q", skip_episode_cb)
        keyboard.add_callback("S", mark_subtask_cb)
        keyboard.reset()
        print("[INFO] Keyboard controls: N=play, B=pause, S=mark subtask, Q=skip")
    
    # Reset environment
    env.reset()
    
    # Annotate episodes
    exported_count = 0
    processed_count = 0
    
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
            processed_count += 1
            print(f"\n[INFO] Annotating episode #{episode_index} ({episode_name})")
            
            episode = dataset_file_handler.load_episode(episode_name, env.device)
            
            # Annotate based on mode
            if args.auto:
                is_annotated = annotate_episode_auto(env, episode, success_term=None)
            else:
                is_annotated = annotate_episode_manual(env, episode, success_term=None, 
                                                        subtask_term_signal_names=subtask_term_signal_names)
            
            if is_annotated and not skip_episode:
                # Set success flag and export
                if env.recorder_manager is not None:
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                exported_count += 1
                print(f"\t✓ Exported annotated episode.")
            else:
                print(f"\t✗ Skipped exporting episode.")
    
    print(f"\n{'='*80}")
    print(f"[INFO] Annotation complete!")
    print(f"[INFO] Exported {exported_count}/{processed_count} annotated episodes")
    print(f"[INFO] Output saved to: {output_dir}/{output_filename}.hdf5")
    print(f"{'='*80}")
    
    # Copy metadata from input dataset to output dataset
    output_file_path = os.path.join(output_dir, f"{output_filename}.hdf5")
    if os.path.exists(output_file_path):
        print(f"\n[INFO] Copying metadata from input to output dataset...")
        copy_dataset_metadata(args.input_file, output_file_path)
    else:
        print(f"[WARNING] Output file not found, skipping metadata copy: {output_file_path}")
    
    # Cleanup
    dataset_file_handler.close()
    env.close()
    
    return exported_count


if __name__ == "__main__":
    successful_count = main()
    simulation_app.close()
    exit(successful_count)

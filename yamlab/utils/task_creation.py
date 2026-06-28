"""Entry point for creating task environments via gym.make()."""
import os
import gymnasium as gym

import yamlab.envs.tasks  # (side-effect: registers task gym envs)
import yamlab.envs.mimic  # (side-effect: registers MimicGen gym envs)

def create_task_environment(
    task_name: str,
    assets_instance_paths: dict[str, str],
    objects_randomization: dict[str, dict] = None,
    init_joint_pos_randomization: float = 0.0,
    mode: str = "teleoperation",  # "teleoperation", "replay", "mimicgen", or "evaluation"
    enable_pose_schedule: bool = False,
    observation_modalities: list = [],
    enable_self_collisions: bool = False,
    num_envs: int | None = None,
    device: str | None = None,
    timeout_steps: int | None = None,
    lerobot_output_root: str | None = None,
    # Domain randomization parameters
    enable_domain_randomization: bool = False,
    hdris_path: str | None = None,
    materials_path: str | None = None,  # Path to local materials (auto-detected if None and DR enabled)
    use_unseen_materials: bool = False,  # For evaluation: use the held-out eval material split if True
    # Frame skipping parameters for LeRobot recording
    discard_first_n_frames: int = 0,  # Number of frames to discard from beginning of each episode
    discard_last_n_frames: int = 0,  # Number of frames to discard from end of each episode
    # Language/task description for multi-task training
    task_description: str | None = None,  # Task description for language-conditioned policies
    # Gripper grasp protection
    enable_gripper_grasp_clamp: bool = False,  # Clamp gripper command on grasp to prevent over-closing
    # Teleop grasp-ray visualization (on-screen beam only; teleoperation mode only)
    enable_grasp_ray_viz: bool = True,  # Show the green/red inter-finger grasp-ray beam
    **task_specific_kwargs
):
    """Create a task environment with new format parameters.
    
    This is a generic utility function that accepts common parameters and passes
    task-specific parameters via **task_specific_kwargs. This allows the function
    to work with any task environment without modification.
    
    Args:
        task_name: Name of the task environment (e.g., "<Task>-v0")
        assets_instance_paths: Dictionary mapping asset names to asset instance folder paths
        objects_randomization: Dictionary mapping object names to region-based randomization parameters.
            Format: {"obj_name": {"region_size": (x, y), "orientation_range": float, "scale_range": (min, max)}}
            region_size defines a bounding box (meters) centered on the object's default position.
            The effective randomization range is computed as max(0, region_size/2 - max_world_half_extent)
            where max_world_half_extent accounts for the object's local dimensions (from asset_size.json)
            rotated over the full yaw range [default_yaw ± orientation_range].
        init_joint_pos_randomization: Joint position randomization range
        mode: Operation mode - one of "teleoperation", "replay", "mimicgen", or "evaluation"
        enable_pose_schedule: If True, enable pose schedule for teleoperation. The pose schedule
            path should be defined in the task environment config. Objects listed in the schedule
            will follow a fixed sequence of poses instead of randomization. The pose advances
            after each successful demo save (X button), and stays at current pose on discard/reset
            (HOME button). Only used when mode="teleoperation".
        observation_modalities: List of observation modalities
        enable_self_collisions: Whether to enable self collisions
        num_envs: Number of parallel environments. If None (default), the
                 per-mode default from the YAML config is used (teleop/replay 1,
                 mimicgen 4, evaluation 64). An explicit value overrides it.
                 Replay always forces 1 regardless.
        device: Sim device ("cpu" / "cuda:0"). If None (default), the per-mode
                 default from the YAML config is used (teleop/replay/mimicgen
                 cpu, evaluation cuda). Pass the AppLauncher --device value so
                 the sim device matches the launched app device.
        timeout_steps: Maximum number of steps per episode for timeout termination (IsaacLab handles this per-env).
                      If None, uses default episode_length_s from config.
        lerobot_output_root: Root directory for LeRobot dataset output (e.g., ./datasets/generated_dataset)
        enable_domain_randomization: Whether to enable visual/appearance and lighting randomization.
                                    Disabled for teleoperation (clean demos), enabled for replay/mimicgen/evaluation.
                                    When enabled, at least one of hdris_path or materials_path must be set.
        hdris_path: Path to folder containing .hdr files for HDRI-based lighting randomization.
                         If provided, lighting randomization is enabled.
        materials_path: Path to local materials folder. If provided, material randomization is enabled.
                             If None but DR is enabled, auto-detects from assets/materials/.
        use_unseen_materials: If True, use the held-out eval material split instead of train (evaluation only)
        discard_first_n_frames: Number of frames to skip from the beginning of each episode when saving LeRobot dataset
        discard_last_n_frames: Number of frames to skip from the end of each episode when saving LeRobot dataset
        task_description: Task description string for language-conditioned multi-task policy training.
                         If provided, this will be saved with each episode in the LeRobot dataset.
                         Example: "Put the target object on top of the base object"
        **task_specific_kwargs: Additional task-specific parameters that will be passed
                               to the task environment constructor. These should be defined
                               in TASK_SPECIFIC_KWARGS in the launch script.
                               Examples:
                               - pick_by_two_hands (for a pick task): If True, stage1
                                 success requires contact with both arms' grippers.
                               - any_other_param (for another task): Task-specific parameter
    
    Returns:
        Unwrapped task environment instance
    """
    # Validate mode
    valid_modes = ["teleoperation", "replay", "mimicgen", "evaluation"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
    
    from yamlab.envs.robot_registry import robot_for_task
    print(f"[INFO] Creating task environment: {task_name} (mode: {mode}, robot: {robot_for_task(task_name)})")
    
    # Set mode-specific variables
    teleoperation = (mode == "teleoperation")
    policy_evaluation = (mode == "evaluation")
    data_generation = (mode == "mimicgen")
    
    # Mode-specific settings
    if mode == "replay":
        # Replay mode: enforce same settings as teleoperation
        num_envs = 1  # Force single environment
        timeout_steps = None  # No timeout termination
        # Task success termination will be disabled in configure_for_replay()
    
    # Build base kwargs with common parameters
    env_kwargs = {
        'assets_instance_paths': assets_instance_paths,
        'init_joint_pos_randomization': init_joint_pos_randomization,
        'mode': mode,  # Pass mode instead of individual boolean flags
        'enable_pose_schedule': enable_pose_schedule,  # For teleoperation pose schedule
        'observation_modalities': observation_modalities,
        'enable_self_collisions': enable_self_collisions,
        'num_envs': num_envs,
        'device': device,
        'timeout_steps': timeout_steps,
        'env_name': task_name,  # Pass env_name for task-relevant object detection in domain randomization
        'enable_gripper_grasp_clamp': enable_gripper_grasp_clamp,
        'enable_grasp_ray_viz': enable_grasp_ray_viz,
    }
    
    # Add domain randomization parameters
    # Domain randomization is disabled for teleoperation (clean demos)
    # and enabled for replay/mimicgen/evaluation
    #
    # When enable_domain_randomization is set, determine which modalities to randomize:
    # - materials_path provided (or auto-detected) → material randomization
    # - hdris_path provided → lighting randomization
    # At least one must be set.
    material_randomization = False
    if not teleoperation and enable_domain_randomization:
        # Auto-detect local materials if materials_path not explicitly provided
        if materials_path is None:
            import yamlab
            default_mat_dir = os.path.join(os.path.dirname(yamlab.__file__), "..", "assets", "materials")
            default_mat_dir = os.path.abspath(default_mat_dir)
            if os.path.isdir(default_mat_dir):
                materials_path = default_mat_dir

        material_randomization = materials_path is not None
        lighting_randomization = hdris_path is not None

        if not material_randomization and not lighting_randomization:
            raise ValueError(
                "--enable_domain_randomization requires at least one of "
                "--materials_path or --hdris_path to be set "
                "(or local materials in assets/materials/)."
            )

    if not teleoperation:
        env_kwargs['enable_domain_randomization'] = enable_domain_randomization
        env_kwargs['hdris_path'] = hdris_path
        env_kwargs['materials_dir'] = materials_path
        env_kwargs['material_randomization'] = material_randomization
        # use_unseen_materials only for evaluation
        if policy_evaluation:
            env_kwargs['use_unseen_materials'] = use_unseen_materials
    
    # Add frame skipping parameters (for replay and mimicgen modes)
    if mode in ["replay", "mimicgen"]:
        env_kwargs['discard_first_n_frames'] = discard_first_n_frames
        env_kwargs['discard_last_n_frames'] = discard_last_n_frames
    
    # Handle data_generation mode - configure for MimicGen
    if data_generation:
        print(f"[INFO] Configuring environment for MimicGen data generation")
        # For data generation, we use LeRobot recording (not HDF5)
        if lerobot_output_root:
            env_kwargs['lerobot_output_root'] = lerobot_output_root
        # Disable timeout for data generation (MimicGen handles episode length)
        env_kwargs['timeout_steps'] = None
        # IMPORTANT: propagate data_generation flag so yam_bimanual_env can configure MimicGen mode
        env_kwargs['data_generation'] = True
        # Enable domain randomization for MimicGen (unless explicitly disabled)
        env_kwargs['enable_domain_randomization'] = enable_domain_randomization
        env_kwargs['hdris_path'] = hdris_path
        env_kwargs['material_randomization'] = material_randomization
        # Pass task_description for language-conditioned training
        if task_description:
            env_kwargs['task_description'] = task_description
            print(f"[INFO] Task description for multi-task training: {task_description}")
        print(f"[INFO] Domain randomization for MimicGen: {enable_domain_randomization}")
        if enable_domain_randomization:
            print(f"[INFO]   Material randomization: {material_randomization}")
            print(f"[INFO]   Lighting randomization: {hdris_path is not None}")
    
    # Add any task-specific kwargs
    env_kwargs.update(task_specific_kwargs)

    # Object randomization. Each object sets its XY range in ONE of two modes:
    #   region_size  -> bounding box; config subtracts the (yaw-swept) object half-extent
    #   position_range -> direct +/- distance for the center (no size subtraction)
    # Asset size (asset_size.json) is loaded only for bounding-box mode.
    if objects_randomization is not None:
        from yamlab.utils.assets import load_asset_size
        objects_rand_tuple = {}
        for obj_name, rand_params in objects_randomization.items():
            region_size = rand_params.get('region_size')
            position_range = rand_params.get('position_range')
            assert (region_size is None) != (position_range is None), (
                f"Object '{obj_name}' must set exactly one of 'region_size' (bounding box) or "
                f"'position_range' (direct +/- distance) under randomization")
            assert 'orientation_range' in rand_params, f"Missing 'orientation_range' for randomizing object '{obj_name}'"
            assert 'scale_range' in rand_params, f"Missing 'scale_range' for randomizing object '{obj_name}'"

            asset_sx, asset_sy = 0.0, 0.0
            if region_size is not None:
                asset_path = assets_instance_paths.get(obj_name)
                if asset_path:
                    size_info = load_asset_size(asset_path)
                    if size_info and 'size' in size_info:
                        asset_sx = size_info['size'].get('x', 0.0)
                        asset_sy = size_info['size'].get('y', 0.0)
                    else:
                        print(f"[WARNING] No asset_size.json for {obj_name}, treating object size as (0, 0)")

            objects_rand_tuple[obj_name] = (
                tuple(region_size) if region_size is not None else None,
                (asset_sx, asset_sy),
                rand_params['orientation_range'],
                tuple(rand_params['scale_range']),
                tuple(position_range) if position_range is not None else None,
            )
        env_kwargs['objects_randomization'] = objects_rand_tuple
        
    task_env = gym.make(task_name, **env_kwargs)
    return task_env.unwrapped

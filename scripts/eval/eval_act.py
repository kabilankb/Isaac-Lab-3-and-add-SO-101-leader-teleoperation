"""
Evaluate a trained ACT checkpoint on a YAM bimanual task and report the success rate.

Creates the task environment in evaluation mode, runs the policy closed-loop until
``--num_episodes`` episodes have finished, and prints per-episode and overall results.
An episode succeeds when the task's success check ends it and fails when it reaches
``--timeout_steps``. Works for checkpoints from ``scripts/train/train_act_m3.py``, with
or without M3 masking, since both are plain ACT checkpoints.

Usage (GUI; drop --viz kit and add --headless to run without a window):
  D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
  python scripts/eval/eval_act.py \
    --checkpoint outputs/train/act_m3/checkpoints/last/pretrained_model \
    --task PutPotOnCooktop-v0 \
    --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 \
    --num_episodes 10 --enable_gripper_clamp --enable_cameras --viz kit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an ACT checkpoint on a YAM bimanual task")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Checkpoint folder holding config.json and model.safetensors (.../pretrained_model)")
parser.add_argument("--task", type=str, default="PutPotOnCooktop-v0", help="Task environment to evaluate on")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                    help="Object asset instance dir, e.g. --asset pot=/path/to/pot "
                         "(repeatable, one per object). Overrides objects.<name>.asset in the task YAML.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--timeout_steps", type=int, default=900,
                    help="Control steps before an episode counts as a failure")
parser.add_argument("--n_action_steps", type=int, default=None,
                    help="Actions executed from each predicted chunk (default: the checkpoint's value)")
parser.add_argument("--camera_width", type=int, default=320, help="Camera sensor render width")
parser.add_argument("--camera_height", type=int, default=240, help="Camera sensor render height")
parser.add_argument("--image_downsample_factor", type=int, default=1,
                    help="Observation image resolution is the sensor resolution // this factor")
parser.add_argument("--init_joint_pos_randomization", type=float, default=0.0,
                    help="Per-reset arm joint randomization range (+/- radians)")
parser.add_argument("--enable_gripper_clamp", action="store_true", default=False,
                    help="Clamp the gripper command on grasp, as in data generation")

# Domain randomization
parser.add_argument("--enable_domain_randomization", action="store_true", default=False,
                    help="Enable material + lighting randomization")
parser.add_argument("--hdris_path", type=str, default=None,
                    help="Folder of .hdr files for HDRI lighting randomization")
parser.add_argument("--materials_path", type=str, default=None,
                    help="Local materials folder for texture randomization (auto-detected if omitted)")
parser.add_argument("--use_unseen_materials", action="store_true", default=False,
                    help="Use the held-out eval material split (out-of-distribution appearance)")

# IsaacLab arg passthrough adds --device / --headless / --enable_cameras / --viz.
AppLauncher.add_app_launcher_args(parser)
# Let an omitted --device fall through to the per-mode YAML default instead of
# AppLauncher's own "cuda:0"; resolve_device fills args.device below.
parser.set_defaults(device=None)
args = parser.parse_args()

from yamlab.configs import (resolve_device, resolve_assets, parse_asset_args,
                              get_objects_randomization, get_task_kwargs)

args.device = resolve_device(args.task, "evaluation", args.device)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import draccus
import torch

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy

from yamlab.robot.spec import get_robot
from yamlab.utils.task_creation import create_task_environment

_ROBOT_SPEC = get_robot("yam")


def load_policy(checkpoint: str, device: str) -> ACTPolicy:
    """Load an ACT checkpoint saved by the LeRobot fork's training script."""
    # The fork writes config.json without a "type" key, so parse it as an ACTConfig directly.
    with draccus.config_type("json"):
        config = draccus.parse(ACTConfig, f"{checkpoint}/config.json", args=[])
    if args.n_action_steps is not None:
        config.n_action_steps = args.n_action_steps
    policy = ACTPolicy.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()
    return policy


def build_policy_batch(obs_dict: dict, policy: ACTPolicy, device: str) -> dict:
    """Convert an environment observation into the batch the policy was trained on.

    Mirrors ``LeRobotRecorderManager._build_frame``: the state is
    ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`` and each camera's
    ``<name>_camera_rgb`` becomes ``observation.images.<lerobot_key>`` as float RGB in [0, 1].
    """
    policy_obs = obs_dict.get("policy", obs_dict)
    state = torch.cat(
        [
            policy_obs["left_arm_joint_pos"],
            policy_obs["left_gripper_state"].reshape(-1, 1),
            policy_obs["right_arm_joint_pos"],
            policy_obs["right_gripper_state"].reshape(-1, 1),
        ],
        dim=-1,
    )
    batch = {"observation.state": state.to(device=device, dtype=torch.float32)}
    for name in _ROBOT_SPEC.camera_names:
        key = f"observation.images.{_ROBOT_SPEC.camera_lerobot_key(name)}"
        if key not in policy.config.image_features:
            continue
        image = policy_obs[f"{name}_camera_rgb"].to(device)
        if image.dtype == torch.uint8:
            image = image.float() / 255
        elif image.max() > 1.0:
            image = image.float() / 255
        batch[key] = image.float().permute(0, 3, 1, 2)  # (N, H, W, 3) -> (N, 3, H, W)
    missing = set(policy.config.image_features) - set(batch)
    if missing:
        raise KeyError(f"The environment does not provide the policy inputs {sorted(missing)}.")
    return batch


def main():
    device = args.device
    assets_instance_paths = resolve_assets(args.task, cli_assets=parse_asset_args(args.asset))
    if not assets_instance_paths:
        raise SystemExit(
            "No object assets specified. Pass --asset pot=/path/to/instance (one per object), "
            "or set objects.<name>.asset in the task YAML."
        )

    policy = load_policy(args.checkpoint, device)
    print(f"[EVAL] Loaded ACT checkpoint {args.checkpoint} "
          f"(chunk {policy.config.chunk_size}, executes {policy.config.n_action_steps} actions per chunk)")

    task_kwargs = get_task_kwargs(args.task, "evaluation")
    task_kwargs.update(timeout_steps=args.timeout_steps, enable_gripper_grasp_clamp=args.enable_gripper_clamp)
    task_env = create_task_environment(
        task_name=args.task,
        assets_instance_paths=assets_instance_paths,
        objects_randomization=get_objects_randomization(args.task),
        init_joint_pos_randomization=args.init_joint_pos_randomization,
        mode="evaluation",
        observation_modalities=["rgb", "proprioception"],
        num_envs=args.num_envs,
        device=device,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        image_downsample_factor=args.image_downsample_factor,
        enable_domain_randomization=args.enable_domain_randomization,
        hdris_path=args.hdris_path,
        materials_path=args.materials_path,
        use_unseen_materials=args.use_unseen_materials,
        **task_kwargs,
    )

    obs_dict, _ = task_env.reset()
    policy.reset()
    num_envs = task_env.num_envs

    results = []  # one dict per finished episode
    episode_steps = torch.zeros(num_envs, dtype=torch.long)
    picked = torch.zeros(num_envs, dtype=torch.bool)
    while simulation_app.is_running() and len(results) < args.num_episodes:
        action = policy.select_action(build_policy_batch(obs_dict, policy, device))
        obs_dict, _, terminated, truncated, _ = task_env.step(action.to(task_env.device))
        episode_steps += 1
        if hasattr(task_env, "stage1_success"):
            picked |= task_env.stage1_success.cpu()

        done = (terminated | truncated).cpu()
        if not done.any():
            continue
        for env_id in done.nonzero().flatten().tolist():
            if len(results) >= args.num_episodes:
                break
            success = bool(terminated[env_id]) and not bool(truncated[env_id])
            results.append({"success": success, "pick": bool(picked[env_id]) or success,
                            "steps": int(episode_steps[env_id])})
            print(f"[EVAL] episode {len(results)}/{args.num_episodes}: "
                  f"{'SUCCESS' if success else 'FAIL'} "
                  f"(pick {'yes' if results[-1]['pick'] else 'no'}, {results[-1]['steps']} steps)")
            episode_steps[env_id] = 0
            picked[env_id] = False
        # The finished environments were reset, so the queued actions no longer apply.
        policy.reset()

    if results:
        n = len(results)
        n_success = sum(r["success"] for r in results)
        n_pick = sum(r["pick"] for r in results)
        print("=" * 60)
        print(f"[EVAL] {args.task}: {n_success}/{n} episodes succeeded ({100 * n_success / n:.1f}%), "
              f"{n_pick}/{n} reached pick ({100 * n_pick / n:.1f}%)")
        print("=" * 60)

    task_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

"""
Evaluation throughput benchmark for the YAM bimanual environment.

Creates the full task environment in evaluation mode (the same environment a real
policy evaluation runs), steps it with a swappable action source, and prints
startup and per-step throughput statistics. It doubles as a template: replace the
body of ``compute_actions`` with your own policy inference; the surrounding code
(env creation, observation handling, timing) stays the same.

The default action source is a deterministic, model-free joint sweep, so the
benchmark runs with no policy or checkpoint.

Usage:
  python scripts/benchmark_eval_throughput.py \
    --task PutPotOnCooktop-v0 \
    --asset pot=/path/to/pot \
    --asset cooktop=/path/to/cooktop \
    --num_envs 128 \
    --observation_modalities "rgb,proprioception" \
    --camera_width 320 \
    --camera_height 240 \
    --image_downsample_factor 1 \
    --enable_cameras \
    --headless
"""

import time

from isaaclab.app import AppLauncher
import argparse

# ---- Benchmark constants (not CLI flags) ----------------------------
NUM_STEPS = 100          # timed environment steps
WARMUP_STEPS = 10        # untimed steps before measurement (let caches/JIT settle)
OBS_BENCH_STEPS = 30     # isolated observation-extraction micro-benchmark iterations
ACTION_DIM = 14          # [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]

parser = argparse.ArgumentParser(description="YAM evaluation-mode throughput benchmark")
parser.add_argument("--task", type=str, default="PutPotOnCooktop-v0", help="Task environment to benchmark")
parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                    help="Object asset instance dir, e.g. --asset pot=/path/to/pot "
                         "(repeatable, one per object). Overrides objects.<name>.asset in the task YAML.")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of parallel environments (default: the evaluation YAML value)")
parser.add_argument("--observation_modalities", type=str, default="rgb,proprioception",
                    help="Comma-separated observation modalities to enable (default: rgb,proprioception)")
parser.add_argument("--camera_width", type=int, default=640, help="Camera sensor render width")
parser.add_argument("--camera_height", type=int, default=480, help="Camera sensor render height")
parser.add_argument("--image_downsample_factor", type=int, default=2,
                    help="Observation image resolution is the sensor resolution // this factor")
parser.add_argument("--init_joint_pos_randomization", type=float, default=0.0,
                    help="Per-reset arm joint randomization range (+/- radians)")

# Domain randomization
parser.add_argument("--enable_domain_randomization", action="store_true", default=False,
                    help="Enable material + lighting randomization (as in a real evaluation)")
parser.add_argument("--hdris_path", type=str, default=None,
                    help="Folder of .hdr files for HDRI lighting randomization")
parser.add_argument("--materials_path", type=str, default=None,
                    help="Local materials folder for texture randomization (auto-detected if omitted)")
parser.add_argument("--use_unseen_materials", action="store_true", default=False,
                    help="Use the held-out eval material split (out-of-distribution appearance)")

# IsaacLab arg passthrough adds --device / --headless / --enable_cameras.
AppLauncher.add_app_launcher_args(parser)
# Let an omitted --device fall through to the per-mode YAML default instead of
# AppLauncher's own "cuda:0"; resolve_device fills args.device below.
parser.set_defaults(device=None)
args = parser.parse_args()

# Resolve the device (explicit --device wins, else the evaluation YAML default,
# cuda) and validate observation modalities BEFORE launching the simulator.
from yamlab.configs import (resolve_device, resolve_assets, parse_asset_args,
                              get_objects_randomization, get_task_kwargs)
from yamlab.utils.observation_modalities import validate_observation_modalities

args.device = resolve_device(args.task, "evaluation", args.device)
observation_modalities = [m.strip() for m in args.observation_modalities.split(",")]
validate_observation_modalities(observation_modalities)

# ---- Launch the simulator (timed) -----------------------------------
_t = time.perf_counter()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
T_APP_LAUNCH = time.perf_counter() - _t

# ---- Post-launch imports (timed) ------------------------------------
_t = time.perf_counter()
import numpy as np
import torch

from yamlab.utils.task_creation import create_task_environment
T_IMPORTS = time.perf_counter() - _t


def compute_actions(obs_dict, step_idx, num_envs, device):
    """Return the action for the current step as a (num_envs, ACTION_DIM) tensor.

    EDIT THIS to call your own policy: read what you need from ``obs_dict`` and
    return a (num_envs, 14) tensor laid out as
    ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`` (arm entries
    are joint-position targets; gripper entries are binary open/close commands).

    The default below is a model-free, deterministic sinusoidal joint sweep so the
    benchmark runs with no policy. It ignores ``obs_dict``.

    Args:
        obs_dict (dict): Latest observation dict from ``env.reset()`` / ``env.step()``.
        step_idx (int): Current step index.
        num_envs (int): Number of parallel environments.
        device (str): Device the action tensor must live on.

    Returns:
        torch.Tensor: Action of shape ``(num_envs, ACTION_DIM)``.
    """
    action = torch.zeros((num_envs, ACTION_DIM), device=device)
    # Gentle low-amplitude sweep applied to all 6 joints of each arm.
    sweep = 0.3 * float(np.sin(2.0 * np.pi * step_idx / 50.0))
    action[:, 0:6] = sweep      # left arm joints
    action[:, 7:13] = sweep     # right arm joints
    # Grippers held open (column 6 = left, column 13 = right).
    action[:, 6] = 1.0
    action[:, 13] = 1.0
    return action


def _sync(device):
    """Block until queued CUDA work finishes, so timers capture execution, not launch."""
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()


def _stats(times_ms):
    """Return mean/median/p95/min/max of a list of millisecond timings."""
    a = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()),
    }


def main():
    device = args.device

    # ---- Create the evaluation environment (timed) ------------------
    assets_instance_paths = resolve_assets(args.task, cli_assets=parse_asset_args(args.asset))
    if not assets_instance_paths:
        raise SystemExit(
            "No object assets specified. Pass --asset pot=/path/to/instance (one per object), "
            "or set objects.<name>.asset in the task YAML."
        )

    _t = time.perf_counter()
    task_env = create_task_environment(
        task_name=args.task,
        assets_instance_paths=assets_instance_paths,
        objects_randomization=get_objects_randomization(args.task),
        init_joint_pos_randomization=args.init_joint_pos_randomization,
        mode="evaluation",
        observation_modalities=observation_modalities,
        num_envs=args.num_envs,
        device=device,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        image_downsample_factor=args.image_downsample_factor,
        enable_domain_randomization=args.enable_domain_randomization,
        hdris_path=args.hdris_path,
        materials_path=args.materials_path,
        use_unseen_materials=args.use_unseen_materials,
        **get_task_kwargs(args.task, "evaluation"),
    )
    t_env_creation = time.perf_counter() - _t

    # ---- First reset (timed) ----------------------------------------
    _t = time.perf_counter()
    obs_dict, _ = task_env.reset()
    _sync(device)
    t_first_reset = time.perf_counter() - _t

    num_envs = task_env.num_envs
    step_dt = float(task_env.step_dt)          # seconds per control step
    control_hz = 1.0 / step_dt

    # ---- Warmup (untimed) -------------------------------------------
    for i in range(WARMUP_STEPS):
        action = compute_actions(obs_dict, i, num_envs, device)
        obs_dict, _, _, _, _ = task_env.step(action)
    _sync(device)

    # Capture steady-state peak memory over the timed window only.
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # ---- Timed stepping loop ----------------------------------------
    step_times, action_times = [], []
    for i in range(NUM_STEPS):
        _sync(device); _t = time.perf_counter()
        action = compute_actions(obs_dict, i, num_envs, device)
        _sync(device); action_times.append((time.perf_counter() - _t) * 1e3)

        _sync(device); _t = time.perf_counter()
        obs_dict, _, _, _, _ = task_env.step(action)
        _sync(device); step_times.append((time.perf_counter() - _t) * 1e3)

    # ---- Isolated observation-extraction micro-benchmark ------------
    # Times the post-render gather of the observation tensors, 
    # separate from the physics/render cost inside step().
    obs_times = []
    if hasattr(task_env, "observation_manager"):
        try:
            for _ in range(OBS_BENCH_STEPS):
                _sync(device); _t = time.perf_counter()
                task_env.observation_manager.compute()
                _sync(device); obs_times.append((time.perf_counter() - _t) * 1e3)
        except Exception as e:
            print(f"[WARNING] observation-extraction micro-benchmark skipped: {e}")

    peak_alloc = peak_reserved = None
    if isinstance(device, str) and device.startswith("cuda"):
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    # ---- Derived throughput numbers ---------------------------------
    s = _stats(step_times)
    achieved_per_env = 1000.0 / s["mean"]                 # steps / s / env
    effective = achieved_per_env * num_envs               # env-steps / s
    realtime_factor = achieved_per_env / control_hz

    _print_report(
        device=device, num_envs=num_envs, control_hz=control_hz, step_dt=step_dt,
        t_app=T_APP_LAUNCH, t_imports=T_IMPORTS, t_env=t_env_creation, t_reset=t_first_reset,
        step_stats=s, action_times=action_times, obs_times=obs_times,
        achieved_per_env=achieved_per_env, effective=effective, realtime_factor=realtime_factor,
        peak_alloc=peak_alloc, peak_reserved=peak_reserved,
    )

    task_env.close()
    simulation_app.close()


def _print_report(*, device, num_envs, control_hz, step_dt, t_app, t_imports, t_env, t_reset,
                  step_stats, action_times, obs_times, achieved_per_env, effective,
                  realtime_factor, peak_alloc, peak_reserved):
    """Print the human-readable summary, then a compact JSON dump of the same numbers."""
    s = step_stats
    a_mean = float(np.mean(action_times)) if action_times else 0.0
    o_mean = float(np.mean(obs_times)) if obs_times else None
    t_startup_total = t_app + t_imports + t_env + t_reset

    line = "=" * 60
    print(f"\n{line}")
    print(" YAM Eval Throughput Benchmark")
    print(line)
    print(f" Task            : {args.task}")
    print(f" Device          : {device}        Num envs: {num_envs}")
    print(f" Obs modalities  : {', '.join(observation_modalities)}")
    dr = "on" if args.enable_domain_randomization else "off"
    if args.enable_domain_randomization:
        dr += f" (materials={'eval' if args.use_unseen_materials else 'train'} split, hdri={'yes' if args.hdris_path else 'no'})"
    print(f" Domain rand     : {dr}")
    print(f" Render          : {args.camera_width}x{args.camera_height} -> downsample "
          f"{args.image_downsample_factor} -> "
          f"{args.camera_width // args.image_downsample_factor}x{args.camera_height // args.image_downsample_factor}")
    print(f" Action source   : builtin joint sweep (edit compute_actions() to plug a policy)")
    print(f" Window          : {NUM_STEPS} timed steps (+{WARMUP_STEPS} warmup)")
    print("-" * 60)
    print(" Startup (one-time)")
    print(f"   App launch ............ {t_app:7.2f} s")
    print(f"   Python imports ........ {t_imports:7.2f} s")
    print(f"   Env creation + scene .. {t_env:7.2f} s")
    print(f"   First reset ........... {t_reset:7.2f} s")
    print(f"   Total ................. {t_startup_total:7.2f} s")
    print("-" * 60)
    print(" Per-step runtime           mean   median    p95     min     max   (ms)")
    print(f"   env.step() ........... {s['mean']:7.2f} {s['median']:7.2f} {s['p95']:7.2f} "
          f"{s['min']:7.2f} {s['max']:7.2f}")
    if o_mean is not None:
        print(f"   obs extraction (gather){o_mean:7.2f}   (isolated; subset of env.step)")
    print(f"   action compute ....... {a_mean:7.2f}   (builtin sweep; your policy goes here)")
    print("-" * 60)
    print(" Throughput")
    print(f"   Target control rate ... {control_hz:7.2f} Hz  (dt {step_dt:.5f} s)")
    print(f"   Achieved per-env ...... {achieved_per_env:7.2f} steps/s")
    print(f"   Effective ............. {effective:7.1f} env-steps/s  (x{num_envs} envs)")
    print(f"   Realtime factor ....... {realtime_factor:7.2f}x")
    if peak_alloc is not None:
        print(f"   Peak GPU memory ....... {peak_alloc:7.2f} GB allocated / {peak_reserved:.2f} GB reserved")
    print(line)

    import json
    summary = {
        "task": args.task, "device": device, "num_envs": num_envs,
        "observation_modalities": observation_modalities,
        "enable_domain_randomization": args.enable_domain_randomization,
        "use_unseen_materials": args.use_unseen_materials,
        "render": {"width": args.camera_width, "height": args.camera_height,
                   "image_downsample_factor": args.image_downsample_factor},
        "num_steps": NUM_STEPS, "warmup_steps": WARMUP_STEPS,
        "startup_s": {"app_launch": t_app, "imports": t_imports, "env_creation": t_env,
                      "first_reset": t_reset, "total": t_startup_total},
        "step_ms": s,
        "obs_extraction_ms_mean": o_mean,
        "action_compute_ms_mean": a_mean,
        "throughput": {"control_hz": control_hz, "achieved_per_env_steps_s": achieved_per_env,
                       "effective_env_steps_s": effective, "realtime_factor": realtime_factor},
        "peak_gpu_gb": {"allocated": peak_alloc, "reserved": peak_reserved},
    }
    print(" JSON:")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

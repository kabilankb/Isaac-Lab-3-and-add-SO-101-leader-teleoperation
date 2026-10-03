# YAMLab on Isaac Sim 6 / Isaac Lab 3

Port of [ARISE-Initiative/yamlab](https://github.com/ARISE-Initiative/yamlab) from Isaac Sim 5.1 /
Isaac Lab 2.3 to Isaac Sim 6.0.1 / Isaac Lab 3.0 (2026-09-27).

## Why

Upstream YAMLab targets Isaac Sim 5.1. On this machine (RTX PRO 5000 Blackwell laptop, NVIDIA
driver **595.84**) every Isaac Sim 5.1 launch segfaults in `librtx.scenedb.plugin.so` while the
RTX renderer starts. Docker does not help, because the container uses the host driver. Isaac
Sim 6 runs on this driver, so YAMLab was ported to it instead of downgrading to driver 580.

## Layout

| Path | What |
|---|---|
| `~/yamlab` | Upstream checkout plus the original `yam_lab` env (Isaac Sim 5.1, cannot launch on driver 595) |
| `~/yamlab6` | **The port.** Git worktree on branch `isaacsim6-port`, not committed yet |
| `~/yamlab6/yamlab_datasets` | Symlink to the downloaded HF dataset `yamlab/yamlab_datasets` |
| `~/yamlab6/.port/` | Scratch space: smoke test `bench.sh`, `verify_replay.py`, logs, outputs |
| `env_yamlab6` | Conda env: a clone of `env_isaaclab` with YAMLab added |
| `~/IsaacLab` | Isaac Lab 3 (`release/3.0.0-beta2`), used as-is and not modified |

### Environment (`env_yamlab6`)

- Cloned from `env_isaaclab`: Python 3.12, Isaac Sim 6.0.1, Isaac Lab 3, torch 2.10+cu128, numpy 2.3.
- `yamlab`, `joylo` and the `lerobotv2.0` LeRobot fork are installed with `pip install --no-deps -e`,
  so pip cannot downgrade the Isaac/torch stack. YAMLab only uses `LeRobotDataset` from LeRobot.
- The runtime deps were installed with the core stack pinned: datasets, av, jsonlines, coacd,
  pymeshlab, open3d, portal, hid, dynamixel-sdk, and others.
- `libhidapi` comes from conda-forge. Without it `isaaclab_tasks` and JoyLo's `hid` fail to import.
- `ffmpeg`/`ffprobe` are symlinks to the `yam_lab` env's binaries (x264 + SVT-AV1). A conda
  ffmpeg install would add `libglvnd`/`libegl` to the env's library path, where they could
  shadow the system GL/Vulkan loaders the RTX renderer uses.

## How to run

```bash
conda activate env_yamlab6     # needed: puts ffprobe on PATH for LeRobot video encoding
cd ~/yamlab6

# Replay a published teleop demo in the GUI (one env)
python scripts/replay_data.py --task PutPotOnCooktop-v0 \
  --dataset_file yamlab_datasets/tasks_data/PutPotOnCooktop/teleop/putpot_000.hdf5 --episode_ids 0 \
  --assets_root_path yamlab_datasets/tasks_data/PutPotOnCooktop/objects \
  --output_root /tmp/replay_out --camera_width 320 --camera_height 240 --image_downsample_factor 1 \
  --enable_gripper_clamp --enable_cameras --viz kit

# Parallel eval throughput benchmark (headless)
D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
python scripts/benchmark_eval_throughput.py --task PutPotOnCooktop-v0 \
  --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 --num_envs 4 \
  --observation_modalities rgb,proprioception --camera_width 320 --camera_height 240 \
  --image_downsample_factor 1 --enable_cameras --headless
```

> **GUI note:** Isaac Lab 3 no longer opens a viewer just because `--headless` is left off. You
> have to pass `--viz kit`. Without it the run is headless.

## Method

The port followed the official agent skills:

- Isaac Lab `isaaclab-migrating-2x-to-3x` (`.agents/skills` on `develop`), which routes to
  `docs/source/migration/migrating_to_isaaclab_3-0.rst`, `scripts/tools/find_quaternions.py`
  and the `WARN_ON_TORCH_QUATF_ACCESS` runtime detector.
- Isaac Sim `isaac-sim-migration` (`.claude/skills` on `develop`), which provides the static audit
  `audit_isaac_sim_6_0.py`.

Steps:

1. **Static audits.** The Isaac Sim 6.0 audit found nothing blocking: its only hits were Isaac Lab
   recorder-manager false positives and the Python 3.12 note. So the work was mostly Isaac Lab
   2.3 → 3.0.
2. **Iterative smoke test** with the eval benchmark until it ran.
3. **Full read-through** of every file that touches Isaac Lab data. Isaac Lab 3's breaking changes
   (XYZW quaternions, `ProxyArray` data, read-only concatenated state buffers) mostly fail
   *silently*, so a green smoke test proves little.
4. **Diffed Isaac Lab Mimic 2.3 vs 3.0** for every internal that YAMLab monkeypatches.
5. **Runtime detectors.** No Isaac Lab deprecation warnings remain from YAMLab code. The quaternion
   detector flags exactly three reads, and all three were triaged as XYZW-correct.

## Changes

23 files changed (+168/−127), plus the new `yamlab/utils/lab3.py`.

### Quaternion order (WXYZ → XYZW)

Runtime code is now XYZW end to end. Legacy WXYZ values are converted where they enter:

- **`yamlab/utils/lab3.py` (new):** `quat_wxyz_to_xyzw`, `quat_xyzw_to_wxyz`, and `as_torch`
  (unwraps a `ProxyArray`).
- **`robot/spec.py`:** `arm_quaternion()` and `camera_quaternion_opengl()` convert the `yam.yaml`
  values, which stay WXYZ as upstream. These feed the arm bases and the camera offsets.
- **Task configs:** the pot, cooktop, mug and mug-tree default rotations were reordered. Each was
  checked by its yaw: −90°, 135°, 15°.
- **`manipulation_env_cfg.py` and `yam_bimanual_env.py`:** default yaw is now read from XYZW.
- **`manipulation_env.py`:** pose-schedule YAML `rot: [w, x, y, z]` is converted when loaded.
- **`utils/grasp.py`:** the teleop grasp-beam fallback quaternions were converted.
- **`utils/transforms.euler2quat`:** the docstring now says XYZW, since Isaac Lab math returns XYZW.
- **Legacy HDF5 demos:** no YAMLab code needed. Isaac Lab 3's `HDF5DatasetFileHandler.load_episode`
  treats files without a `format_version` attribute as WXYZ and converts every `root_pose`. New
  files are stamped `format_version=1`. MimicGen `datagen_info` stores 4×4 matrices, which have
  no quaternion order.

### Silent bugs fixed (no error on Isaac Lab 3, wrong behavior)

| Where | Bug | Effect |
|---|---|---|
| `utils/task_logic.check_ontop_success` | Tilt computed from `quat[:,1], quat[:,2]` (the WXYZ x, y) | An upright pot at −90° yaw measured 90° tilt, so **PutPotOnCooktop could never succeed**. Fixed to `[:,0], [:,1]` and verified: 0° upright, 20° for a 20° roll |
| `manipulation_env._apply_scheduled_poses_to_default_state` | Wrote into `data.default_root_state` | In Lab 3 that property is rebuilt from pose + velocity on every read, so the write was dropped and **pose schedules did nothing**. Now writes `default_root_pose.torch` |
| Task cfgs / spec | WXYZ literals interpreted as XYZW | Objects, arm bases and cameras would have spawned wrongly rotated |

### Crashes fixed

| Where | Issue | Fix |
|---|---|---|
| `yam_bimanual_env_cfg.py` | `sim_utils.PhysxCfg` / `sim.physx` removed | `isaaclab_physx.physics.PhysxCfg` via `SimulationCfg.physics=` (and `cfg.sim.physics.*` elsewhere) |
| `scripts/replay_data.py` | `write_root_pose_to_sim`, `write_joint_state_to_sim`, `set_joint_*_target` removed | `_index` variants with keyword args |
| `utils/physics_events.py` | `root_physx_view.get_material_properties().clone()`: Sim 6 returns a `wp.array` | Warp↔torch helpers, as Isaac Lab 3's own event does. Mass/inertia now go through `set_masses_index` / `set_inertias_index` |
| `utils/mimic_patches.drop_unused_trajectory_buffers` | Lab 3's `MultiWaypoint.execute` returns a `bool`, so the wrapper's `.get()` would crash MimicGen on the first waypoint | Detects the new contract and skips itself (Lab 3 already fixed the memory growth upstream) |
| `envs/mimic/yam_mimic_env.py` | `root_physx_view.get_jacobians()` returns warp | `data.body_com_jacobian_w.torch`: the same raw COM-referenced Jacobian as 2.x, so the IK behaves as upstream |
| `utils/assets.get_asset_usd_path` | A relative `--assets_root_path` produced a relative USD reference, which USD resolves against the stage, giving an empty pot prim and a contact-sensor `ValueError` | Now resolves to an absolute path. Also an upstream bug |

### API modernisation (worked through deprecation bridges)

- **Data reads:** every asset/sensor `.data.*` read now uses explicit `.torch`: robot grasp
  detection, task success, mimic env, teleop server, perception terms. Implicit use goes through
  a deprecation bridge that also bypasses the quaternion detector.
- **Setters and materials:** `set_joint_position_target` → `set_joint_position_target_index`.
  The gripper material cfg uses `PhysxRigidBodyMaterialCfg`.
- **`scripts/mimic/generate_dataset.py`:** passes `data_gen_tasks=` to `env_loop`, so a failing
  generation task raises instead of hanging.
- **Kept on purpose:** the deprecated schema aliases (`sim_utils.RigidBodyPropertiesCfg`, etc.)
  remain until Isaac Lab 4.0. They are thin subclasses of the `Physx*` cfgs, per the skill's
  smallest-change rule.

## Verification

| Test | Result |
|---|---|
| Isaac Sim 5.1 stock install (`yam_lab`), 4-env benchmark | Segfault in `librtx.scenedb` (the driver-595 issue) |
| Isaac Sim 6 port: eval benchmark, 4 envs, RGB | Runs, about 40 env-steps/s at 320×240 |
| Same, with domain randomization (eval materials + HDRIs) | Runs. 36 MDL variants and 10 HDRIs loaded |
| `replay_data.py` on legacy WXYZ teleop demos (PutPotOnCooktop) | All 9 episodes replay. Episode 0 reaches **pick → place success**. LeRobot dataset + 3 camera MP4s written |
| Visual check of the recorded top-camera video | Pot upright on the table, grasped by both handles, placed on the cooktop |
| Same replay command with relative paths, GUI (`--viz kit`) | Works after the `assets.py` fix |
| `find_quaternions.py --likely-wxyz` on `yamlab/` and `scripts/` | No remaining candidates |

### Not yet tested

- **MimicGen `annotate_demos.py` and `generate_dataset.py`:** the code paths were reviewed and
  patched, but have not been run.
- **HangMugOnTree:** reviewed and patched, not run. It includes the runtime friction handover event.
- **Open-loop (physics-only) replay:** replay resets to the recorded state every step, so
  physics parity between PhysX 5 and Isaac Sim 6's PhysX is not proven yet.
- **Teleoperation with real leader hardware** (JoyLo or SO-101). The follower and SO-101 client are verified with mock leaders only.

## SO-101 leader control (added 2026-09-27)

The two YAM arms can now be driven by two **LeRobot SO-101 leader arms**. This replaces the
JoyLo leader and keeps the same follower RPC and HDF5 recording. The full guide is in
[`joylo/README_SO101.md`](joylo/README_SO101.md).

- **New files:**
  - `joylo/joylo/agents/so101_agent.py`: Feetech STS3215 reader and joint-to-joint mapper.
  - `joylo/scripts/calibrate_so101.py`: rest pose, measured joint signs, gripper range.
  - `joylo/scripts/launch_so101.py`: teleop loop with SPACE/r/q recording keys, `--mock`, `--dry_run`.
  - `joylo/configs/so101_yam_mapping.yaml`: the joint mapping.
- **Mapping:** pan/lift/elbow/wrist-flex/wrist-roll/gripper → YAM joint1/2/3/4/6/fingers.
  YAM joint5 (wrist yaw) is held at 0, because the SO-101 has no such joint. Positive YAM
  directions were derived from the joint axes in `yam.usd`, and calibration measures each SO-101
  sign against them.
- **Environment:** `feetech-servo-sdk` was added to `env_yamlab6`.
- **More Isaac Lab 3 fixes found by running the teleop follower:**
  - `SimulationContext.add_render_callback` is missing in 3.0-beta2, so the grasp-beam overlay
    now registers into `sim._render_callbacks` (`yam_bimanual_env.py`).
  - The follower's `get_joint_pos` reports grippers as binary 0/1, so the SO-101 start-up blend
    only blends arm joints.
- **Verified with mock leaders** against the live GUI follower (`launch_follower.py` on Isaac
  Sim 6): the arms track the commands, and a recorded demo is a valid `format_version=1` dataset.
  **Not yet run on real SO-101 hardware.**
- **Note:** `launch_follower.py` output is block-buffered when redirected to a file, so check
  `ss -ltn | grep 11333` to see whether it is up.

## Known caveats

- These are Isaac Sim 6 physics, not the physics upstream tuned against. Data generated here may
  differ slightly from the published datasets.
- Isaac Sim 6 dropped the PhysX particle cloth and deformable bodies. YAMLab's tasks are rigid-body
  only, so they are unaffected.
- `env_yamlab6` depends on `yam_lab` for `ffmpeg`/`ffprobe` through the symlinks. Keep that env,
  or install a static ffmpeg on `PATH`.
- `~/IsaacLab` has uncommitted Robotis OMX work, and the port only imports from it.

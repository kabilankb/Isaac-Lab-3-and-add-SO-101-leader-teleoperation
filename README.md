# YAMLab on Isaac Sim 6: with SO-101 leader-arm teleoperation

This checkout (`~/yamlab6`, branch `isaacsim6-port`) is
[ARISE-Initiative/yamlab](https://github.com/ARISE-Initiative/yamlab), a bimanual YAM robot
framework for data collection, MimicGen data generation and parallel evaluation. It has been
ported from Isaac Sim 5.1 / Isaac Lab 2.3 to **Isaac Sim 6.0.1 / Isaac Lab 3.0**, and extended so
the two simulated YAM arms can be teleoperated with **two LeRobot SO-101 leader arms**.

The port exists because Isaac Sim 5.1 segfaults at launch on this laptop's NVIDIA driver
(595.84), even in Docker. Isaac Sim 6 runs on it.

| Document | Contents |
|---|---|
| **This README** | Overview, setup, and the commands you need |
| [`TELEOP_QUICKSTART.md`](TELEOP_QUICKSTART.md) | Run sheet for this laptop's two SO-101 leaders (serials, ports, steps) |
| [`joylo/README_SO101.md`](joylo/README_SO101.md) | Full SO-101 leader guide: mapping, calibration, troubleshooting |
| [`PORT_ISAACSIM6.md`](PORT_ISAACSIM6.md) | Engineering notes: every change made for Isaac Sim 6, and how it was verified |
| [`README_UPSTREAM.md`](README_UPSTREAM.md) | Upstream YAMLab README: tasks, parameters, pipeline details |

---

## What works

| Capability | Status on Isaac Sim 6 |
|---|---|
| Replay of recorded / published teleop demos → LeRobot dataset (3 camera videos) | ✅ Verified. PutPotOnCooktop pick + place succeed |
| GUI viewing (`--viz kit`) | ✅ |
| Parallel eval throughput benchmark, with and without domain randomization | ✅ About 40 env-steps/s with 4 envs, RGB |
| Teleop follower (`launch_follower.py`) + recording to HDF5 | ✅ |
| **SO-101 leader teleop**: two leaders (bimanual) or one (switch arms with `a`) | ✅ Both leaders detected and calibrated |
| MimicGen annotate / generate | ⚠️ Code ported and reviewed, not yet run end to end |
| HangMugOnTree task | ⚠️ Ported, not yet run |
| JoyLo (Dynamixel + JoyCon) leader | ⚠️ Unchanged from upstream, untested (no hardware) |

---

## Setup

Everything is already installed on this laptop. For reference:

| Item | Location |
|---|---|
| Code | `~/yamlab6` (git worktree of `~/yamlab`, branch `isaacsim6-port`, not committed) |
| Conda env | `env_yamlab6`: Python 3.12, Isaac Sim 6.0.1, Isaac Lab 3 (`~/IsaacLab`), torch 2.10+cu128 |
| Datasets | `~/yamlab6/yamlab_datasets` → `~/yamlab/yamlab_datasets` (HF `yamlab/yamlab_datasets`) |
| Leader SDK | `feetech-servo-sdk` (in `env_yamlab6`) |

Always start with:

```bash
cd ~/yamlab6
conda activate env_yamlab6   # also puts ffmpeg/ffprobe on PATH for video encoding
```

**Datasets.** The object assets, teleop demos, annotated demos, HDRIs and materials are complete.
The pre-generated LeRobot training sets are only partly downloaded (the download stopped on a
network error). To resume it:

```bash
cd ~/yamlab && uvx --from huggingface_hub python scripts/download_data.py --all --local-dir ~/yamlab/yamlab_datasets
```

---

## 1. Watch a demo (GUI)

```bash
python scripts/replay_data.py --task PutPotOnCooktop-v0 \
  --dataset_file yamlab_datasets/tasks_data/PutPotOnCooktop/teleop/putpot_000.hdf5 --episode_ids 0 \
  --assets_root_path yamlab_datasets/tasks_data/PutPotOnCooktop/objects \
  --output_root /tmp/replay_out --camera_width 320 --camera_height 240 --image_downsample_factor 1 \
  --enable_gripper_clamp --enable_cameras --viz kit
```

- Leave off `--episode_ids` to replay all 9 demos.
- Swap `--viz kit` for `--headless` to run without a window.
- The output is a LeRobot dataset in `--output_root`.

> **`--viz kit` is required for a window.** Isaac Lab 3 no longer opens one just because
> `--headless` is omitted. The window is titled "Isaac Lab 3.0.0".

## 2. Teleoperate with SO-101 leader arms

### Leaders on this laptop

| Leader | Serial device | Calibrated |
|---|---|---|
| Left | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085532-if00` | ✅ |
| Right | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C116959-if00` | ✅ |

```bash
L=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085532-if00
R=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C116959-if00
python joylo/scripts/scan_so101.py      # health check: each leader should print "OK: all 6 SO-101 servos"
```

### Start (two terminals)

**Terminal 1** starts the simulated YAM arms. Wait until `ss -ltn | grep 11333` shows a line.

```bash
D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
python joylo/scripts/launch_follower.py --task PutPotOnCooktop-v0 \
  --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 --port 11333 \
  --demos_per_asset 25 --data_output_dir ./recorded_data --data_output_filename so101_demos \
  --enable_gripper_clamp --enable_cameras --viz kit
```

**Terminal 2** starts the leaders. Use a real terminal window, because the keys are read from it.

```bash
python joylo/scripts/launch_so101.py --left_port $L --right_port $R --enable_recording
```

| Key (terminal 2) | Action |
|---|---|
| `SPACE` | start recording, then press again to **save** the demo (the task resets for the next one) |
| `r` | discard the current recording and reset the task |
| `a` | *single-leader mode only:* switch which YAM arm the leader drives |
| `q` | quit |

- **On start,** the sim arms ease onto the leader poses over 1.5 s.
- **Task progress** (`pick` / `place` / `task_success`) prints live. Save when `task_success: True`.
- **Demos** go to `recorded_data/so101_demos.hdf5`.
- **One leader only:** pass just `--left_port` (or `--right_port`). It drives one YAM arm, and `a` switches arms.

### How the leader maps onto the YAM

| SO-101 joint | → YAM joint |
|---|---|
| shoulder_pan | joint1 (base yaw) |
| shoulder_lift | joint2 (shoulder) |
| elbow_flex | joint3 (elbow) |
| wrist_flex | joint4 (wrist pitch) |
| *(none)* | joint5 (wrist yaw), **held at 0°** |
| wrist_roll | joint6 (wrist roll) |
| gripper | fingers (open −4.75 cm … closed 0) |

The mapping is joint-to-joint, relative to a shared folded rest pose, and calibration measures the
direction of every joint. The arms have different link lengths, so watch the sim window rather
than your own hand. Per-joint `scale` / `ref_deg` tweaks go in `joylo/configs/so101_yam_mapping.yaml`.

### Re-calibrating a leader

This is needed after reassembling a leader or swapping boards. It's interactive, so run it in a
terminal. Each leader is merged into `joylo/configs/so101_calibration.json` separately.

```bash
python joylo/scripts/calibrate_so101.py --left_port $L     # or --right_port $R, or both
```

The steps are: hold the folded rest pose, then do five "move this joint that way" checks, then
open and close the gripper fully.

### Stopping

1. Press `q` in terminal 2.
2. Close the Isaac Sim window.

If the follower doesn't exit, force-stop it: `pkill -9 -f launch_follower.py`.

## 3. Use the recorded demos

```bash
# replay a recorded demo in the GUI
python scripts/replay_data.py --task PutPotOnCooktop-v0 --dataset_file recorded_data/so101_demos.hdf5 \
  --episode_ids 0 --assets_root_path yamlab_datasets/tasks_data/PutPotOnCooktop/objects \
  --output_root /tmp/my_replay --enable_gripper_clamp --enable_cameras --viz kit

# MimicGen: annotate subtasks, then generate more data (ported, not yet run end to end)
python scripts/mimic/annotate_demos.py --task PutPotOnCooktop-Mimic-v0 \
  --input_file recorded_data/so101_demos.hdf5 --output_file recorded_data/so101_annotated.hdf5 \
  --assets_root_path yamlab_datasets/tasks_data/PutPotOnCooktop/objects --auto --enable_cameras --headless
```

Recorded files are standard Isaac Lab 3 datasets (`format_version=1`, XYZW quaternions). The
older published demos (WXYZ) are converted automatically when loaded.

## 4. Benchmark parallel evaluation

```bash
D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
python scripts/benchmark_eval_throughput.py --task PutPotOnCooktop-v0 \
  --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 --num_envs 4 \
  --observation_modalities rgb,proprioception --camera_width 320 --camera_height 240 \
  --image_downsample_factor 1 --enable_cameras --headless
```

Put your own policy into `compute_actions()` in that script to turn it into a real evaluation. Add
`--enable_domain_randomization --hdris_path yamlab_datasets/HDRIs/indoor/eval --materials_path
yamlab_datasets/materials --use_unseen_materials` for held-out appearance.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No sim window | Add `--viz kit`, then check Alt+Tab for the "Isaac Lab 3.0.0" window with the tabletop |
| `ffprobe` not found at the end of a replay | `conda activate env_yamlab6` (don't call the env's python without activating it) |
| `Calibration … not found` | Run `calibrate_so101.py` in a terminal and answer every prompt |
| `servo id 1 … did not answer ping` | Run `scan_so101.py`. `NO REPLY` means check the 5 V supply and cable. `GARBLED` means check the board's jumper and cable |
| Teleop `Connection refused` | The follower isn't ready yet. Wait for port 11333 |
| A YAM joint moves the wrong way | Recalibrate that leader, moving each joint exactly as prompted |
| An HDF5 file can't be opened | The follower still has it open. Stop it, or read it with `HDF5_USE_FILE_LOCKING=FALSE` |
| Follower won't close | `pkill -9 -f launch_follower.py` |

## Known limitations

- **Physics:** Isaac Sim 6 physics are not the ones upstream tuned against, so newly generated
  data may differ slightly from the published datasets.
- **Wrist yaw:** YAM joint5 isn't teleoperable with the SO-101, which has no wrist-yaw joint.
- **Untested:** MimicGen generation and HangMugOnTree have been ported but not yet run end to end.
- **Local only:** the port is not committed. `~/IsaacLab` is used as-is and contains your
  uncommitted Robotis OMX work.

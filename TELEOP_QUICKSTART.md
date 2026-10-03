# YAMLab teleop quick start: SO-101 leader → simulated YAM arms (Isaac Sim 6)

A run sheet for this laptop's setup. Background: `PORT_ISAACSIM6.md` (the Isaac Sim 6 port) and
`joylo/README_SO101.md` (the full leader-control guide).

## Your hardware

| Leader | Board (USB serial) | Device path | Status |
|---|---|---|---|
| **Left** | `5AE6085532` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085532-if00` (usually `/dev/ttyACM1`) | Working, calibrated |
| **Right** | `5A7C116959` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C116959-if00` (usually `/dev/ttyACM0`) | Working (replaced the faulty `5AE6081549` board) |

Both leaders work, so use **bimanual mode** (below). Single-leader mode (`a` switches arms) is
still available when only one leader is connected.

Check the boards at any time:

```bash
python joylo/scripts/scan_so101.py            # every connected board
```

A healthy leader prints `OK: all 6 SO-101 servos` and each servo's position.

## Every session

Open a terminal and run:

```bash
cd ~/yamlab6
conda activate env_yamlab6
L=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085532-if00
R=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C116959-if00
```

### 1. Calibrate (first time only, or after reassembling the leader)

```bash
python joylo/scripts/calibrate_so101.py --left_port $L
```

Answer each prompt in the terminal. Each one waits for ENTER:

1. **Rest pose.** Fold the leader like the YAM start pose: upper arm lying back, forearm folded
   forward on top, wrist straight, gripper pointing forward.
2. **Five direction checks.** Move the named joint about 30–45° in the stated direction, hold,
   press ENTER, then go back to rest.
3. **Gripper.** Fully open and press ENTER, then fully close and press ENTER.

It saves `joylo/configs/so101_calibration.json`.

### 2. Start the simulated arms (terminal 1)

```bash
D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
python joylo/scripts/launch_follower.py --task PutPotOnCooktop-v0 \
  --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 --port 11333 \
  --demos_per_asset 25 --data_output_dir ./recorded_data --data_output_filename so101_demos \
  --enable_gripper_clamp --enable_cameras --viz kit
```

- Keep `--viz kit`. Without it, Isaac Lab 3 runs with no window.
- It's ready when `ss -ltn | grep 11333` shows a line (about 30 s). Its own log output is delayed.
- The window is titled "Isaac Lab 3.0.0". If you have other Isaac sessions open, pick the one
  showing the tabletop with the pot.

### 3. Start teleop (terminal 2, a real terminal window)

```bash
python joylo/scripts/launch_so101.py --left_port $L --enable_recording
```

The leader drives the **left** YAM arm first; add `--start_arm right` to begin on the right.
Keys in this terminal:

| Key | Action |
|---|---|
| `a` | switch which YAM arm the leader drives (the other arm holds its pose) |
| `SPACE` | start recording, then press again to **save** the demo (the task resets) |
| `r` | discard the recording and reset the task |
| `q` | quit |

- **On start and on each switch,** the driven arm eases onto the leader's pose over 1.5 s. Hold
  the leader roughly in that arm's pose before pressing `a`.
- **YAM joint5 (wrist yaw)** stays at 0°, because the SO-101 has no such joint.
- **Task progress lines** (`pick: … place: …`) show live. Save when `task_success: True`.
- **Demos** go to `recorded_data/so101_demos.hdf5`.

### 4. Stop

Press `q` in terminal 2, then close the Isaac Sim window, which also stops terminal 1.

## Bimanual (both leaders)

Calibrate each leader once (they merge into one file), then drive both YAM arms at once:

```bash
python joylo/scripts/calibrate_so101.py --left_port $L --right_port $R
python joylo/scripts/launch_so101.py --left_port $L --right_port $R --enable_recording
```

## After recording

```bash
# watch a saved demo in the GUI
python scripts/replay_data.py --task PutPotOnCooktop-v0 --dataset_file recorded_data/so101_demos.hdf5 \
  --episode_ids 0 --assets_root_path yamlab_datasets/tasks_data/PutPotOnCooktop/objects \
  --output_root /tmp/my_replay --enable_gripper_clamp --enable_cameras --viz kit
```

The file is a standard Isaac Lab 3 dataset, so it also feeds `scripts/mimic/annotate_demos.py` →
`generate_dataset.py` (MimicGen).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Calibration … not found` | Run step 1 in a terminal and answer all the prompts |
| `Cannot open SO-101 leader port` | The port is in use: close `scan_so101.py` / other teleop runs, or replug the leader |
| `servo id 1 … did not answer ping` | Check the servo power (5 V supply) and cable, then run `scan_so101.py` |
| A YAM joint moves the wrong way | Re-run calibration and move that joint exactly as the prompt says |
| Motion feels too big or too small | Change that joint's `scale` in `joylo/configs/so101_yam_mapping.yaml` |
| No sim window | Add `--viz kit` to the follower, and check Alt+Tab for the tabletop window |
| `Connection refused` in teleop | The follower is not up yet. Wait for port 11333 |

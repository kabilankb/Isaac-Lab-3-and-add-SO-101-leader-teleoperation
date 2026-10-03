# SO-101 leader control for the bimanual YAM (Isaac Sim 6)

This drives the two simulated YAM arms with two **LeRobot SO-101 leader arms**, one leader per
YAM arm. It replaces the JoyLo leader (`launch_joylo.py`) and uses the same follower
(`launch_follower.py`), the same RPC command and the same HDF5 recording. Demos collected with
it go straight into the replay → annotate → MimicGen pipeline.

```
 SO-101 leader (left)  ─┐   USB serial (Feetech STS3215, 1 Mbps)
 SO-101 leader (right) ─┤
                        ▼
  joylo/scripts/launch_so101.py ── portal RPC :11333 ──► joylo/scripts/launch_follower.py
  (read servos → map → 14-dim cmd)                        (Isaac Sim 6 YAM sim + HDF5 recorder)
```

## How the mapping works

The mapping is joint-to-joint. Each YAM joint follows one SO-101 joint, relative to a shared
rest pose:

| YAM joint | Moves | SO-101 source |
|---|---|---|
| joint1 | base yaw | `shoulder_pan` |
| joint2 | shoulder pitch | `shoulder_lift` |
| joint3 | elbow | `elbow_flex` |
| joint4 | wrist pitch | `wrist_flex` |
| joint5 | wrist yaw | *none. Held at 0°, because the SO-101 has no wrist-yaw joint* |
| joint6 | wrist roll | `wrist_roll` |
| fingers | gripper | `gripper`, scaled linearly between the calibrated open/closed positions to −4.75 cm … 0 |

`yam_angle = sign × (so101_angle − so101_rest_angle)`. The result is clipped to the YAM joint
limits. Calibration measures the `sign` values, so no hand-tuning is needed.

The two arms have different link lengths, so this feels like puppeteering the joints: the YAM
hand will not land exactly where the SO-101 hand is. Watch the sim viewport while you work.

Settings live in `configs/so101_yam_mapping.yaml`: per-joint `scale`, `ref_deg`, limits, control
rate, start-up blend time and the per-step rate limit.

## Requirements

- The `env_yamlab6` conda env (see `../PORT_ISAACSIM6.md`). The leader side only adds
  `feetech-servo-sdk`, which is already installed.
- Two SO-101 **leader** arms. The servo IDs must be 1–6 in the order `shoulder_pan … gripper`,
  which is LeRobot's standard setup. The leader's servo torque is switched off, so you move it by hand.
- Serial access: your user is already in the `dialout` group.

## 1. Find the ports

Plug the leaders in **one at a time** so you can tell them apart:

```bash
ls -l /dev/serial/by-id/          # stable names, recommended
ls /dev/ttyACM*                   # e.g. /dev/ttyACM0, /dev/ttyACM1
```

Use the `/dev/serial/by-id/...` paths if you can: `ttyACM0/1` can swap between reboots.

## 2. Calibrate (once per leader pair)

```bash
conda activate env_yamlab6 && cd ~/yamlab6
python joylo/scripts/calibrate_so101.py \
    --left_port  /dev/serial/by-id/<left-leader> \
    --right_port /dev/serial/by-id/<right-leader>
```

The script does the following for each arm, left then right:

1. **Rest pose.** Fold the leader like the YAM start pose: upper arm lying back, forearm folded
   forward on top, wrist straight, gripper pointing forward. Then press ENTER.
2. **Direction check.** For each joint you are told which physical direction to move it (for
   example "LIFT the upper arm up from the folded rest pose"). Move it about 30–45° in that
   direction, hold, press ENTER, then go back to rest. The script records the sign.
3. **Gripper.** Fully open it and press ENTER, then fully close it and press ENTER.

The result is written to `joylo/configs/so101_calibration.json`. Re-run the script if you
reassemble a leader or swap which leader is left or right.

## 3. Teleoperate

Terminal 1 starts the follower, the Isaac Sim 6 GUI with the task. Keep `--viz kit`: without it,
Isaac Lab 3 runs headless.

```bash
conda activate env_yamlab6 && cd ~/yamlab6
D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
python joylo/scripts/launch_follower.py --task PutPotOnCooktop-v0 \
    --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 \
    --port 11333 --demos_per_asset 25 \
    --data_output_dir ./recorded_data --data_output_filename so101_demos \
    --enable_gripper_clamp --enable_cameras --viz kit
```

Wait until port 11333 is listening (`ss -ltn | grep 11333`). This takes about 30 s, and the
first launch compiles shaders, which takes longer.

Terminal 2 starts the SO-101 leaders:

```bash
conda activate env_yamlab6 && cd ~/yamlab6
python joylo/scripts/launch_so101.py \
    --left_port  /dev/serial/by-id/<left-leader> \
    --right_port /dev/serial/by-id/<right-leader> \
    --enable_recording
```

When the leader starts, the sim arms ease from their current pose to the leader pose over
1.5 s. After that they follow the leaders at 30 Hz.

| Key (in terminal 2) | Action |
|---|---|
| `SPACE` | start recording, then press again to **save** the demo (the task resets for the next one) |
| `r` | discard the current recording and reset the task |
| `a` | single leader only: switch which YAM arm the leader drives (the other holds) |
| `q` | quit |

`[TASK PROGRESS] pick: … place: …` lines show the sim's stage detection live. Save when
`task_success: True`. Demos are written to `./recorded_data/so101_demos.hdf5`.

To use another task, swap the follower's `--task` and assets, for example
`--task HangMugOnTree-v0 --asset mug=... --asset mug_tree=...`.

### With a single leader

A single SO-101 can drive **one YAM arm at a time**. The other arm holds its pose, and `a`
switches arms. That's enough to try things out, and two-handed tasks can be done step by step
(position one arm, press `a`, bring in the other).

```bash
# calibrate just this leader (merged into so101_calibration.json)
python joylo/scripts/calibrate_so101.py --left_port /dev/serial/by-id/<leader>

# teleop: the leader drives the LEFT YAM arm first
python joylo/scripts/launch_so101.py --left_port /dev/serial/by-id/<leader> --enable_recording
#   add --start_arm right to begin on the right arm
```

When you switch, the newly driven arm eases from where it is onto the leader's pose over 1.5 s,
so put the leader roughly in that arm's pose before pressing `a`.

### Without hardware

```bash
python joylo/scripts/launch_so101.py --mock             # synthetic leader motion into the running sim
python joylo/scripts/launch_so101.py --mock --dry_run   # no sim: just print the mapped 14-dim command
```

## 4. Use the demos

The recorded HDF5 is a standard Isaac Lab 3 dataset (`format_version=1`, XYZW quaternions):

```bash
python scripts/replay_data.py --task PutPotOnCooktop-v0 --dataset_file recorded_data/so101_demos.hdf5 \
    --assets_root_path yamlab_datasets/tasks_data/PutPotOnCooktop/objects --output_root out/replay \
    --enable_gripper_clamp --enable_cameras --headless
python scripts/mimic/annotate_demos.py --task PutPotOnCooktop-Mimic-v0 --input_file recorded_data/so101_demos.hdf5 ...
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open SO-101 leader port` | Wrong port, or another program (e.g. `lerobot-teleoperate`) has the port open |
| `servo id N ... did not answer ping` | Bad cable or power, or the IDs are not 1–6. Check with LeRobot's `lerobot-setup-motors` |
| A YAM joint moves the wrong way | Re-run calibration and move that joint in exactly the direction printed |
| The YAM pose is offset from the leader's | Recalibrate the rest pose more carefully, or adjust that joint's `ref_deg` in `so101_yam_mapping.yaml` |
| Motion is too small or large for comfort | Set the joint's `scale` in `so101_yam_mapping.yaml` (default 1.0) |
| The gripper barely moves | The open/closed readings were too close. Recalibrate, opening and closing fully |
| The HDF5 cannot be opened while the follower runs | The recorder keeps it open. Stop the follower, or read it with `HDF5_USE_FILE_LOCKING=FALSE` |
| No sim window | Add `--viz kit` to the follower. Several windows are titled "Isaac Lab 3.0.0", so look for the tabletop scene |

## Files

| File | Purpose |
|---|---|
| `joylo/agents/so101_agent.py` | `SO101Leader` (Feetech sync-read, torque off) and `SO101ToYamMapper` |
| `scripts/calibrate_so101.py` | Interactive rest-pose, joint-sign and gripper calibration |
| `scripts/launch_so101.py` | Teleop loop: blend-in, rate limiting, keyboard recording controls, `--mock` / `--dry_run` |
| `configs/so101_yam_mapping.yaml` | Joint mapping, YAM limits and directions, loop settings |
| `configs/so101_calibration.json` | Written by calibration (per-arm rest ticks, signs, gripper range) |

## Status

- **Verified without hardware.** Mock leaders drove the live Isaac Sim 6 follower. The sim
  joints tracked the commands, joint5 stayed at 0, and recording produced a valid demo (267 steps,
  full gripper range, `format_version=1`).
- **Not yet tested on real SO-101 hardware.** The calibration and the Feetech reader will get
  their first real run with the arms connected.

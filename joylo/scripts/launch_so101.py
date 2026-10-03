"""Teleoperate the bimanual YAM follower (sim) with two SO-101 leader arms.

Drop-in replacement for launch_joylo.py: it talks to the same follower RPC server started by
launch_follower.py and sends the same 14-dim command
[left joint1..6, left finger, right joint1..6, right finger]. Recording is driven from the
keyboard (the SO-101 has no buttons):

  SPACE  start recording / save the trajectory
  r      discard the current recording and reset the task
  a      (single leader only) switch which YAM arm the leader drives; the other arm holds
  q      quit

Usage (follower first, in another terminal, see PORT_ISAACSIM6.md):
  python joylo/scripts/launch_so101.py --left_port /dev/ttyACM0 --right_port /dev/ttyACM1 \
      --calibration joylo/configs/so101_calibration.json --enable_recording

  # One leader: it drives the LEFT YAM arm first; press `a` to switch to the right arm.
  python joylo/scripts/launch_so101.py --left_port /dev/ttyACM1 --enable_recording

  # No hardware: drive the sim with a synthetic motion to check the pipeline.
  python joylo/scripts/launch_so101.py --mock
"""

import argparse
import math
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from joylo.agents.so101_agent import SO101Leader, SO101ToYamMapper, load_calibration, load_mapping
from joylo.follower_client import BimanualFollowerClient

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class MockLeader:
    """Synthetic SO-101 readings: slow joint sweeps around the rest pose, gripper opening/closing."""

    def __init__(self, calib: dict, phase: float):
        self.ref = np.asarray(calib["ref_ticks"], dtype=np.float64)
        self.g_open, self.g_closed = calib["gripper_open_ticks"], calib["gripper_closed_ticks"]
        self.phase, self.t0 = phase, time.time()

    def read_ticks(self) -> np.ndarray:
        t = time.time() - self.t0 + self.phase
        ticks = self.ref.copy()
        amp = np.array([250, 500, 700, 300, 400]) * (0.5 - 0.5 * math.cos(0.4 * t))  # 0 at start
        ticks[:5] += amp * np.array([math.sin(0.5 * t), 1, 1, math.sin(0.7 * t), math.sin(0.3 * t)])
        g = 0.5 - 0.5 * math.cos(0.8 * t)
        ticks[5] = self.g_open + g * (self.g_closed - self.g_open)
        return np.round(ticks).astype(np.int64) % 4096

    def close(self):
        pass


MOCK_CALIB = {"ref_ticks": [2048] * 6, "signs": {}, "gripper_open_ticks": 2600, "gripper_closed_ticks": 2048}


@contextmanager
def raw_keyboard():
    """Put stdin in cbreak mode so single key presses can be polled without Enter."""
    if not sys.stdin.isatty():
        yield
        return
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def poll_key() -> str | None:
    """Return a pressed key, or None."""
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    parser = argparse.ArgumentParser(description="SO-101 leaders -> bimanual YAM follower teleop")
    parser.add_argument("--left_port", help="Serial port of the leader driving the LEFT YAM arm")
    parser.add_argument("--right_port", help="Serial port of the leader driving the RIGHT YAM arm")
    parser.add_argument("--calibration", default=str(CONFIG_DIR / "so101_calibration.json"))
    parser.add_argument("--mapping", default=str(CONFIG_DIR / "so101_yam_mapping.yaml"))
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11333, help="Follower RPC port")
    parser.add_argument("--enable_recording", action="store_true", help="Enable SPACE start/save recording")
    parser.add_argument("--mock", action="store_true", help="Use synthetic leaders (no hardware)")
    parser.add_argument("--dry_run", action="store_true", help="Print the mapped command instead of sending it")
    parser.add_argument("--start_arm", choices=["left", "right"], default=None,
                        help="Single leader: YAM arm it drives first (default: the side of the given port)")
    args = parser.parse_args()

    cfg = load_mapping(args.mapping)
    sides = ("left_arm", "right_arm")
    slices = {"left_arm": slice(0, 7), "right_arm": slice(7, 14)}
    if args.mock:
        calib = {"left_arm": MOCK_CALIB, "right_arm": MOCK_CALIB}
        ports = {"left_arm": None if args.right_port and not args.left_port else "mock",
                 "right_arm": None if args.left_port and not args.right_port else "mock"}
        leaders = {s: MockLeader(MOCK_CALIB, 2.0 * i) for i, s in enumerate(sides) if ports[s]}
    else:
        ports = {"left_arm": args.left_port, "right_arm": args.right_port}
        if not any(ports.values()):
            sys.exit("[ERROR] give --left_port and/or --right_port (or use --mock)")
        if not Path(args.calibration).exists():
            sys.exit(f"[ERROR] Calibration {args.calibration} not found; run joylo/scripts/calibrate_so101.py first")
        calib = load_calibration(args.calibration)
        for s in sides:
            if ports[s] and s not in calib:
                sys.exit(f"[ERROR] No '{s}' entry in {args.calibration}; calibrate that leader first")
        leaders = {s: SO101Leader(ports[s]) for s in sides if ports[s]}
    mappers = {s: SO101ToYamMapper(cfg["mapping"], calib[s]) for s in leaders}

    # Single-leader mode: one leader drives the `active` YAM arm; the other arm holds its last
    # command. `a` switches arms. The leader keeps its own calibration whichever arm it drives.
    single = len(leaders) == 1
    solo = next(iter(leaders)) if single else None
    active = (args.start_arm + "_arm") if (single and args.start_arm) else solo
    held = np.zeros(14)
    held[6] = held[13] = float(cfg["mapping"]["finger_open_pos"])

    def leader_command() -> np.ndarray:
        if not single:
            return np.concatenate([mappers[s](leaders[s].read_ticks()) for s in sides])
        cmd = held.copy()
        cmd[slices[active]] = mappers[solo](leaders[solo].read_ticks())
        return cmd

    dt = 1.0 / float(cfg.get("control_frequency", 30.0))
    blend_steps = max(1, int(float(cfg.get("startup_blend_s", 1.5)) / dt))
    max_step = float(cfg.get("max_step_rad", 0.35))
    arm_idx = np.r_[0:6, 7:13]  # finger entries (6, 13) are not rate limited

    follower = None
    if not args.dry_run:
        follower = BimanualFollowerClient(host=args.server_host, port=args.port)
        print(f"[INFO] Connected to follower ({follower.num_dofs()} DOFs) at {args.server_host}:{args.port}")

    if follower is not None:
        sim_q = np.asarray(follower.get_joint_pos(), dtype=np.float64)
        held[arm_idx] = sim_q[arm_idx]  # the idle arm starts by holding its current sim pose

    def start_blend():
        start = leader_command()
        if follower is not None:
            # get_joint_pos reports the grippers as a binary 0/1 state, not a finger position,
            # so only the arm joints are blended from the sim pose.
            start[arm_idx] = np.asarray(follower.get_joint_pos(), dtype=np.float64)[arm_idx]
        return start, 0

    blend_from, blend_i = start_blend()
    last_cmd = blend_from.copy()
    recording = False
    prev_progress = None
    last_poll = 0.0
    print("[INFO] Keys: SPACE start/save recording | r discard + reset | a switch arm (single leader) | q quit")
    print(f"[INFO] Leaders: {'MOCK ' if args.mock else ''}" + ", ".join(f"{s}={ports[s]}" for s in leaders)
          + "; YAM joint5 (wrist yaw) held fixed")
    if single:
        print(f"[INFO] Single leader: driving the {active.split('_')[0].upper()} YAM arm -- press `a` to switch arms")

    try:
        with raw_keyboard():
            while True:
                t0 = time.time()
                target = leader_command()
                if blend_i < blend_steps:  # ease in from the sim pose after start / reset
                    a = (blend_i + 1) / blend_steps
                    target = (1 - a) * blend_from + a * target
                    blend_i += 1
                step = np.clip(target[arm_idx] - last_cmd[arm_idx], -max_step, max_step)
                cmd = target.copy()
                cmd[arm_idx] = last_cmd[arm_idx] + step
                last_cmd = cmd

                if follower is None:
                    print("L " + " ".join(f"{math.degrees(v):7.1f}" for v in cmd[:6]) + f" f={cmd[6]:+.4f} | "
                          "R " + " ".join(f"{math.degrees(v):7.1f}" for v in cmd[7:13]) + f" f={cmd[13]:+.4f}",
                          end="\r", flush=True)
                else:
                    follower.command_bimanual_joint_pos(cmd)

                key = poll_key()
                if key == "q":
                    break
                if single and key == "a":
                    held[:] = last_cmd  # freeze both arms where they are
                    active = "right_arm" if active == "left_arm" else "left_arm"
                    blend_from, blend_i = last_cmd.copy(), 0  # ease the newly active arm onto the leader
                    print(f"\n[ARM] Leader now drives the {active.split('_')[0].upper()} YAM arm "
                          "(the other arm holds its pose)")
                if follower is not None and key == " " and args.enable_recording:
                    if not recording and follower.start_recording():
                        recording = True
                        print("\n[REC] Recording started -- SPACE to save, r to discard")
                    elif recording and follower.save_trajectory():
                        recording = False
                        print("\n[REC] Trajectory saved; task reset for the next demo")
                        blend_from, blend_i = start_blend()
                elif follower is not None and key == "r":
                    result, done = follower.reset_task()
                    recording = False
                    print(f"\n[RESET] Task reset (demo {result.get('demo_count')}/{result.get('demos_per_asset')})")
                    blend_from, blend_i = start_blend()
                    if done:
                        print("[INFO] All demos collected")
                        break

                if follower is not None and time.time() - last_poll > 0.25:
                    last_poll = time.time()
                    try:
                        progress = follower.get_task_info()[0].get("intermediate_success", {})
                        if progress != prev_progress:
                            print("\n[TASK PROGRESS] " + "; ".join(f"{k}: {bool(v)}" for k, v in progress.items()))
                            prev_progress = progress
                    except Exception:
                        pass

                time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Shutting down SO-101 teleop")
        if follower is not None and args.enable_recording:
            try:
                follower.close_data_collector()
            except Exception:
                pass
        for leader in leaders.values():
            leader.close()


if __name__ == "__main__":
    main()

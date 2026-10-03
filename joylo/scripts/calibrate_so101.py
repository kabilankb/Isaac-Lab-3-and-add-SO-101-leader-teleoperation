"""Calibrate two SO-101 leader arms for driving the bimanual YAM in simulation.

For each leader (left, then right) this records:
  1. the rest pose: servo positions with the leader folded like the YAM all-zero pose
     (upper arm lying back, forearm folded forward, gripper pointing forward);
  2. the sign of each mapped joint: you move the joint in the stated direction and the script
     measures which way the servo turned;
  3. the gripper open and closed servo positions.

Usage:
  python joylo/scripts/calibrate_so101.py --left_port /dev/ttyACM0 --right_port /dev/ttyACM1 \
      --output joylo/configs/so101_calibration.json

  # One leader only (it can later drive either YAM arm; see launch_so101.py):
  python joylo/scripts/calibrate_so101.py --left_port /dev/ttyACM1

Find the ports with ``ls /dev/serial/by-id/`` (plug the arms in one at a time to tell them apart).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from joylo.agents.so101_agent import SO101_JOINTS, TICKS_PER_REV, SO101Leader, load_mapping

MIN_MOVE_TICKS = 150  # ~13 deg: a deliberate move, well above servo noise


def _avg_ticks(leader: SO101Leader, seconds: float = 0.5) -> np.ndarray:
    """Average readings over a short window (circular mean per servo)."""
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        samples.append(leader.read_ticks())
        time.sleep(0.02)
    ang = np.stack(samples).astype(np.float64) * 2 * np.pi / TICKS_PER_REV
    mean = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
    return np.round(np.mod(mean, 2 * np.pi) * TICKS_PER_REV / (2 * np.pi)).astype(np.int64) % TICKS_PER_REV


def _wrapped_delta(a: int, b: int) -> int:
    """Signed tick difference a - b wrapped to (-2048, 2048]."""
    return (int(a) - int(b) + TICKS_PER_REV // 2) % TICKS_PER_REV - TICKS_PER_REV // 2


def _wait(msg: str):
    input(f"\n>>> {msg}\n    Press ENTER when ready...")


def calibrate_arm(side: str, port: str, mapping: dict) -> dict:
    """Run the interactive calibration for one leader arm."""
    print(f"\n{'=' * 70}\n  {side.upper()} SO-101 leader on {port}\n{'=' * 70}")
    leader = SO101Leader(port)
    try:
        _wait(f"Put the {side} leader in the REST pose: upper arm folded back/down, forearm folded "
              "forward on top of it, wrist straight, gripper pointing forward and closed-ish. "
              "This must match the YAM start pose.")
        ref = _avg_ticks(leader)
        print(f"    rest ticks: {dict(zip(SO101_JOINTS, ref.tolist()))}")

        signs = {}
        for j in mapping["mapping"]["yam_joints"]:
            src = j.get("source")
            if src is None:
                continue
            idx = SO101_JOINTS.index(src)
            while True:
                _wait(f"[{j['name']} <- {src}] Starting from rest, {j['positive']} by ~30-45 deg and HOLD it.")
                moved = _avg_ticks(leader)
                d = _wrapped_delta(moved[idx], ref[idx])
                if abs(d) >= MIN_MOVE_TICKS:
                    signs[j["name"]] = 1.0 if d > 0 else -1.0
                    print(f"    {src} moved {d:+d} ticks -> sign {signs[j['name']]:+.0f}")
                    break
                print(f"    {src} only moved {d:+d} ticks; move it further and try again.")
            _wait("Return the arm to the rest pose.")

        _wait("Fully OPEN the gripper and hold it.")
        g_open = int(_avg_ticks(leader)[SO101_JOINTS.index("gripper")])
        _wait("Fully CLOSE the gripper and hold it.")
        g_closed = int(_avg_ticks(leader)[SO101_JOINTS.index("gripper")])
        if abs(_wrapped_delta(g_closed, g_open)) < MIN_MOVE_TICKS:
            print("[WARNING] Gripper open/closed readings are very close; gripper control will be weak.")
        return {
            "port": port,
            "ref_ticks": ref.tolist(),
            "signs": signs,
            "gripper_open_ticks": g_open,
            "gripper_closed_ticks": g_closed,
        }
    finally:
        leader.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate two SO-101 leaders for YAM teleop")
    parser.add_argument("--left_port", help="Serial port of the leader driving the LEFT YAM arm")
    parser.add_argument("--right_port", help="Serial port of the leader driving the RIGHT YAM arm")
    parser.add_argument("--mapping", default=str(Path(__file__).resolve().parents[1] / "configs" / "so101_yam_mapping.yaml"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "configs" / "so101_calibration.json"))
    args = parser.parse_args()

    if not (args.left_port or args.right_port):
        sys.exit("[ERROR] give --left_port and/or --right_port")
    if args.left_port and args.left_port == args.right_port:
        sys.exit("[ERROR] left and right ports must differ")
    mapping = load_mapping(args.mapping)
    out = Path(args.output)
    # Merge into an existing calibration so the two leaders can be calibrated separately.
    calib = json.loads(out.read_text()) if out.exists() else {}
    for side, port in (("left", args.left_port), ("right", args.right_port)):
        if port:
            calib[f"{side}_arm"] = calibrate_arm(side, port, mapping)
    out.write_text(json.dumps(calib, indent=2))
    print(f"\n[INFO] Calibration saved to {args.output}")


if __name__ == "__main__":
    main()

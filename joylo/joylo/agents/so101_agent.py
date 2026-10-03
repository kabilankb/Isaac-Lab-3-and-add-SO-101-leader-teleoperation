"""SO-101 leader arm (LeRobot, Feetech STS3215 servos) as a YAM teleoperation input.

Each SO-101 leader has six servos on one USB serial bus (IDs 1-6: shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper). The leader is back-driven by hand, so torque stays
off; this module only sync-reads Present_Position.

:class:`SO101Leader` reads raw servo ticks. :class:`SO101ToYamMapper` turns them into one YAM
arm's 7-dim command ``[joint1..joint6, finger]`` with a joint-to-joint mapping: each YAM joint is
``yam_ref + sign * scale * (ticks - ticks_ref)`` in radians, where the ``_ref`` values come from
a calibration taken with both arms in the same (folded rest) pose. YAM joint5 (wrist yaw) has no
SO-101 counterpart and is held at a constant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TICKS_PER_REV = 4096
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_POSITION = 56


def _decode_ticks(raw: int) -> int:
    """Decode an STS3215 position word (bit 15 is a sign bit when a homing offset is set)."""
    return -(raw & 0x7FFF) if raw & 0x8000 else raw


class SO101Leader:
    """Reads the six joint positions (raw ticks) of one SO-101 leader arm."""

    def __init__(self, port: str, motor_ids=(1, 2, 3, 4, 5, 6), baudrate: int = 1_000_000):
        """Open the serial bus and set up a sync read of Present_Position.

        Args:
            port (str): Serial device, e.g. ``/dev/ttyACM0``.
            motor_ids (Sequence[int]): Servo IDs in SO101_JOINTS order.
            baudrate (int): Bus baudrate (LeRobot default for SO-101 is 1 Mbps).

        Raises:
            RuntimeError: If the port cannot be opened or a servo does not answer.
        """
        import scservo_sdk as scs

        self._scs = scs
        self.port = port
        self.motor_ids = list(motor_ids)
        self._port = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)  # STS series: protocol_end 0
        if not self._port.openPort():
            raise RuntimeError(f"Cannot open SO-101 leader port {port}")
        if not self._port.setBaudRate(baudrate):
            raise RuntimeError(f"Cannot set baudrate {baudrate} on {port}")
        for mid in self.motor_ids:
            _, result, _ = self._packet.ping(self._port, mid)
            if result != scs.COMM_SUCCESS:
                raise RuntimeError(f"SO-101 servo id {mid} on {port} did not answer ping "
                                   f"({self._packet.getTxRxResult(result)})")
            # Leader is moved by hand: make sure the servo is not holding position.
            self._packet.write1ByteTxRx(self._port, mid, ADDR_TORQUE_ENABLE, 0)
        self._reader = scs.GroupSyncRead(self._port, self._packet, ADDR_PRESENT_POSITION, 2)
        for mid in self.motor_ids:
            self._reader.addParam(mid)
        self._last = None

    def read_ticks(self) -> np.ndarray:
        """Return the current raw positions, shape (6,), int ticks (0-4095 = one revolution).

        On a transient bus error the previous reading is returned (and the error raised if there
        is none yet).
        """
        result = self._reader.txRxPacket()
        if result != self._scs.COMM_SUCCESS:
            if self._last is not None:
                return self._last
            raise RuntimeError(f"SO-101 sync read failed on {self.port}: "
                               f"{self._packet.getTxRxResult(result)}")
        ticks = np.array([_decode_ticks(self._reader.getData(mid, ADDR_PRESENT_POSITION, 2))
                          for mid in self.motor_ids], dtype=np.int64)
        self._last = ticks
        return ticks

    def close(self):
        """Close the serial port."""
        self._port.closePort()


class SO101ToYamMapper:
    """Joint-to-joint mapping from SO-101 leader ticks to one YAM arm's 7-dim command."""

    def __init__(self, mapping_cfg: dict, calibration: dict):
        """Build the mapper.

        Args:
            mapping_cfg (dict): The ``mapping`` block of ``so101_yam_mapping.yaml``.
            calibration (dict): One arm's calibration (``ref_ticks``, ``gripper_open_ticks``,
                ``gripper_closed_ticks``) as written by ``calibrate_so101.py``.
        """
        self.joints = mapping_cfg["yam_joints"]
        self.limits = np.array([j["limits_deg"] for j in self.joints], dtype=np.float64) * math.pi / 180.0
        self.ref_ticks = np.asarray(calibration["ref_ticks"], dtype=np.int64)
        self.g_open = float(calibration["gripper_open_ticks"])
        self.g_closed = float(calibration["gripper_closed_ticks"])
        self.finger_open = float(mapping_cfg["finger_open_pos"])
        self.finger_closed = float(mapping_cfg["finger_closed_pos"])
        # Measured signs from calibration override the YAML defaults.
        cal_signs = calibration.get("signs", {})
        self.signs = [float(cal_signs.get(j["name"], j.get("sign", 1.0))) for j in self.joints]

    def _delta_rad(self, ticks: np.ndarray, idx: int) -> float:
        """Signed angle (rad) of servo ``idx`` from its calibration reference, wrapped to ±180°."""
        d = (int(ticks[idx]) - int(self.ref_ticks[idx]) + TICKS_PER_REV // 2) % TICKS_PER_REV - TICKS_PER_REV // 2
        return d * 2.0 * math.pi / TICKS_PER_REV

    def __call__(self, ticks: np.ndarray) -> np.ndarray:
        """Map raw leader ticks to ``[joint1..joint6, finger]`` (rad, rad, ..., m), clipped to limits."""
        q = np.zeros(7, dtype=np.float64)
        for i, j in enumerate(self.joints):
            src = j.get("source")
            if src is None:
                q[i] = math.radians(j.get("fixed_deg", 0.0))
            else:
                src_idx = SO101_JOINTS.index(src)
                q[i] = math.radians(j.get("ref_deg", 0.0)) + self.signs[i] * j.get("scale", 1.0) * self._delta_rad(ticks, src_idx)
            q[i] = min(max(q[i], self.limits[i, 0]), self.limits[i, 1])
        # Gripper: linear between the calibrated open/closed servo positions.
        g_idx = SO101_JOINTS.index("gripper")
        span = self.g_closed - self.g_open
        frac = 0.0 if abs(span) < 1e-6 else (float(ticks[g_idx]) - self.g_open) / span
        frac = min(max(frac, 0.0), 1.0)
        q[6] = self.finger_open + frac * (self.finger_closed - self.finger_open)
        return q


def load_mapping(path: str | Path) -> dict:
    """Load ``so101_yam_mapping.yaml``."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_calibration(path: str | Path) -> dict:
    """Load the two-arm calibration JSON written by ``calibrate_so101.py``."""
    with open(path) as f:
        return json.load(f)

"""Scan every connected SO-101 / Feetech bus board and report which servos answer.

Usage:
  python joylo/scripts/scan_so101.py                 # all /dev/serial/by-id boards, 1 Mbps
  python joylo/scripts/scan_so101.py --port /dev/ttyACM0 --all_bauds

A healthy SO-101 leader answers IDs 1-6 cleanly. "garbled" replies on every ID mean the board's
servo bus is not wired or jumpered correctly; no replies at all mean the servos are unpowered or
not connected.
"""

import argparse
import glob

import scservo_sdk as scs

JOINTS = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
BAUDS = [1_000_000, 500_000, 250_000, 115_200, 57_600, 38_400]


def scan(dev: str, bauds: list[int], max_id: int):
    port, ph = scs.PortHandler(dev), scs.PacketHandler(0)
    if not port.openPort():
        print(f"{dev}: cannot open (in use by another program?)")
        return
    print(f"\n{dev}")
    for baud in bauds:
        port.setBaudRate(baud)
        ok, garbled = [], []
        for i in range(max_id + 1):
            r = ph.ping(port, i)[1]
            if r == scs.COMM_SUCCESS:
                ok.append(i)
            elif r == scs.COMM_RX_CORRUPT:
                garbled.append(i)
        status = ("OK: all 6 SO-101 servos" if ok == [1, 2, 3, 4, 5, 6] else
                  "GARBLED: check the board jumper and servo cable" if garbled and not ok else
                  "NO REPLY: check servo power and cable" if not ok else "PARTIAL")
        print(f"  {baud:>9} baud: {status}  (clean ids {ok}, garbled {len(garbled)})")
        for i in ok:
            pos, r, _ = ph.read2ByteTxRx(port, i, 56)
            print(f"      id {i} {JOINTS.get(i, '?'):13s} position {pos if r == scs.COMM_SUCCESS else 'read error'}")
    port.closePort()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", help="One serial device (default: every /dev/serial/by-id board)")
    p.add_argument("--all_bauds", action="store_true", help="Also try 500k..38.4k baud")
    p.add_argument("--max_id", type=int, default=20)
    a = p.parse_args()
    devs = [a.port] if a.port else sorted(glob.glob("/dev/serial/by-id/*"))
    if not devs:
        print("No serial boards found. Is the leader's USB plugged in?")
    for dev in devs:
        scan(dev, BAUDS if a.all_bauds else BAUDS[:1], a.max_id)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate one LeRobot v2.0 dataset with MimicGen across several object pairs.

Each annotated source file holds demos for one object pair (e.g. pot_003 +
cooktop_003). This script runs ``scripts/mimic/generate_dataset.py`` once per
source file ("part"), each with its own seed and domain randomization, then
merges the parts into one dataset with ``merge_lerobot_v2.py``.

Parts are kept under ``--work-dir`` and a part that already holds its episodes
is skipped, so an interrupted run resumes where it stopped.

Usage (from the repo root, in env_yamlab6):
    python scripts/pipeline/generate.py --episodes 20 --parts 10 \\
        --output datasets/yam_put_pot
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_lerobot_v2 import merge  # noqa: E402

TASKS = {
    "PutPotOnCooktop": {
        "gym_id": "PutPotOnCooktop-Mimic-v0",
        "annotated": "annotated_putpot_{:03d}.hdf5",
        "part": "putpot_{:03d}",
        "description": "Put the pot on the cooktop",
    },
    "HangMugOnTree": {
        "gym_id": "HangMugOnTree-Mimic-v0",
        "annotated": "annotated_hangmug_{:03d}.hdf5",
        "part": "hangmug_{:03d}",
        "description": "Hang the mug on the mug tree",
    },
}


def _episodes_in(root: Path) -> int:
    info = root / "meta" / "info.json"
    return json.loads(info.read_text())["total_episodes"] if info.is_file() else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="PutPotOnCooktop")
    parser.add_argument("--episodes", type=int, required=True, help="Successful episodes in the final dataset")
    parser.add_argument("--parts", type=int, default=10, help="Object pairs (annotated source files) to spread them over")
    parser.add_argument("--first-source", type=int, default=0, help="Index of the first annotated source file")
    parser.add_argument("--output", type=Path, required=True, help="Merged dataset root (must not exist)")
    parser.add_argument("--work-dir", type=Path, help="Where parts are kept (default: <output>_parts)")
    parser.add_argument("--data-root", type=Path, default=REPO / "yamlab_datasets")
    parser.add_argument("--seed", type=int, default=0, help="Part k uses seed + k")
    parser.add_argument("--no-dr", action="store_true", help="Disable domain randomization (HDRI lighting + materials)")
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--timeout-per-part", type=float, default=3600.0, help="Seconds before a part is abandoned")
    args = parser.parse_args()

    task = TASKS[args.task]
    task_data = args.data_root / "tasks_data" / args.task
    output = args.output.resolve()
    work_dir = (args.work_dir or output.with_name(output.name + "_parts")).resolve()
    if output.exists():
        sys.exit(f"{output} already exists; choose a new --output")
    if args.parts < 1 or args.episodes < args.parts:
        sys.exit("--parts must be between 1 and --episodes")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Spread the episodes as evenly as possible over the parts.
    base, extra = divmod(args.episodes, args.parts)
    plan = [(args.first_source + k, base + (1 if k < extra else 0)) for k in range(args.parts)]
    print(f"[generate] {args.task}: {args.episodes} episodes over {args.parts} object pairs "
          f"{[source for source, _ in plan]}, DR {'off' if args.no_dr else 'on'}")

    parts: list[Path] = []
    for k, (source, wanted) in enumerate(plan):
        annotated = task_data / "annotated" / task["annotated"].format(source)
        part = work_dir / task["part"].format(source)
        if not annotated.is_file():
            sys.exit(f"missing source demos: {annotated}")
        if _episodes_in(part) == wanted:
            print(f"[generate] part {k + 1}/{args.parts} {part.name}: already has {wanted} episodes, skipping")
            parts.append(part)
            continue
        shutil.rmtree(part, ignore_errors=True)
        command = [
            sys.executable, str(REPO / "scripts" / "mimic" / "generate_dataset.py"),
            "--task", task["gym_id"],
            "--input_file", str(annotated),
            "--output_root", str(part),
            "--generation_num_trials", str(wanted),
            "--num_envs", "1",
            "--assets_root_path", str(task_data / "objects"),
            "--observation_modalities", "rgb,proprioception",
            "--camera_width", str(args.camera_width), "--camera_height", str(args.camera_height),
            "--image_downsample_factor", "1",
            "--task_description", task["description"],
            "--enable_gripper_clamp",
            "--seed", str(args.seed + k),
            "--enable_cameras", "--headless",
        ]
        if not args.no_dr:
            command += [
                "--enable_domain_randomization",
                "--hdris_path", str(args.data_root / "HDRIs" / "indoor" / "train"),
                "--materials_path", str(args.data_root / "materials" / "train"),
            ]
        log_path = work_dir / f"{part.name}.log"
        print(f"[generate] part {k + 1}/{args.parts} {part.name}: {wanted} episodes, seed {args.seed + k} "
              f"(log {log_path})", flush=True)
        started = time.monotonic()
        with log_path.open("w") as log:
            try:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout_per_part)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        got = _episodes_in(part)
        print(f"[generate] part {k + 1}/{args.parts} {part.name}: exit {code}, {got}/{wanted} episodes, "
              f"{time.monotonic() - started:.0f}s", flush=True)
        if got != wanted:
            sys.exit(f"part {part.name} produced {got}/{wanted} episodes; see {log_path}. "
                     "Rerun the same command to retry it (finished parts are kept).")
        parts.append(part)

    info = merge(parts, output, dataset_name=output.name)
    print(f"[generate] merged {len(parts)} parts -> {output}: {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames")
    return 0 if info["total_episodes"] == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())

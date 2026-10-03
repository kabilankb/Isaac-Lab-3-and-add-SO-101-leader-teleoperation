#!/usr/bin/env python3
"""Unattended MimicGen data generation for YAMLab: annotated demos -> LeRobot datasets.

Runs ``scripts/mimic/generate_dataset.py`` once per annotated HDF5 file (each file is one
object-asset pair), writing one LeRobot v2.0 dataset per file under
``<output-dir>/<task>/lerobot_mimic_<name>``, the layout of the published
``tasks_data/<task>/lerobot_mimic`` sets. Afterwards each dataset is read back and a
``manifest.json`` is written with the episode counts, so the result does not depend on
how Isaac Sim exits.

The generator deletes an existing dataset at its output path, so a finished file is
skipped on the next run (``--skip-complete``, on by default): rerunning the same command
after an interruption continues where it stopped.

Standard library only; Isaac Sim is launched as a subprocess.

    python scripts/yamlab_generate.py --task PutPotOnCooktop --files 0-3 -n 20
    python scripts/yamlab_generate.py --task HangMugOnTree --files all -n 10 --dr
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "mimic" / "generate_dataset.py"

# Task -> Mimic env id, annotated-file prefix, task text (as in the published datasets).
TASKS = {
    "PutPotOnCooktop": ("PutPotOnCooktop-Mimic-v0", "annotated_putpot_", "Put the pot on the cooktop"),
    "HangMugOnTree": ("HangMugOnTree-Mimic-v0", "annotated_hangmug_", "Hang the mug on the mug tree"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yamlab-generate",
        description="Generate YAMLab MimicGen demonstrations as LeRobot datasets, unattended.",
        epilog="Arguments after -- go to generate_dataset.py unchanged, e.g. -- --discard_first_n_frames 8",
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="PutPotOnCooktop")
    parser.add_argument(
        "--files",
        default="0",
        help="annotated files to use: 'all', indices and ranges like '0-3,7' (default 0)",
    )
    parser.add_argument("-n", "--trials", type=int, default=10, help="successful demos per file (default 10)")
    parser.add_argument("--num-envs", type=int, default=4, help="parallel envs per run (default 4)")
    parser.add_argument("--dr", action="store_true", help="domain randomization: train HDRIs + materials")
    parser.add_argument("--camera", default="320x240", help="camera WIDTHxHEIGHT (default 320x240)")
    parser.add_argument("--modalities", default="rgb,proprioception", help="observation modalities")
    parser.add_argument("--task-description", help="task text stored in the dataset (default: the published one)")
    parser.add_argument("--seed", type=int, help="generation seed (each file gets seed + its index)")
    parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window (default: headless)")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("YAMLAB_DATA", str(REPO_ROOT / "yamlab_datasets")),
        help="yamlab_datasets folder (default: $YAMLAB_DATA or <repo>/yamlab_datasets)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("YAMLAB_OUTPUT", str(REPO_ROOT / "generated")),
        help="where datasets go, one per file under <output-dir>/<task>/ (default: $YAMLAB_OUTPUT or <repo>/generated)",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("YAMLAB_PYTHON"),
        help="Python that runs Isaac Lab (default: $YAMLAB_PYTHON, /isaac-sim/python.sh, or this interpreter)",
    )
    parser.add_argument(
        "--no-skip-complete",
        dest="skip_complete",
        action="store_false",
        help="regenerate files whose dataset already has --trials episodes",
    )
    parser.add_argument("--print-commands", action="store_true", help="print the commands and exit")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    args.extra = extra
    if args.trials < 1 or args.num_envs < 1:
        parser.error("--trials and --num-envs must be at least 1")
    try:
        args.width, args.height = (int(v) for v in args.camera.lower().split("x"))
    except ValueError:
        parser.error(f"--camera must look like 320x240, got {args.camera!r}")
    return args


def select_files(spec: str, available: list[Path]) -> list[Path]:
    """'all' or comma-separated indices/ranges into the sorted annotated files."""
    if spec.strip().lower() == "all":
        return list(available)
    chosen: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        start, stop = int(low), int(high or low)
        if start > stop:
            raise SystemExit(f"[generate] bad range {part!r} in --files")
        chosen.extend(range(start, stop + 1))
    out_of_range = [i for i in chosen if not 0 <= i < len(available)]
    if out_of_range:
        raise SystemExit(f"[generate] --files {out_of_range} out of range: {len(available)} annotated file(s)")
    return [available[i] for i in dict.fromkeys(chosen)]


def isaac_python(explicit: str | None) -> list[str]:
    if explicit:
        return shlex.split(explicit)
    if Path("/isaac-sim/python.sh").is_file():
        return ["/isaac-sim/python.sh"]
    return [sys.executable]


def dataset_name(annotated: Path) -> str:
    """annotated_putpot_000.hdf5 -> lerobot_mimic_putpot_000, as published."""
    return "lerobot_mimic_" + annotated.stem.removeprefix("annotated_")


def count_episodes(root: Path) -> int | None:
    try:
        return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])
    except (OSError, ValueError, KeyError):
        return None


def build_command(args: argparse.Namespace, python: list[str], annotated: Path, output: Path, index: int) -> list[str]:
    task_id, _, description = TASKS[args.task]
    data_root = Path(args.data_root)
    command = python + [
        str(GENERATE_SCRIPT),
        "--task", task_id,
        "--input_file", str(annotated),
        "--output_root", str(output),
        "--generation_num_trials", str(args.trials),
        "--num_envs", str(args.num_envs),
        "--assets_root_path", str(data_root / "tasks_data" / args.task / "objects"),
        "--observation_modalities", args.modalities,
        "--camera_width", str(args.width), "--camera_height", str(args.height),
        "--image_downsample_factor", "1",
        "--task_description", args.task_description or description,
        "--enable_gripper_clamp",
        "--enable_cameras",
    ]
    command += ["--viz", "kit"] if args.gui else ["--headless"]
    if args.gui and os.environ.get("YAMLAB_KIT_ARGS"):  # e.g. window size on a streamed display
        command += ["--kit_args", os.environ["YAMLAB_KIT_ARGS"]]
    if args.dr:
        command += [
            "--enable_domain_randomization",
            "--hdris_path", str(data_root / "HDRIs" / "indoor" / "train"),
            "--materials_path", str(data_root / "materials"),
        ]
    if args.seed is not None:
        command += ["--seed", str(args.seed + index)]
    return command + list(args.extra)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, prefix, _ = TASKS[args.task]
    annotated_dir = Path(args.data_root) / "tasks_data" / args.task / "annotated"
    available = sorted(annotated_dir.glob(f"{prefix}*.hdf5"))
    if not available:
        raise SystemExit(f"[generate] no {prefix}*.hdf5 in {annotated_dir}; download the task's annotated part")
    files = select_files(args.files, available)
    python = isaac_python(args.python)
    out_dir = Path(args.output_dir).expanduser().resolve() / args.task
    plan = [(available.index(f), f, out_dir / dataset_name(f)) for f in files]

    if args.print_commands:
        for index, annotated, output in plan:
            print(shlex.join(build_command(args, python, annotated, output, index)))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy 2 OpenBLAS threads crash Kit's startup fork
    print(f"[generate] {args.task}: {len(plan)} file(s) x {args.trials} demo(s) -> {out_dir}", flush=True)

    results = []
    interrupted = False
    for number, (index, annotated, output) in enumerate(plan, 1):
        existing = count_episodes(output)
        if args.skip_complete and existing is not None and existing >= args.trials:
            print(f"[generate] [{number}/{len(plan)}] {output.name}: already has {existing} episode(s), skipped")
            results.append({"file": annotated.name, "dataset": str(output), "episodes": existing,
                            "exit_code": None, "minutes": 0.0, "skipped": True})
            continue
        command = build_command(args, python, annotated, output, index)
        print(f"[generate] [{number}/{len(plan)}] {annotated.name} -> {output.name}", flush=True)
        print(f"[generate] {shlex.join(command)}", flush=True)
        started = time.monotonic()
        try:
            return_code = subprocess.call(command, cwd=REPO_ROOT, env=environment)
        except KeyboardInterrupt:
            return_code, interrupted = 130, True
        episodes = count_episodes(output) or 0
        minutes = (time.monotonic() - started) / 60.0
        print(f"[generate] {output.name}: {episodes}/{args.trials} episode(s) in {minutes:.1f} min (exit {return_code})",
              flush=True)
        results.append({"file": annotated.name, "dataset": str(output), "episodes": episodes,
                        "exit_code": return_code, "minutes": round(minutes, 2), "skipped": False})
        if interrupted:
            break

    complete = sum(1 for r in results if r["episodes"] >= args.trials)
    manifest = {
        "task": args.task,
        "trials_per_file": args.trials,
        "domain_randomization": args.dr,
        "seed": args.seed,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete_files": complete,
        "total_episodes": sum(r["episodes"] for r in results),
        "files": results,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[generate] {complete}/{len(plan)} file(s) complete, {manifest['total_episodes']} episode(s); "
          f"manifest: {out_dir / 'manifest.json'}")
    if interrupted:
        return 130
    return 0 if complete == len(plan) else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Merge LeRobot v2.0 datasets written by YAMLab into one dataset.

The YAMLab LeRobot recorder deletes an existing dataset at ``--output_root``
instead of appending, so multi-run generation writes one dataset per run and
merges them afterwards. Episodes are renumbered in input order; parquet
``episode_index`` / ``index`` columns, video file names, ``episodes.jsonl``,
``info.json`` totals, and the statistics in ``stats.json`` / ``metadata.json``
(recomputed from all frames with the fork's own ``compute_stats_from_parquet``)
are all rewritten. YAMLab's ``embodiment.json`` and ``modality.json`` are
copied from the first input.

Usage:
    python scripts/pipeline/merge_lerobot_v2.py --output merged/ part_a/ part_b/ ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from lerobot.common.datasets.compute_stats import compute_stats_from_parquet
from lerobot.common.datasets.utils import serialize_dict

# info.json keys that must match across inputs for a merge to be meaningful.
MUST_MATCH = ("codebase_version", "robot_type", "fps", "features", "data_path", "video_path")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def merge(inputs: list[Path], output: Path, dataset_name: str | None = None) -> dict:
    infos = [json.loads((root / "meta" / "info.json").read_text()) for root in inputs]
    for root, info in zip(inputs[1:], infos[1:]):
        for key in MUST_MATCH:
            if info.get(key) != infos[0].get(key):
                raise ValueError(f"{root}: info.json '{key}' differs from {inputs[0]}")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output")

    info = dict(infos[0])
    chunks_size = int(info["chunks_size"])
    video_keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
    (output / "meta").mkdir(parents=True)

    tasks: dict[str, int] = {}
    episodes: list[dict] = []
    global_index = 0
    for root in inputs:
        part_tasks = {row["task_index"]: row["task"] for row in _read_jsonl(root / "meta" / "tasks.jsonl")}
        for episode in _read_jsonl(root / "meta" / "episodes.jsonl"):
            old = int(episode["episode_index"])
            new = len(episodes)
            old_chunk, new_chunk = old // chunks_size, new // chunks_size

            frame = pd.read_parquet(root / info["data_path"].format(episode_chunk=old_chunk, episode_index=old))
            frame["episode_index"] = new
            frame["index"] = range(global_index, global_index + len(frame))
            if "task_index" in frame.columns:
                frame["task_index"] = [
                    tasks.setdefault(part_tasks[int(t)], len(tasks)) for t in frame["task_index"]
                ]
            data_path = output / info["data_path"].format(episode_chunk=new_chunk, episode_index=new)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(data_path)
            global_index += len(frame)

            for key in video_keys:
                source = root / info["video_path"].format(episode_chunk=old_chunk, video_key=key, episode_index=old)
                target = output / info["video_path"].format(episode_chunk=new_chunk, video_key=key, episode_index=new)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

            for task in episode.get("tasks", []):
                tasks.setdefault(task, len(tasks))
            if int(episode["length"]) != len(frame):
                raise ValueError(f"{root} episode {old}: episodes.jsonl length != parquet rows")
            episodes.append({**episode, "episode_index": new})

    total_episodes = len(episodes)
    info.update(
        total_episodes=total_episodes,
        total_frames=global_index,
        total_tasks=len(tasks),
        total_videos=total_episodes * len(video_keys),
        total_chunks=(total_episodes - 1) // chunks_size + 1 if total_episodes else 0,
        splits={"train": f"0:{total_episodes}"},
    )
    meta = output / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(meta / "episodes.jsonl", episodes)
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])

    stats = compute_stats_from_parquet(output)
    (meta / "stats.json").write_text(json.dumps(serialize_dict(stats), indent=4))

    for name in ("embodiment.json", "modality.json"):
        if (inputs[0] / "meta" / name).exists():
            shutil.copy2(inputs[0] / "meta" / name, meta / name)
    metadata_path = inputs[0] / "meta" / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["dataset_name"] = dataset_name or output.name
        metadata["dataset_statistics"] = {
            key: {stat: values.tolist() for stat, values in feature.items() if stat in ("mean", "std", "min", "max")}
            for key, feature in stats.items()
        }
        (meta / "metadata.json").write_text(json.dumps(metadata, indent=4))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="LeRobot v2.0 dataset roots, merged in this order")
    parser.add_argument("--output", required=True, type=Path, help="New dataset root (must not exist)")
    parser.add_argument("--dataset-name", help="metadata.json dataset_name (default: output folder name)")
    args = parser.parse_args()
    info = merge([p.resolve() for p in args.inputs], args.output.resolve(), args.dataset_name)
    print(
        f"[merge] {len(args.inputs)} datasets -> {args.output}: "
        f"{info['total_episodes']} episodes, {info['total_frames']} frames, {info['total_videos']} videos"
    )


if __name__ == "__main__":
    main()

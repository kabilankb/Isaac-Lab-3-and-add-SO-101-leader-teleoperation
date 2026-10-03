#!/usr/bin/env python
"""Check a generated LeRobot v2.0 dataset, then publish it to the Hugging Face Hub.

Refuses to upload unless the dataset loads, has the expected episode count,
and every episode's parquet rows match ``episodes.jsonl`` and has all its
camera videos. Log in first with ``huggingface-cli login``.

Usage (in env_yamlab6):
    python scripts/pipeline/push_to_hub.py datasets/yam_put_pot \\
        --repo-id kabilanKB/yam_put_pot --expect-episodes 20 [--private]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def check(root: Path, expect_episodes: int | None) -> list[str]:
    problems: list[str] = []
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    video_keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
    if expect_episodes is not None and len(episodes) != expect_episodes:
        problems.append(f"{len(episodes)} episodes, expected {expect_episodes}")
    if info["total_episodes"] != len(episodes):
        problems.append("info.json total_episodes != episodes.jsonl")
    for episode in episodes:
        index = episode["episode_index"]
        chunk = index // info["chunks_size"]
        rows = len(pd.read_parquet(root / info["data_path"].format(episode_chunk=chunk, episode_index=index)))
        if rows != episode["length"]:
            problems.append(f"episode {index}: {rows} parquet rows, episodes.jsonl says {episode['length']}")
        for key in video_keys:
            if not (root / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=index)).is_file():
                problems.append(f"episode {index}: missing video {key}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Dataset root")
    parser.add_argument("--repo-id", required=True, help="e.g. kabilanKB/yam_put_pot")
    parser.add_argument("--expect-episodes", type=int, help="Refuse to push unless the dataset has exactly this many")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--check-only", action="store_true", help="Run the checks without uploading")
    args = parser.parse_args()

    root = args.root.resolve()
    problems = check(root, args.expect_episodes)
    dataset = LeRobotDataset(args.repo_id, root=root, local_files_only=True)
    print(f"[push] {root}: {dataset.num_episodes} episodes, {dataset.num_frames} frames, "
          f"cameras {dataset.meta.video_keys}")
    if problems:
        print("[push] check failed; not uploading:\n  " + "\n  ".join(problems))
        return 1
    if args.check_only:
        print("[push] checks passed (--check-only, nothing uploaded)")
        return 0
    dataset.push_to_hub(
        private=args.private,
        tags=["LeRobot", "yamlab", "bimanual", "yam", "isaac-sim", "mimicgen", "simulation"],
    )
    print(f"[push] published https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

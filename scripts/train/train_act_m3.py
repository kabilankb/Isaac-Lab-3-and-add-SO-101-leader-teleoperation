#!/usr/bin/env python
r"""Train an ACT policy on a YAMLab LeRobot v2.0 dataset with M3 modality masking.

M3 (arXiv 2608.22419) hides both wrist cameras together and a random subset of the
action queries during training, and always keeps the top camera visible. See
``yamlab/training/m3_act.py``. The saved checkpoints are ordinary ACT checkpoints.

This wraps the LeRobot fork's ``lerobot/scripts/train.py``: the flags below are
handled here, and every other argument is passed straight through to it.

Query masking needs at least two decoder layers; the fork's ACT default is one.

Usage:
    python scripts/train/train_act_m3.py --dataset-root datasets/yam_put_pot \
        --output_dir=outputs/train/act_m3 --policy.n_decoder_layers=4 \
        --steps=20000 --batch_size=8

    # same run without masking, as the baseline to compare against
    python scripts/train/train_act_m3.py --dataset-root datasets/yam_put_pot --no-m3 \
        --output_dir=outputs/train/act_baseline --policy.n_decoder_layers=4 \
        --steps=20000 --batch_size=8

Loading a checkpoint (the fork's config.json has no "type" key, so pass the config):
    with draccus.config_type("json"):
        config = draccus.parse(ACTConfig, f"{ckpt}/config.json", args=[])
    policy = ACTPolicy.from_pretrained(ckpt, config=config)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot v2.0 dataset folder")
    parser.add_argument("--ego-camera", default="observation.images.top_rgb", help="camera that is never masked")
    parser.add_argument("--wrist-mask-prob", type=float, default=0.3, help="chance of hiding both wrist cameras")
    parser.add_argument("--query-mask-prob", type=float, default=0.1, help="chance of hiding each action query")
    parser.add_argument("--no-query-rescale", action="store_true", help="do not rescale the visible queries")
    parser.add_argument("--no-m3", action="store_true", help="train plain ACT (baseline)")
    parser.add_argument(
        "--preload-frames",
        action="store_true",
        help="decode every video into RAM once (about 0.7 MB per frame per 320x240 camera) instead of "
        "decoding per sample; removes the video-decoding bottleneck for datasets that fit in memory",
    )
    args, lerobot_args = parser.parse_known_args()

    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        parser.error(f"{dataset_root} is not a LeRobot dataset (no meta/info.json)")

    # The fork has no dataset root option: it loads LEROBOT_HOME / repo_id. LEROBOT_HOME is read
    # when lerobot is imported, so it must be set before the imports below.
    os.environ["LEROBOT_HOME"] = str(dataset_root.parent)
    sys.argv = [
        sys.argv[0],
        f"--dataset.repo_id={dataset_root.name}",
        "--dataset.local_files_only=true",
        "--policy.type=act",
        *lerobot_args,
    ]

    import numpy as np
    import torch

    from lerobot.common.datasets import factory as dataset_factory
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies import factory
    from lerobot.common.utils.utils import init_logging
    from lerobot.scripts import train as lerobot_train

    from yamlab.training.m3_act import M3ACTPolicy, M3Config

    def make_dataset(cfg):
        # YAMLab datasets have no image statistics in stats.json, which the fork's make_dataset
        # needs in order to overwrite them. Load without that step and set ImageNet statistics here.
        cfg.dataset.use_imagenet_stats = False
        dataset = dataset_factory.make_dataset(cfg)
        for key in dataset.meta.camera_keys:
            dataset.meta.stats[key] = {
                name: torch.tensor(value, dtype=torch.float32)
                for name, value in dataset_factory.IMAGENET_STATS.items()
            }
        if args.preload_frames:
            preload_frames(dataset)
        return dataset

    lerobot_train.make_dataset = make_dataset

    def preload_frames(dataset):
        """Decode all videos of `dataset` into memory and serve frames from there."""
        import av

        episodes = dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes)
        frames = {}  # (episode index, video key) -> (T, C, H, W) uint8
        for ep_idx in episodes:
            for key in dataset.meta.video_keys:
                with av.open(str(dataset.root / dataset.meta.get_video_file_path(ep_idx, key))) as container:
                    decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
                length = dataset.meta.episodes[ep_idx]["length"]
                if len(decoded) != length:
                    raise RuntimeError(f"Episode {ep_idx} {key}: {len(decoded)} video frames, expected {length}.")
                frames[ep_idx, key] = torch.from_numpy(np.stack(decoded)).permute(0, 3, 1, 2).contiguous()
        n_bytes = sum(video.numel() for video in frames.values())
        logging.info(f"Preloaded {len(frames)} videos into memory ({n_bytes / 2**30:.1f} GiB).")
        dataset._preloaded_frames = frames

    def _query_videos(self, query_timestamps, ep_idx):
        frames = getattr(self, "_preloaded_frames", None)
        if frames is None:
            return _query_videos_from_files(self, query_timestamps, ep_idx)
        item = {}
        for key, query_ts in query_timestamps.items():
            video = frames[ep_idx, key]
            indices = [min(max(round(ts * self.fps), 0), len(video) - 1) for ts in query_ts]
            item[key] = (video[indices].float() / 255).squeeze(0)
        return item

    _query_videos_from_files = LeRobotDataset._query_videos
    LeRobotDataset._query_videos = _query_videos

    # `datasets` >= 4 returns a lazy Column for `dataset[key]`, which the fork passes to torch.stack.
    def _query_hf_dataset(self, query_indices):
        return {
            key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _get_query_timestamps(self, current_ts, query_indices=None):
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = list(self.hf_dataset.select(query_indices[key])["timestamp"])
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    LeRobotDataset._query_hf_dataset = _query_hf_dataset
    LeRobotDataset._get_query_timestamps = _get_query_timestamps

    init_logging()
    if args.no_m3:
        logging.info("M3 masking is off: training plain ACT.")
    else:
        M3ACTPolicy.m3 = M3Config(
            ego_camera=args.ego_camera,
            wrist_mask_prob=args.wrist_mask_prob,
            query_mask_prob=args.query_mask_prob,
            query_rescale=not args.no_query_rescale,
        )
        logging.info(f"M3 masking is on: {M3ACTPolicy.m3}")
        get_policy_class = factory.get_policy_class
        factory.get_policy_class = lambda name: M3ACTPolicy if name == "act" else get_policy_class(name)

    lerobot_train.train()


if __name__ == "__main__":
    main()

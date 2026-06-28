"""Download YAMLab datasets (assets, demos, generated LeRobot data) from the Hub.

Everything lives in one Hugging Face dataset, ``yamlab/yamlab_datasets``, laid out as::

    HDRIs/indoor/{train,eval}/              # DR lighting maps (shared across tasks)
    materials/{train,eval}/<category>/      # DR materials (shared across tasks)
    tasks_data/<Task>/
        objects/                             # USD object instances
        teleop/                              # raw teleoperation demos (HDF5)
        annotated/                           # MimicGen-annotated demos (HDF5)
        lerobot_mimic/                       # generated LeRobot v2.0 dataset
        lerobot_mimic_domain_randomization/  # generated LeRobot v2.0 dataset (DR variant)

This resolves the repo paths for a selection and fetches them with
``huggingface_hub.snapshot_download`` (folder-aware, resumable, dedup-cached).

Examples:
    python scripts/download_data.py --all
    python scripts/download_data.py --domain_randomization
    python scripts/download_data.py --task PutPotOnCooktop
    python scripts/download_data.py --task PutPotOnCooktop --parts lerobot,objects
    python scripts/download_data.py --task HangMugOnTree --parts teleop --local-dir ./data
"""

import argparse

REPO = "yamlab/yamlab_datasets"

# Logical part name -> subfolder under tasks_data/<task>/
PARTS = {
    "objects":    "objects",
    "teleop":     "teleop",
    "annotated":  "annotated",
    "lerobot":    "lerobot_mimic",
    "lerobot_dr": "lerobot_mimic_domain_randomization",
}
# Parts each task actually publishes.
TASKS = {
    "PutPotOnCooktop": ["objects", "teleop", "annotated", "lerobot", "lerobot_dr"],
    "HangMugOnTree":   ["objects", "teleop", "annotated", "lerobot"],
}
# Shared (cross-task) assets: name -> top-level folder.
SHARED = {"hdris": "HDRIs", "materials": "materials"}


def build_patterns(tasks, parts, shared):
    """Build ``snapshot_download`` allow-patterns for the requested selection."""
    pats = []
    for t in tasks:
        avail = TASKS[t]
        for p in (parts or avail):
            if p not in PARTS:
                raise SystemExit(f"unknown part '{p}'; choices: {list(PARTS)}")
            if p not in avail:
                print(f"[skip] task '{t}' has no part '{p}'")
                continue
            pats.append(f"tasks_data/{t}/{PARTS[p]}/**")
    for s in sorted(shared):
        pats.append(f"{SHARED[s]}/**")
    return pats


def main():
    ap = argparse.ArgumentParser(description="Download YAMLab datasets from the Hugging Face Hub.")
    ap.add_argument("--all", action="store_true", help="Download the entire dataset.")
    ap.add_argument("--task", action="append", default=[], choices=list(TASKS),
                    help="Task to download (repeatable). Defaults to all of that task's parts.")
    ap.add_argument("--parts", default=None,
                    help=f"Comma-separated subset of {list(PARTS)} (applies to --task selections).")
    ap.add_argument("--domain_randomization", action="store_true",
                    help="Download the domain-randomization assets (HDRIs + materials).")
    ap.add_argument("--hdris", action="store_true", help="Download the HDRI lighting maps only.")
    ap.add_argument("--materials", action="store_true", help="Download the DR materials only.")
    ap.add_argument("--repo", default=REPO, help=f"Dataset repo id (default: {REPO}).")
    ap.add_argument("--local-dir", default="./yamlab_datasets", help="Local download directory.")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    shared = set()
    if args.domain_randomization:
        shared |= {"hdris", "materials"}
    if args.hdris:
        shared.add("hdris")
    if args.materials:
        shared.add("materials")
    parts = [p.strip() for p in args.parts.split(",")] if args.parts else None

    if args.all:
        patterns = None
    else:
        if not args.task and not shared:
            ap.error("nothing selected: pass --all, --domain_randomization/--hdris/--materials, and/or --task NAME")
        patterns = build_patterns(args.task, parts, shared)
        print("include patterns:")
        for p in patterns:
            print("  ", p)

    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=patterns,
    )
    print(f"\nDownloaded to: {path}")


if __name__ == "__main__":
    main()

"""
Precompute USD asset sizes with IsaacSim (headless) and write asset_size.json next to each asset.

- Loads each USD into a stage
- Computes world-space axis-aligned bounding box using UsdGeom.BBoxCache
- Saves size_x/size_y/size_z and min/max extents to asset_size.json

Usage examples:
  python precompute_asset_sizes.py --usd /path/to/asset.usd
  python precompute_asset_sizes.py --dir /path/to/category_dir
"""

import os
import sys
import json
import glob
import argparse

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, Gf


def compute_usd_world_aabb(usd_path: str) -> dict:
    """Compute world-space axis-aligned bounding box for a USD asset.

    Returns a dict with min, max, and size along x/y/z.
    """
    ctx = omni.usd.get_context()
    if not ctx.open_stage(usd_path):
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    stage: Usd.Stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Stage is None after opening USD file")

    root_prim = stage.GetDefaultPrim()
    if root_prim is None or not root_prim.IsValid():
        # Fallback to pseudo-root
        root_prim = stage.GetPseudoRoot()

    # Compute world-space bbox using BBoxCache
    # Note: Older bindings don't accept keyword args; pass purposes positionally.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        True,   # useExtentsHint
        False,  # ignoreVisibility
    )
    bbox = bbox_cache.ComputeWorldBound(root_prim)
    aligned = bbox.ComputeAlignedBox()
    # aligned is a Gf.Range3d with min/max
    bmin: Gf.Vec3d = aligned.GetMin()
    bmax: Gf.Vec3d = aligned.GetMax()

    size_x = float(bmax[0] - bmin[0])
    size_y = float(bmax[1] - bmin[1])
    size_z = float(bmax[2] - bmin[2])

    return {
        "min": [float(bmin[0]), float(bmin[1]), float(bmin[2])],
        "max": [float(bmax[0]), float(bmax[1]), float(bmax[2])],
        "size": {"x": size_x, "y": size_y, "z": size_z},
    }


def write_asset_size_json(usd_path: str, data: dict, filename: str = "asset_size.json") -> str:
    out_dir = os.path.dirname(os.path.abspath(usd_path))
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path



def find_usds_in_category_dir(category_dir: str) -> list[str]:
    """Find exactly one USD per immediate subfolder under a category directory.

    Assumptions:
      - Each asset has its own subfolder under category_dir
      - The USD file name equals the subfolder name, e.g., mug_scale_2/mug_scale_2.usd
      - Fallback: if that exact name doesn't exist, pick the first *.usd under the subfolder
    """
    category_dir = os.path.abspath(category_dir)
    usd_paths: list[str] = []
    if not os.path.isdir(category_dir):
        return usd_paths

    for entry in sorted(os.listdir(category_dir)):
        subdir = os.path.join(category_dir, entry)
        if not os.path.isdir(subdir) or entry.startswith('.'):
            continue
        expected_usd = os.path.join(subdir, f"{entry}.usd")
        if os.path.isfile(expected_usd):
            usd_paths.append(expected_usd)
            continue
        # Fallback: first *.usd inside subdir
        candidates = sorted(glob.glob(os.path.join(subdir, "*.usd")))
        if candidates:
            usd_paths.append(candidates[0])
    return usd_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=str, default=None, help="Path to a single USD file")
    parser.add_argument("--dir", type=str, default=None, help="Category directory with one subfolder per asset")
    args = parser.parse_args()

    usd_paths: list[str] = []
    if args.usd:
        usd_paths.append(os.path.abspath(args.usd))
    if args.dir:
        cat_usds = find_usds_in_category_dir(args.dir)
        usd_paths.extend(cat_usds)

    usd_paths = [p for p in usd_paths if os.path.isfile(p) and p.lower().endswith(".usd")]
    if not usd_paths:
        print("[ERROR] No USD files provided or found. Use --usd or --dir.")
        sys.exit(1)

    print("Asset Size Precompute (IsaacSim)")
    print("=" * 60)
    for i, usd_path in enumerate(usd_paths):
        try:
            print(f"[{i+1}/{len(usd_paths)}] {usd_path}")
            aabb = compute_usd_world_aabb(usd_path)
            out = {
                "size": {
                    "x": aabb["size"]["x"],
                    "y": aabb["size"]["y"],
                    "z": aabb["size"]["z"],
                }
            }
            out_path = write_asset_size_json(usd_path, out)
            print(f"  -> wrote {out_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to process {usd_path}: {e}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()



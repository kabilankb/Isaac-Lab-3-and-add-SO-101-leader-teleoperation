"""JSON serialization and file I/O helpers."""

import os
import json
from typing import Any, Dict

import h5py
import torch


def serialize_dict(data: Any) -> Any:
    """Recursively convert torch.Tensors in a data structure to plain Python types.

    Scalars become Python numbers and larger tensors become nested lists so the
    result is JSON-serializable; dicts/lists/tuples are walked recursively.
    """
    if isinstance(data, torch.Tensor):
        return data.item() if data.numel() == 1 else data.cpu().numpy().tolist()
    if isinstance(data, dict):
        return {key: serialize_dict(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(serialize_dict(item) for item in data)
    return data


def resolve_split_dir(asset_dir: str, use_unseen: bool = False) -> str:
    """Resolve a train/eval split directory.

    ``asset_dir`` may point directly at a folder of assets, or at a parent that
    holds ``train/`` and ``eval/`` subfolders. In the latter case the ``eval``
    folder is returned when ``use_unseen`` is set, otherwise ``train``. Used for
    both material and HDRI directories.
    """
    if not asset_dir:
        return asset_dir
    split = "eval" if use_unseen else "train"
    sub = os.path.join(asset_dir, split)
    return sub if os.path.isdir(sub) else asset_dir


def save_json(file_path: str, data: Dict):
    """Write a dict to a JSON file, creating parent directories as needed.

    Args:
        file_path (str): Destination path for the JSON file.
        data (Dict): JSON-serializable data to write (indented with 4 spaces).
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def copy_dataset_metadata(input_file: str, output_file: str):
    """Copy metadata attributes from one HDF5 dataset's ``data`` group to another's.

    Preserves attributes such as ASSETS_INSTANCE_PATHS that downstream stages
    (e.g. data generation) need but the annotation export does not regenerate.

    Args:
        input_file (str): Path to the input HDF5 dataset.
        output_file (str): Path to the output HDF5 dataset.
    """
    try:
        with h5py.File(input_file, 'r') as f_in:
            if 'data' not in f_in:
                print("[WARNING] Input dataset has no 'data' group, skipping metadata copy")
                return

            data_group_in = f_in['data']

            # Metadata attributes to copy first (the rest are copied afterwards).
            metadata_attrs = ['ASSETS_INSTANCE_PATHS']

            with h5py.File(output_file, 'a') as f_out:
                if 'data' not in f_out:
                    print("[WARNING] Output dataset has no 'data' group, creating it")
                    f_out.create_group('data')

                data_group_out = f_out['data']

                for attr_name in metadata_attrs:
                    if attr_name in data_group_in.attrs:
                        # h5py handles bytes/string conversion on assignment.
                        data_group_out.attrs[attr_name] = data_group_in.attrs[attr_name]
                        print(f"[INFO] Copied metadata attribute: {attr_name}")
                    else:
                        print(f"[WARNING] Metadata attribute '{attr_name}' not found in input dataset")

                # Copy any remaining attributes not already present in the output.
                for attr_name in data_group_in.attrs.keys():
                    if attr_name not in metadata_attrs and attr_name not in data_group_out.attrs:
                        data_group_out.attrs[attr_name] = data_group_in.attrs[attr_name]
                        print(f"[INFO] Copied additional metadata attribute: {attr_name}")

    except Exception as e:
        print(f"[ERROR] Failed to copy metadata from input to output dataset: {e}")
        import traceback
        traceback.print_exc()

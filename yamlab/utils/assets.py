"""Asset metadata helpers (asset_size.json + USD path lookup)."""
import os
import json

OBJ_Z_OFFSET = 0.005  # 5mm above-table offset for object initial placement


def load_assets_instance_paths_from_dataset(dataset_file: str, assets_root_path: str = None):
    """Load ASSETS_INSTANCE_PATHS from an HDF5 dataset file.

    The dataset's ``data`` group stores each object's asset as a relative
    ``category/instance`` path (e.g. "PotFactory/PotFactory_000"). These are
    joined with ``assets_root_path`` to reconstruct full instance paths.

    Args:
        dataset_file (str): Path to the HDF5 dataset file.
        assets_root_path (str): Root path joined with each recorded category/instance.
            If None, the recorded relative paths are returned as-is.

    Returns:
        dict: Mapping from object name to asset instance path, or None if the
            attribute is unavailable or the file cannot be read.
    """
    import h5py

    try:
        with h5py.File(dataset_file, 'r') as f:
            if 'data' not in f:
                print(f"[WARNING] 'data' group not found in dataset file")
                return None

            data_group = f['data']
            if 'ASSETS_INSTANCE_PATHS' not in data_group.attrs:
                print(f"[WARNING] 'ASSETS_INSTANCE_PATHS' attribute not found in dataset")
                return None

            assets_attr = data_group.attrs['ASSETS_INSTANCE_PATHS']
            if isinstance(assets_attr, bytes):
                assets_attr = assets_attr.decode('utf-8')

            assets_relative_paths = json.loads(assets_attr)
            print(f"[INFO] Loaded ASSETS_INSTANCE_PATHS from dataset: {assets_relative_paths}")

            if assets_root_path is not None:
                return {
                    obj_name: os.path.join(assets_root_path, rel_path)
                    for obj_name, rel_path in assets_relative_paths.items()
                }
            print(f"[WARNING] assets_root_path not provided, using relative paths as-is")
            return assets_relative_paths

    except Exception as e:
        print(f"[ERROR] Failed to load ASSETS_INSTANCE_PATHS from dataset: {e}")
        return None


def load_asset_size(asset_folder_path: str) -> dict:
    """Load asset size information from asset_size.json in the asset folder.

    Args:
        asset_folder_path: Path to the asset instance folder.

    Returns:
        Dictionary with size info, or None if the file is missing or unreadable.
    """
    asset_size_path = os.path.join(asset_folder_path, "asset_size.json")
    if os.path.exists(asset_size_path):
        try:
            with open(asset_size_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load asset size from {asset_size_path}: {e}")
    return None


def get_asset_usd_path(asset_folder_path: str) -> str:
    """Return the USD file path for an asset instance folder.

    Args:
        asset_folder_path: Path to the asset instance folder.

    Returns:
        Absolute path to the .usd file.

    Raises:
        FileNotFoundError: If no .usd file is found in the folder.
    """
    folder_name = os.path.basename(asset_folder_path)
    usd_path = os.path.join(asset_folder_path, f"{folder_name}.usd")

    if os.path.exists(usd_path):
        return usd_path

    for file in os.listdir(asset_folder_path):
        if file.lower().endswith('.usd'):
            return os.path.join(asset_folder_path, file)
    raise FileNotFoundError(f"No USD file found in asset folder: {asset_folder_path}")


def find_contact_body_link(usd_path: str, default: str = "base_link") -> str:
    """Find the rigid-body prim sub-path to target in an object's contact-sensor filter.

    Prefers a prim named ``default`` ("base_link", as produced by the URDF->USD
    pipeline); otherwise the first prim with RigidBodyAPI (fallback CollisionAPI),
    relative to the default prim. Falls back to ``default`` on any failure.

    Args:
        usd_path (str): Path to the asset's .usd file.
        default (str): Link name to prefer and fall back to.

    Returns:
        str: Sub-path to append after "{ENV}/<obj>" ("" if the body is the asset root).
    """
    try:
        from pxr import Usd, UsdPhysics
    except Exception:
        return default
    try:
        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            return default
        root = stage.GetDefaultPrim()
        if not root or not root.IsValid():
            return default

        root_path = root.GetPath().pathString
        rigid = collider = None
        for prim in Usd.PrimRange(root):
            if prim.GetName() == default:
                return default
            if rigid is None and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid = prim
            if collider is None and prim.HasAPI(UsdPhysics.CollisionAPI):
                collider = prim

        target = rigid or collider
        if target is not None:
            return target.GetPath().pathString[len(root_path):].lstrip("/")
    except Exception as e:
        print(f"[WARNING] find_contact_body_link failed for {usd_path}: {e}")
    return default

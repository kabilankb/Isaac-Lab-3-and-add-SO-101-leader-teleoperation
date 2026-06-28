"""Experiment configuration (layered YAML).

All tunable per-task / per-mode experiment knobs live in YAML:

    configs/defaults.yaml          global defaults + per-mode (device, num_envs, ...)
    configs/tasks/<task>.yaml       per-task overrides (randomization, kwargs,
                                    success thresholds, friction, mimic signals)

`configs/loader.py` resolves the layered merge. This package exposes thin
accessors over the loader that return the shapes the rest of the codebase
expects, so call sites stay simple.

Robot/hardware calibration (camera intrinsics & poses, arm base poses,
actuator gains, finger geometry) is NOT here -- see configs/robot/yam.yaml.
"""

import math
import os

from . import loader
from .loader import resolve, VALID_MODES

# Maps the short mode names used by the teleop/mimic entry scripts to the
# canonical execution modes the loader (and defaults.yaml `modes:`) use.
_KWARGS_MODE_ALIASES = {
    "teleop": "teleoperation",
    "teleoperation": "teleoperation",
    "mimic": "mimicgen",
    "mimicgen": "mimicgen",
}


def get_task_config(env_name: str, mode: str = None, overrides: dict = None) -> dict:
    """Full resolved config dict for an (env_name, mode) pair.

    Thin pass-through to :func:`loader.resolve`. Used by task env/cfg modules
    to read task-specific sections (``friction``, ``mimic_signals``,
    ``success``) instead of importing module-level constants.

    Args:
        env_name (str): Exact gym env ID.
        mode (str | None): Execution mode, or None to skip per-mode layers.
        overrides (dict | None): Highest-precedence overrides (e.g. CLI flags).

    Returns:
        dict: The resolved config dict.
    """
    return loader.resolve(env_name, mode, overrides)


def list_env_names() -> list:
    """All gym env IDs declared across the task YAMLs (``env_names:`` lists).

    Returns:
        list: Sorted list of declared gym env IDs.
    """
    return sorted(loader._build_env_index().keys())


def resolve_device(env_name: str, mode: str, cli_device: str = None) -> str:
    """Pick the device for a run (used to launch the app AND set the sim).

    An explicit CLI device wins; otherwise the per-mode default from the YAML
    config is used (teleop/replay/mimicgen -> cpu, evaluation -> cuda).

    Entry scripts must override the AppLauncher ``--device`` default to None
    (``parser.set_defaults(device=None)``) so an omitted flag falls through to
    the YAML default here instead of AppLauncher's own "cuda:0". The returned
    value should be assigned back to ``args.device`` (so AppLauncher uses it)
    AND passed as ``device=`` to create_task_environment (so the sim matches).

    Args:
        env_name (str): Gym env ID (e.g. "<Task>-v0").
        mode (str): One of the four execution modes.
        cli_device (str | None): The user-supplied --device, or None if omitted.

    Returns:
        str: Device string, e.g. "cpu" or "cuda:0".
    """
    if cli_device:
        return cli_device
    return get_task_config(env_name, mode).get("sim", {}).get("device", "cpu")


def get_task_kwargs(env_name: str, mode: str = "teleop") -> dict:
    """Runtime kwargs forwarded to the task-env constructor.

    Args:
        env_name (str): Gym env ID (e.g. "<Task>-v0").
        mode (str): ``"teleop"`` (launch_follower) or ``"mimic"`` (generate_dataset).
            Mapped to the loader's canonical mode so the per-mode ``kwargs``
            overrides in the task YAML apply.

    Returns:
        dict: The resolved ``kwargs`` block (empty dict if the task has none).
    """
    resolved_mode = _KWARGS_MODE_ALIASES.get(mode, mode)
    return dict(get_task_config(env_name, resolved_mode).get("kwargs", {}))


def get_objects_config(env_name: str) -> dict:
    """The task's ``objects`` block, or {} if the task declares none.

    Args:
        env_name (str): Exact gym env ID.

    Returns:
        dict: ``{name: {mass, asset, randomization}}``, or {} if none declared.
    """
    return get_task_config(env_name).get("objects", {}) or {}


def get_grasp_detect_map(env_name: str) -> dict:
    """Per-arm grasp-detection targets: ``{arm_name: [object_name, ...]}``.

    Read from the task's ``grasp_detect`` block. Returns {} if the task
    declares none (callers then fall back to the task cfg's
    CONTACT_OBJECT_NAMES applied to both arms). Arms with an empty list are
    dropped so an absent/empty arm gets no contact sensors.

    Args:
        env_name (str): Exact gym env ID.

    Returns:
        dict: ``{arm_name: [object_name, ...]}``, or {} if none declared.
    """
    detect = get_task_config(env_name).get("grasp_detect") or {}
    return {arm: list(objs) for arm, objs in detect.items() if objs}


def _object_randomization(env_name: str, object_name: str) -> dict:
    """The ``randomization`` sub-block for one object, or {}.

    Args:
        env_name (str): Exact gym env ID.
        object_name (str): Object key within the task's ``objects`` block.

    Returns:
        dict: The object's ``randomization`` block, or {} if absent.
    """
    return (get_objects_config(env_name).get(object_name) or {}).get("randomization") or {}


def get_objects_randomization(env_name: str):
    """Per-object spawn-pose randomization for a task.

    XY position range is set ONE of two ways (mutually exclusive):
      - ``region_size: [rx, ry]``   -- bounding box the object footprint must stay
        inside; the effective center range subtracts the object's half-extent.
      - ``position_range: [dx, dy]`` -- direct +/- distance the object center may move.

    Args:
        env_name (str): Exact gym env ID.

    Returns:
        dict | None: ``{obj_name: {"region_size", "position_range",
        "orientation_range" (rad), "scale_range"}}`` (the unused XY-mode key is None),
        or None if the task declares no objects.
    """
    objs = get_objects_config(env_name)
    if not objs:
        return None
    out = {}
    for name in objs:
        r = _object_randomization(env_name, name)
        out[name] = {
            "region_size": tuple(r["region_size"]) if "region_size" in r else None,
            "position_range": tuple(r["position_range"]) if "position_range" in r else None,
            "orientation_range": math.radians(r.get("orientation_range_deg", 0.0)),
            "scale_range": tuple(r.get("scale_range", (1.0, 1.0))),
        }
    return out


def pose_range_for(env_name: str, object_name: str) -> dict:
    """Build a 6-axis ``pose_range`` dict for a reset event term.

    All six SE(3) keys are present; only x/y and yaw (+/- orientation range, radians)
    are populated. x/y is +/- ``position_range`` (direct mode) or +/- ``region_size``/2
    (bounding-box mode). Objects without a randomization block return an all-zero range.

    Args:
        env_name (str): Exact gym env ID.
        object_name (str): Object key within the task's ``objects`` block.

    Returns:
        dict: ``{axis: (lo, hi)}`` over x, y, z, roll, pitch, yaw.
    """
    r = _object_randomization(env_name, object_name)
    rad = math.radians(r.get("orientation_range_deg", 0.0))
    if "position_range" in r:
        dx, dy = r["position_range"]
        x_range, y_range = (-dx, dx), (-dy, dy)
    elif "region_size" in r:
        sx, sy = r["region_size"]
        x_range, y_range = (-sx / 2.0, sx / 2.0), (-sy / 2.0, sy / 2.0)
    else:
        return {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    return {
        "x": x_range,
        "y": y_range,
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (-rad, rad),
    }


def parse_asset_args(items) -> dict:
    """Parse repeatable ``--asset name=path`` CLI values into ``{name: path}``.

    Args:
        items (Sequence[str] | None): Raw ``name=path`` strings, or None.

    Returns:
        dict: ``{name: path}`` mapping.

    Raises:
        ValueError: if an item is missing the ``=`` separator.
    """
    out = {}
    for item in (items or []):
        if "=" not in item:
            raise ValueError(f"--asset expects 'name=path', got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def resolve_assets(env_name: str, cli_assets: dict = None,
                   dataset_paths: dict = None, assets_root: str = None) -> dict:
    """Resolve the ``{object_name: instance_dir}`` mapping for a task run.

    Merges three sources, highest precedence first:
      1. ``cli_assets``    -- explicit ``{name: path}`` overrides (e.g. parsed from
                              ``--asset name=path``).
      2. ``dataset_paths`` -- ``{name: path}`` recorded with a dataset; relative
                              paths are joined with ``assets_root``.
      3. the task YAML ``objects.<name>.asset`` default.
    Object names are validated against the task's declared objects.

    Args:
        env_name (str): Exact gym env ID.
        cli_assets (dict | None): Explicit ``{name: path}`` overrides.
        dataset_paths (dict | None): ``{name: path}`` recorded with a dataset;
            relative paths are joined with ``assets_root``.
        assets_root (str | None): Root dir for relative ``dataset_paths`` entries.

    Returns:
        dict: ``{object_name: instance_dir}`` for the run.

    Raises:
        ValueError: if a resolved name is not a declared object of the task.
    """
    objs = get_objects_config(env_name)
    resolved = {name: spec["asset"] for name, spec in objs.items() if spec.get("asset")}

    for name, path in (dataset_paths or {}).items():
        resolved[name] = (os.path.join(assets_root, path)
                          if assets_root and not os.path.isabs(path) else path)

    resolved.update(cli_assets or {})

    if objs:
        unknown = [n for n in resolved if n not in objs]
        if unknown:
            raise ValueError(
                f"asset names {unknown} are not declared objects of {env_name!r} "
                f"(declared: {list(objs)})"
            )
    return resolved

"""Layered YAML config loader.

Resolves a single config dict for a (env_name, mode) pair by deep-merging,
lowest -> highest precedence:

    1. configs/defaults.yaml                     (global defaults)
    2. defaults.yaml `modes:`[mode]              (global per-mode)
    3. tasks/<file>.yaml shared body             (task, all variants)
    4. tasks/<file>.yaml `modes:`[mode]          (task per-mode)
    5. tasks/<file>.yaml `variants:`[env_name]   (env-name-specific)
    6. variants[env_name] `modes:`[mode]         (env-name per-mode)
    7. overrides=                                (CLI / call-site, wins)

A later layer's value replaces an earlier layer's value for the same
key; nested dicts merge recursively, scalars and lists replace wholesale.

Task YAMLs are NOT matched by filename or gym version suffix. Each task
file declares an ``env_names:`` list of the exact gym IDs it serves; the
loader builds an env_name -> file index from those lists. This lets users
register their own variants (``<Task>-v1``) by adding the ID to a file,
and lets a single file serve several variants. Per-variant differences
(a variant with more/fewer knobs) go in an optional ``variants:`` block
keyed by the exact gym ID, which deep-merges over the shared task body.

This module is IsaacLab-free (yaml + pydantic + stdlib) so entry scripts can
resolve config BEFORE importing IsaacLab / launching the simulator. Every
resolved config is validated against the schema in
:mod:`yamlab.configs.schema`, so a misspelled key or wrong value type raises
here rather than being silently dropped downstream.
"""

import copy
import glob
import os

import yaml

from .schema import validate_resolved

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULTS_PATH = os.path.join(_CONFIG_DIR, "defaults.yaml")
_TASKS_DIR = os.path.join(_CONFIG_DIR, "tasks")

# Canonical execution modes (must match the keys under `modes:` in the YAML
# and the `mode` argument threaded through create_task_environment).
VALID_MODES = ("teleoperation", "replay", "mimicgen", "evaluation")

# Reserved top-level keys in a task YAML that are loader bookkeeping, not
# config values; stripped from the resolved result.
_RESERVED_KEYS = ("env_names", "variants", "modes")


def _load_yaml(path: str) -> dict:
    """Load a YAML file into a dict, returning {} if it is missing or empty.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed contents, or {} if absent/empty.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested dicts merge key-by-key; any non-dict value (scalar or list)
    replaces the base value wholesale. Neither input is mutated.

    Args:
        base (dict): Lower-precedence config layer.
        override (dict): Higher-precedence config layer.

    Returns:
        dict: A new merged dict.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _build_env_index() -> dict:
    """Scan all task YAMLs and map every declared env name -> file path.

    Returns:
        dict: ``{env_name: yaml_path}`` for every declared env name.

    Raises:
        ValueError: if two task files declare the same env name.
    """
    index = {}
    for path in sorted(glob.glob(os.path.join(_TASKS_DIR, "*.yaml"))):
        data = _load_yaml(path)
        for name in (data.get("env_names") or []):
            if name in index and index[name] != path:
                raise ValueError(
                    f"env name {name!r} declared in both "
                    f"{os.path.basename(index[name])} and {os.path.basename(path)}."
                )
            index[name] = path
    return index


def _apply_mode_layer(config: dict, layer: dict, mode: str) -> dict:
    """Merge ``layer`` minus its reserved keys, then ``layer['modes'][mode]``.

    Lets a file (or variant block) carry both plain keys and per-mode
    overrides, with the per-mode block taking precedence over the plain
    keys at the same level.

    Args:
        config (dict): The config accumulated so far (lower precedence).
        layer (dict): The YAML layer to merge in.
        mode (str | None): Execution mode whose ``modes`` sub-block to apply, or None.

    Returns:
        dict: A new merged config dict.
    """
    mode_section = (layer.get("modes") or {})
    body = {k: v for k, v in layer.items() if k not in _RESERVED_KEYS}
    merged = _deep_merge(config, body)
    if mode is not None and mode in mode_section:
        merged = _deep_merge(merged, mode_section[mode])
    return merged


def resolve(env_name: str = None, mode: str = None, overrides: dict = None) -> dict:
    """Resolve the layered config for an (env_name, mode) pair.

    Args:
        env_name (str | None): Exact gym env ID (e.g. "<Task>-Mimic-v0"). If None or
            not declared in any task YAML's ``env_names``, only the global
            defaults (+ mode + overrides) are used.
        mode (str | None): One of VALID_MODES, or None to skip per-mode layers.
        overrides (dict | None): Highest-precedence dict (e.g. CLI flags). Keys absent
            here fall back to lower layers. None is treated as {}.

    Returns:
        dict: A merged config dict with the reserved bookkeeping keys removed.

    Raises:
        ValueError: if ``mode`` is given but not in VALID_MODES.
    """
    if mode is not None and mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {list(VALID_MODES)}.")

    config = _apply_mode_layer({}, _load_yaml(_DEFAULTS_PATH), mode)

    if env_name is not None:
        path = _build_env_index().get(env_name)
        if path:
            task_layer = _load_yaml(path)
            # Shared task body (+ its per-mode block).
            config = _apply_mode_layer(config, task_layer, mode)
            # Env-name-specific variant block (+ its own per-mode block).
            variant = (task_layer.get("variants") or {}).get(env_name)
            if variant:
                config = _apply_mode_layer(config, variant, mode)

    if overrides:
        config = _deep_merge(config, overrides)

    for key in _RESERVED_KEYS:
        config.pop(key, None)

    return validate_resolved(config, env_name=env_name, mode=mode)

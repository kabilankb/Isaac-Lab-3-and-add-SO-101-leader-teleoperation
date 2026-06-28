"""Typed schema for a resolved experiment config.

The layered YAML (``defaults.yaml`` + ``tasks/<task>.yaml``) is merged into a
plain dict by :mod:`yamlab.configs.loader`. This module declares what that
merged dict is allowed to contain and validates it, so a misspelled key or a
wrong value type fails loudly at config-resolution time instead of being
silently ignored.

The schema covers the resolved config *after* the loader has applied the mode
layer and stripped the bookkeeping keys (``env_names``, ``variants``,
``modes``). Globally-defined sections (``sim``, ``rendering``, ``recording``,
``domain_randomization``) are validated strictly -- unknown keys are rejected. The task-defined numeric
blocks (``success``, ``mimic_signals``, ``friction``, ``kwargs``) hold
task-specific keys read by name in the task code, so they are validated for
value type but allow task-chosen key names.

Validation only raises; :func:`validate_resolved` returns the dict unchanged so
downstream consumers keep their plain-dict access.
"""

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError


class _Strict(BaseModel):
    """Base for sections whose key set is fixed (unknown keys are an error)."""
    model_config = ConfigDict(extra="forbid")


# ---- Simulation / physics -------------------------------------------

class PhysxCfg(_Strict):
    min_position_iteration_count: int
    min_velocity_iteration_count: int


class SimCfg(_Strict):
    dt: float
    decimation: int
    render_interval: int
    enable_scene_query_support: bool
    physx: PhysxCfg
    episode_length_s: float
    # Mode-dependent (set under modes:[mode].sim); absent when resolving without a mode.
    device: Optional[str] = None
    num_envs: Optional[int] = None


# ---- Rendering / recording ------------------------------------------
# Camera render output + LeRobot frame trimming. (Arm controller gains and
# grasp-detection thresholds are robot properties in configs/robot/<robot>.yaml.)

class RenderingCfg(_Strict):
    render_width: int
    render_height: int
    image_downsample_factor: int


class RecordingCfg(_Strict):
    discard_first_n_frames: int
    discard_last_n_frames: int


# ---- Domain randomization -------------------------------------------

class ToggleRangeCfg(_Strict):
    """A toggleable randomization knob; carries a `range` or `range_deg`."""
    enabled: bool
    range: Optional[list[float]] = None
    range_deg: Optional[list[float]] = None


class PerObjectMaterialCfg(_Strict):
    # `include` is "all" or a list of category folder names.
    include: Union[str, list[str]]
    exclude: Optional[list[str]] = None


class MaterialsCfg(_Strict):
    num_variants_per_material: int
    include: list[str]
    exclude: list[str]
    randomize_workstation: bool
    randomize_robot_arms: bool
    texture_rotation: ToggleRangeCfg
    texture_translation: ToggleRangeCfg
    color_tint: ToggleRangeCfg
    tint_workstation_and_robots: ToggleRangeCfg
    per_object: Optional[dict[str, PerObjectMaterialCfg]] = None


class LightingCfg(_Strict):
    dome_intensity_range: list[float]
    dome_color_temperature_range: list[float]
    hdri_rotation_range_deg: list[float]


class DomainRandomizationCfg(_Strict):
    randomize_interval_steps: int
    materials: MaterialsCfg
    lighting: LightingCfg


# ---- Objects --------------------------------------------------------

class ObjectRandomizationCfg(_Strict):
    # XY range is set EITHER as region_size (bounding box) OR position_range
    # (direct +/- distance); the two are mutually exclusive (validated in code).
    region_size: Optional[list[float]] = None
    position_range: Optional[list[float]] = None
    orientation_range_deg: float = 0.0
    scale_range: Optional[list[float]] = None


class ObjectCfg(_Strict):
    mass: Optional[float] = None          # omitted for articulated objects
    asset: Optional[str] = None           # default instance dir; CLI --asset overrides
    randomization: Optional[ObjectRandomizationCfg] = None


# ---- Top-level resolved config --------------------------------------

class ResolvedConfig(_Strict):
    """The full resolved config for one (env_name, mode) pair.

    Global sections are always present (from defaults.yaml). Mode-dependent
    fields (device/num_envs/enable_domain_randomization/pose_schedule) are
    present only after a mode layer is applied. Task-defined sections
    (objects/kwargs/success/mimic_signals/friction) are present only for
    tasks that declare them.
    """
    # Always present (global defaults). sim carries the mode-dependent device/num_envs
    # (under modes:[mode].sim) and episode_length_s.
    sim: SimCfg
    rendering: RenderingCfg
    recording: RecordingCfg
    domain_randomization: DomainRandomizationCfg

    # Present after a mode layer is applied (modes:[mode] in the YAML).
    enable_domain_randomization: Optional[bool] = None
    pose_schedule: Optional[dict[str, list[dict[str, Any]]]] = None

    # Task-defined sections. The numeric blocks hold task-specific keys read by
    # name in the task code, so their key names are not fixed here.
    objects: Optional[dict[str, ObjectCfg]] = None
    # Per-arm grasp targets: {arm_name: [object_name, ...]} (which object each arm grasps).
    grasp_detect: Optional[dict[str, list[str]]] = None
    kwargs: Optional[dict[str, Any]] = None
    success: Optional[dict[str, float]] = None
    mimic_signals: Optional[dict[str, float]] = None
    friction: Optional[dict[str, float]] = None


def validate_resolved(config: dict, env_name: Optional[str] = None,
                      mode: Optional[str] = None) -> dict:
    """Validate a resolved config dict against :class:`ResolvedConfig`.

    Args:
        config (dict): The merged config dict from the loader (reserved keys
            already stripped).
        env_name (str | None): Env ID being resolved, for the error message.
        mode (str | None): Mode being resolved, for the error message.

    Returns:
        dict: ``config`` unchanged (so callers keep plain-dict access).

    Raises:
        ValueError: if the config has an unknown key, a wrong value type, or an
            object that sets both ``region_size`` and ``position_range``.
    """
    try:
        ResolvedConfig.model_validate(config)
    except ValidationError as e:
        raise ValueError(
            f"Invalid config for env_name={env_name!r}, mode={mode!r}:\n{e}"
        ) from None

    for name, spec in (config.get("objects") or {}).items():
        rand = (spec or {}).get("randomization") or {}
        if rand.get("region_size") is not None and rand.get("position_range") is not None:
            raise ValueError(
                f"Object {name!r} (env_name={env_name!r}) sets both 'region_size' and "
                f"'position_range' under randomization; they are mutually exclusive."
            )
    return config

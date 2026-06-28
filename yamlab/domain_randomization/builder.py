"""Build a DomainRandomizationCfg from resolved YAML config.

Tunable randomization values come from the layered config
(configs/defaults.yaml ``domain_randomization:`` plus per-task overrides);
this module maps that dict onto the typed configclasses the
DomainRandomizationManager consumes. Runtime inputs that are not part of the
shared config -- whether DR is enabled, the material/HDRI directories, the
held-out split, and which objects to randomize -- are passed in separately.
"""

from typing import List, Optional

from .configs import (
    DomainRandomizationCfg,
    MaterialRandomizationCfg,
    LightingRandomizationCfg,
    PerObjectMaterialCfg,
)


def _as_category_list(include) -> List[str]:
    """Normalize a category include spec to a list.

    Args:
        include (str or list[str]): A single category name (e.g. ``"all"``) or
            a list of category names.

    Returns:
        list[str]: The categories as a list.
    """
    return [include] if isinstance(include, str) else list(include)


def _per_object_materials(materials_yaml: dict) -> dict:
    """Build the ``{object_name: PerObjectMaterialCfg}`` map from a materials YAML block.

    Args:
        materials_yaml (dict): The ``materials`` config dict, optionally with a
            ``per_object`` mapping of include/exclude specs.

    Returns:
        dict: Maps each object name to its ``PerObjectMaterialCfg``.
    """
    out = {}
    for obj, spec in (materials_yaml.get("per_object") or {}).items():
        out[obj] = PerObjectMaterialCfg(
            include=_as_category_list(spec.get("include", ["all"])),
            exclude=list(spec.get("exclude", []) or []),
        )
    return out


def build_domain_randomization_cfg(
    dr_yaml: dict,
    *,
    enabled: bool,
    material_randomization: bool = True,
    lighting_randomization: bool = True,
    materials_dir: Optional[str] = None,
    hdris_path: Optional[str] = None,
    use_unseen_materials: bool = False,
    object_names: Optional[List[str]] = None,
    prim_groups_per_object: Optional[dict] = None,
) -> DomainRandomizationCfg:
    """Construct a DomainRandomizationCfg from resolved YAML + runtime inputs.

    Args:
        dr_yaml (dict): The resolved ``domain_randomization`` config dict.
        enabled (bool): Master switch; when False a disabled config is returned.
        material_randomization (bool): Enable material randomization.
        lighting_randomization (bool): Enable lighting randomization.
        materials_dir (None or str): Material asset directory (CLI-provided).
        hdris_path (None or str): HDRI asset directory (CLI-provided).
        use_unseen_materials (bool): Use the held-out material split (evaluation).
        object_names (None or list[str]): Objects to randomize; overridden by
            ``materials.per_object`` keys when those are present.
        prim_groups_per_object (None or dict): Per-object mesh grouping for
            shared-material binding.

    Returns:
        DomainRandomizationCfg: The assembled configuration (disabled when ``enabled`` is False).
    """
    if not enabled:
        return DomainRandomizationCfg(enabled=False)

    dr_yaml = dr_yaml or {}
    mats = dr_yaml.get("materials", {}) or {}
    light = dr_yaml.get("lighting", {}) or {}

    per_object = _per_object_materials(mats)
    # Objects named under materials.per_object define the randomization set;
    # otherwise fall back to the provided object_names.
    objects = list(per_object) if per_object else list(object_names or [])

    # Only keys present in the YAML are set; absent keys keep the configclass
    # default (the single code-side fallback), so values aren't duplicated here.
    mat_kwargs = dict(
        enabled=material_randomization,
        object_names=objects,
        materials_dir=materials_dir,
        use_unseen_materials=use_unseen_materials,
        prim_groups_per_object=prim_groups_per_object or {},
        per_object_materials=per_object,
    )
    if "include" in mats:
        mat_kwargs["include"] = _as_category_list(mats["include"])
    if "exclude" in mats:
        mat_kwargs["exclude"] = list(mats["exclude"])
    if "num_variants_per_material" in mats:
        mat_kwargs["num_variants_per_material"] = mats["num_variants_per_material"]
    if "randomize_workstation" in mats:
        mat_kwargs["randomize_workstation"] = mats["randomize_workstation"]
    if "randomize_robot_arms" in mats:
        mat_kwargs["randomize_robot_arms"] = mats["randomize_robot_arms"]
    tex_rot = mats.get("texture_rotation", {}) or {}
    if "enabled" in tex_rot:
        mat_kwargs["randomize_texture_rotation"] = tex_rot["enabled"]
    if "range_deg" in tex_rot:
        mat_kwargs["texture_rotation_range"] = tuple(tex_rot["range_deg"])
    tex_trans = mats.get("texture_translation", {}) or {}
    if "enabled" in tex_trans:
        mat_kwargs["randomize_texture_translation"] = tex_trans["enabled"]
    if "range" in tex_trans:
        mat_kwargs["texture_translation_range"] = tuple(tex_trans["range"])
    tint = mats.get("color_tint", {}) or {}
    if "enabled" in tint:
        mat_kwargs["randomize_color_tint"] = tint["enabled"]
    if "range" in tint:
        mat_kwargs["color_tint_range"] = tuple(tint["range"])
    ws_tint = mats.get("tint_workstation_and_robots", {}) or {}
    if "enabled" in ws_tint:
        mat_kwargs["tint_workstation_and_robots"] = ws_tint["enabled"]
    if "range" in ws_tint:
        mat_kwargs["tint_range"] = tuple(ws_tint["range"])
    materials_cfg = MaterialRandomizationCfg(**mat_kwargs)

    use_hdri = hdris_path is not None
    light_kwargs = dict(
        enabled=lighting_randomization,
        use_hdri_textures=use_hdri,
        hdris_path=hdris_path,
        randomize_hdri_rotation=use_hdri,
    )
    if "dome_intensity_range" in light:
        light_kwargs["dome_intensity_range"] = tuple(light["dome_intensity_range"])
    if "dome_color_temperature_range" in light:
        light_kwargs["dome_color_temperature_range"] = tuple(light["dome_color_temperature_range"])
    if "hdri_rotation_range_deg" in light:
        light_kwargs["hdri_rotation_range"] = tuple(light["hdri_rotation_range_deg"])
    lighting_cfg = LightingRandomizationCfg(**light_kwargs)

    dr_kwargs = dict(enabled=True, materials=materials_cfg, lighting=lighting_cfg)
    if "randomize_on_reset" in dr_yaml:
        dr_kwargs["randomize_on_reset"] = dr_yaml["randomize_on_reset"]
    if "randomize_interval_steps" in dr_yaml:
        dr_kwargs["randomize_interval_steps"] = dr_yaml["randomize_interval_steps"]
    return DomainRandomizationCfg(**dr_kwargs)

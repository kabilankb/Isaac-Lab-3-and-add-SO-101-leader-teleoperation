"""Configuration dataclasses defining what is randomized and with which parameters."""

from dataclasses import field
from typing import Optional, List, Tuple, Dict
from isaaclab.utils import configclass


@configclass
class PerObjectMaterialCfg:
    """Per-object material category selection.

    Categories are the material folder names discovered under the materials
    directory (e.g. Wood, Metals, Carpet).

    Attributes:
        include: Category names to use, or ["all"] / "all" for every discovered
            category. Default ["all"].
        exclude: Category names to drop after ``include`` is applied. Default [].
        use_local_materials: Whether to also use explicit local .mdl files.
        local_material_paths: Explicit .mdl paths for this object.
    """
    include: List[str] = field(default_factory=lambda: ["all"])
    exclude: List[str] = field(default_factory=list)
    use_local_materials: bool = False
    local_material_paths: List[str] = field(default_factory=list)


@configclass
class MaterialRandomizationCfg:
    """Visual appearance randomization for scene objects via swapped MDL materials.

    Materials are preloaded at startup and a random variant is bound to each
    target object's mesh prims on reset.

    Attributes:
        enabled (bool): Whether material randomization is active.
        object_names (list[str]): Object names to randomize (e.g. ["obj_0", "obj_1"]);
            empty means no objects are randomized.
        randomize_workstation (bool): Also randomize workstation materials.
        randomize_robot_arms (bool): Also randomize both robot arm materials.
        materials_dir (None or str): Folder of material category subfolders to sample from.
        use_bundled_materials (bool): Use materials discovered under ``materials_dir``.
        use_local_materials (bool): Also use explicit local .mdl files (global default).
        local_material_paths (list[str]): Explicit .mdl paths (global default).
        per_object_materials (dict): Maps object name to its
            ``PerObjectMaterialCfg`` include/exclude spec; objects absent from the
            map fall back to the global ``include`` / ``exclude`` below.
        prim_groups_per_object (dict): Per-object mesh grouping for shared-material
            binding (see field comment).
        include (list[str]): Default material categories (folder names), or ["all"]
            for every discovered category.
        exclude (list[str]): Categories dropped after ``include`` is applied.
        randomize_texture_rotation (bool): Randomly rotate textures.
        texture_rotation_range (tuple): Rotation range in degrees.
        randomize_texture_translation (bool): Randomly translate textures in UV space.
        texture_translation_range (tuple): UV translation range.
        randomize_color_tint (bool): Apply a random diffuse color tint.
        color_tint_range (tuple): RGB tint multiplier range (min, max).
        num_variants_per_material (int): Preloaded variants per base material.
        tint_workstation_and_robots (bool): In-place tint workstation/arms without
            swapping their textures.
        tint_range (tuple): RGB tint multiplier range for the in-place tint.
        use_unseen_materials (bool): Use the held-out eval material split (evaluation only).
    """
    enabled: bool = False
    object_names: List[str] = field(default_factory=list)
    
    # Asset randomization flags
    # By default, only task-relevant objects are randomized (not workstation/robot arms)
    # Set these to True if you want to randomize workstation/robot arm materials as well
    randomize_workstation: bool = False
    randomize_robot_arms: bool = False
    
    # Material sources
    # materials_dir: a folder of <Category>/*.mdl subfolders to randomize over.
    #   Pass the split dir directly (e.g. .../materials/train or .../eval), the
    #   same way hdris_path points at a folder of .hdr files. If it is instead a
    #   ROOT that contains train/ and eval/ subfolders, the split is chosen by
    #   use_unseen_materials (eval = held-out). Local disk only; categories are
    #   the folder names. Set via configure_domain_randomization() / --materials_path.
    materials_dir: Optional[str] = None
    use_bundled_materials: bool = True
    use_local_materials: bool = False
    local_material_paths: List[str] = field(default_factory=list)
    
    # Per-object material configuration
    # Allows specifying different material categories for different objects
    # Example: {"obj_0": PerObjectMaterialCfg(include=["Wood"])}
    per_object_materials: Dict[str, PerObjectMaterialCfg] = field(default_factory=dict)

    # Per-object prim grouping: assigns ONE material per group instead of
    # the default per-mesh randomization. Maps:
    #   {object_name: {group_name: [substring_patterns]}}
    # A Mesh prim whose USD path contains any of a group's substrings is
    # assigned to that group; otherwise it lands in the implicit
    # "default" group. Each group gets exactly one random material per
    # reset/interval, shared across all its meshes. Useful for assets
    # whose pieces should look like one continuous painted surface in
    # real life rather than independently textured parts.
    # Example:
    #   {"obj_0": {"group_a": ["part_substring"]}}
    # → all prims whose path contains "part_substring" share one
    #   material; everything else shares another (the "default" group).
    prim_groups_per_object: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    
    # Default material categories for objects without a per-object entry.
    # include: category folder names, or ["all"] / "all" for every discovered
    # category; exclude: categories dropped after include. Default = all.
    include: List[str] = field(default_factory=lambda: ["all"])
    exclude: List[str] = field(default_factory=list)

    # Texture transform randomization
    randomize_texture_rotation: bool = True
    randomize_texture_translation: bool = True
    texture_rotation_range: Tuple[float, float] = (0.0, 360.0)
    texture_translation_range: Tuple[float, float] = (0.0, 100.0)
    
    # Color tint randomization
    randomize_color_tint: bool = True
    color_tint_range: Tuple[float, float] = (0.5, 1.5)
    
    # Number of material variants to preload per base material.
    # These field defaults are the fallback when a key is absent from the YAML
    # config; configs/defaults.yaml domain_randomization: is the source of truth
    # and should be kept consistent with them.
    num_variants_per_material: int = 2
    
    # In-place color tint for workstation and robot arms (without swapping textures).
    # Applies a random diffuse_tint to existing materials on each reset.
    tint_workstation_and_robots: bool = True
    tint_range: Tuple[float, float] = (0.9, 1.1)

    # Use holdout/unseen materials (for evaluation only)
    # If True, uses the held-out eval material split instead of train
    use_unseen_materials: bool = False


@configclass
class LightingRandomizationCfg:
    """Scene lighting variation via dome light intensity/color and optional HDRI.

    Attributes:
        enabled (bool): Whether lighting randomization is active.
        randomize_dome_intensity (bool): Randomize dome light intensity.
        dome_intensity_range (tuple): Dome intensity range (min, max).
        randomize_dome_color (bool): Randomize dome light color.
        dome_color_temperature_range (tuple): Color temperature range in Kelvin.
        use_hdri_textures (bool): Use HDRI textures for environment lighting.
        hdris_path (None or str): Folder containing .hdr files.
        randomize_hdri_rotation (bool): Randomly rotate the HDRI environment.
        hdri_rotation_range (tuple): HDRI rotation range in degrees.
        randomize_directional_light (bool): Randomize directional light properties.
        directional_intensity_range (tuple): Directional light intensity range.
    """
    enabled: bool = False
    
    # Dome light randomization
    randomize_dome_intensity: bool = True
    dome_intensity_range: Tuple[float, float] = (300.0, 1200.0)
    randomize_dome_color: bool = True
    dome_color_temperature_range: Tuple[float, float] = (2000.0, 8000.0)
    
    # HDRI-based lighting
    use_hdri_textures: bool = False
    hdris_path: Optional[str] = None
    randomize_hdri_rotation: bool = True
    hdri_rotation_range: Tuple[float, float] = (0.0, 360.0)
    
    # Directional light
    randomize_directional_light: bool = False
    directional_intensity_range: Tuple[float, float] = (1000.0, 5000.0)


@configclass
class DomainRandomizationCfg:
    """Top-level configuration aggregating material and lighting randomization.

    Attributes:
        enabled (bool): Master switch for all domain randomization.
        materials (MaterialRandomizationCfg): Material randomization configuration.
        lighting (LightingRandomizationCfg): Lighting randomization configuration.
        randomize_on_reset (bool): Apply randomization on environment reset.
        randomize_interval_steps (int): Also apply every N steps (0 = only on reset).
    """
    enabled: bool = False
    materials: MaterialRandomizationCfg = field(default_factory=MaterialRandomizationCfg)
    lighting: LightingRandomizationCfg = field(default_factory=LightingRandomizationCfg)
    
    # When to apply randomization
    randomize_on_reset: bool = True
    randomize_interval_steps: int = 30  # 0 = only on reset, >0 = also every N steps (default: every step)
    
    def __post_init__(self):
        """Propagate enabled state to sub-configs."""
        if self.enabled:
            # Sub-configs can still be individually disabled
            pass
        else:
            # If master is disabled, disable all sub-configs
            self.materials.enabled = False
            self.lighting.enabled = False

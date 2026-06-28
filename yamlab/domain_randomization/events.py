"""IsaacLab event terms and an orchestrator class for applying material and lighting randomization.

The event terms integrate with IsaacLab's EventManager to apply randomization
at reset and step intervals; ``DomainRandomizationManager`` offers an equivalent
state-encapsulated alternative to the module-global event functions.

Usage:
    @configclass
    class EventCfg:
        randomize_materials = EventTerm(
            func=randomize_object_materials,
            mode="reset",
            params={...}
        )
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Dict, Any

import torch
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .configs import DomainRandomizationCfg
from .materials import MaterialLibrary
from .lighting import LightingRandomizer


# Global storage for randomization state
# This allows event terms to access preloaded materials/lighting across calls
_material_library: Optional[MaterialLibrary] = None
_lighting_randomizer: Optional[LightingRandomizer] = None
_domain_rand_cfg: Optional[DomainRandomizationCfg] = None


def _find_dome_light_path(stage) -> Optional[str]:
    """Locate a dome light prim path in the stage, checking common paths then traversing.

    Args:
        stage: USD stage.

    Returns:
        None or str: Prim path of the first dome light found, or None.
    """
    try:
        from pxr import UsdLux
    except ImportError:
        return None
    
    # Common paths to check
    paths_to_check = [
        "/World/Light",
        "/World/DomeLight", 
        "/World/skyLight",
        "/World/Lighting/DomeLight",
    ]
    
    for path in paths_to_check:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdLux.DomeLight):
            return path
    
    # Search for any dome light
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            return str(prim.GetPath())
    
    return None


def randomize_object_materials(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_names: Optional[List[str]] = None,
) -> None:
    """Event term that binds random materials to the configured objects in the given environments.

    Each mesh prim is randomized independently unless the object has a
    ``prim_groups_per_object`` entry, in which case grouped binding is used.
    Optionally also randomizes the workstation and robot arms per config.

    Args:
        env: The IsaacLab environment.
        env_ids (torch.Tensor): int, shape (N,); environment indices being reset/stepped.
        object_names (None or list[str]): Objects to randomize; falls back to
            the material config's ``object_names`` when None.

    Example usage in EventCfg:
        randomize_materials = EventTerm(
            func=randomize_object_materials,
            mode="reset",
            params={"object_names": ["obj_0", "obj_1"]}
        )
    """
    global _material_library, _domain_rand_cfg
    
    if _material_library is None or not _domain_rand_cfg.materials.enabled:
        return
    
    stage = env.sim.stage
    
    # Get list of assets to randomize
    assets_to_randomize = []
    
    # Add task objects
    if object_names is None:
        object_names = _domain_rand_cfg.materials.object_names
    assets_to_randomize.extend([(name, f"/World/envs/env_{{idx}}/{name}") for name in object_names])
    
    # Add workstation if enabled
    if _domain_rand_cfg.materials.randomize_workstation:
        assets_to_randomize.append(("Workstation", "/World/envs/env_{idx}/Workstation"))
    
    # Add robot arms if enabled
    if _domain_rand_cfg.materials.randomize_robot_arms:
        assets_to_randomize.append(("LeftArm", "/World/envs/env_{idx}/LeftArm"))
        assets_to_randomize.append(("RightArm", "/World/envs/env_{idx}/RightArm"))
    
    if not assets_to_randomize:
        return
    
    # For each environment
    prim_groups_per_object = _domain_rand_cfg.materials.prim_groups_per_object
    for env_idx in env_ids.tolist():
        for asset_name, path_template in assets_to_randomize:
            # Construct prim path for this asset in this environment
            prim_path = path_template.format(idx=env_idx)

            # Get the asset prim
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            # If a group config exists for this asset, use the grouped
            # binder so each named group gets ONE shared material.
            # Otherwise fall back to the original per-mesh randomization.
            groups = prim_groups_per_object.get(asset_name)
            if groups:
                _bind_material_grouped(
                    stage, prim, _material_library,
                    object_name=asset_name, prim_groups=groups,
                )
            else:
                _bind_material_recursive(
                    stage, prim, _material_library, object_name=asset_name,
                )


def _bind_material_recursive(stage, prim, library: MaterialLibrary, object_name: Optional[str] = None) -> None:
    """Recursively bind an independently sampled material to each visual geometry prim in a subtree.

    Binds materials to all visual geometry prims (Mesh, Capsule, Cylinder,
    Sphere, etc.) and skips collision geometry (prims under a "collisions" scope).

    Args:
        stage: USD stage.
        prim: Subtree root prim to bind materials to.
        library (MaterialLibrary): Material library to sample from.
        object_name (None or str): Object name for per-object material selection.
    """
    try:
        from pxr import UsdGeom
    except ImportError:
        return
    
    # Skip collision geometry - only randomize visual geometry
    prim_name = prim.GetName()
    if prim_name == "collisions" or "collision" in prim_name.lower():
        # Skip recursing into collisions scope
        return
    
    # Check if this prim is any type of visual geometry that can have materials
    # This includes: Mesh, Capsule, Cylinder, Sphere, Cube, Cone, etc.
    # All geometry prims inherit from UsdGeom.Imageable
    if prim.IsA(UsdGeom.Imageable):
        # Check if it's actually a geometry prim (not just a container/scope)
        # Geometry prims have specific types like Mesh, Capsule, etc.
        # We exclude Scope and Xform as they're typically containers
        if (prim.IsA(UsdGeom.Mesh) or 
            prim.IsA(UsdGeom.Capsule) or 
            prim.IsA(UsdGeom.Cylinder) or 
            prim.IsA(UsdGeom.Sphere) or 
            prim.IsA(UsdGeom.Cube) or 
            prim.IsA(UsdGeom.Cone) or
            prim.IsA(UsdGeom.Points) or
            prim.IsA(UsdGeom.Curves)):
            # This is a visual geometry prim that can have materials
            # Pass object_name for per-object material selection
            material_path = library.get_random_material(object_name=object_name)
            if material_path:
                library.bind_to_prim(stage, str(prim.GetPath()), material_path)
    
    # Recurse to children (but skip collisions scope)
    for child in prim.GetChildren():
        _bind_material_recursive(stage, child, library, object_name=object_name)


def _bind_material_grouped(
    stage,
    root_prim,
    library: MaterialLibrary,
    object_name: Optional[str] = None,
    prim_groups: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Group-aware variant of ``_bind_material_recursive`` that shares one material per named group.

    Walks the prim subtree once, sorts each Mesh-like visual prim into a named
    group by whether its USD path contains any of the group's substring
    patterns, then samples ONE random material per group and binds it to every
    prim in that group. Useful when an asset's pieces should look like one
    continuous material rather than independently textured parts.

    Args:
        stage: USD stage.
        root_prim: Subtree root (typically ``/World/envs/env_N/obj_0``).
        library (MaterialLibrary): Material library for sampling.
        object_name (None or str): Forwarded to ``library.get_random_material``
            so per-object category filters (``PerObjectMaterialCfg``) apply.
        prim_groups (None or dict): ``{group_name: [substring_patterns]}``. A
            Mesh prim whose path contains a substring from a group's list is
            assigned to that group; the first matching group wins (dict
            iteration order). Prims matching no group go to the implicit
            ``"default"`` group. If None or empty, falls back to per-mesh
            randomization via ``_bind_material_recursive``.
    """
    if not prim_groups:
        _bind_material_recursive(stage, root_prim, library, object_name=object_name)
        return

    try:
        from pxr import UsdGeom
    except ImportError:
        return

    # Walk the subtree and bucket visual prims by group.
    buckets: Dict[str, List[Any]] = {"default": []}
    for name in prim_groups:
        buckets.setdefault(name, [])

    def _walk(p):
        pname = p.GetName()
        if pname == "collisions" or "collision" in pname.lower():
            return
        if p.IsA(UsdGeom.Imageable) and (
            p.IsA(UsdGeom.Mesh)
            or p.IsA(UsdGeom.Capsule)
            or p.IsA(UsdGeom.Cylinder)
            or p.IsA(UsdGeom.Sphere)
            or p.IsA(UsdGeom.Cube)
            or p.IsA(UsdGeom.Cone)
            or p.IsA(UsdGeom.Points)
            or p.IsA(UsdGeom.Curves)
        ):
            path_str = str(p.GetPath())
            chosen = "default"
            for gname, patterns in prim_groups.items():
                if gname == "default":
                    continue
                if any(pat in path_str for pat in patterns):
                    chosen = gname
                    break
            buckets[chosen].append(p)
        for child in p.GetChildren():
            _walk(child)

    _walk(root_prim)

    # One material per non-empty group, bound to every member.
    for gname, prims in buckets.items():
        if not prims:
            continue
        material_path = library.get_random_material(object_name=object_name)
        if not material_path:
            continue
        for p in prims:
            library.bind_to_prim(stage, str(p.GetPath()), material_path)


def _apply_color_tint_recursive(stage, prim, tint_range: tuple) -> None:
    """Apply a single random diffuse tint in-place to every bound material in a subtree.

    Modifies the shader in place without swapping the material, so the original
    texture is preserved while its color shifts slightly. One tint is sampled
    per top-level call and shared across all meshes of the asset so it looks
    uniformly tinted.

    Args:
        stage: USD stage.
        prim: Root prim to traverse.
        tint_range (tuple): (min, max) for each RGB channel of the tint multiplier.
    """
    try:
        from pxr import UsdShade, Sdf, Gf
    except ImportError:
        return

    # Sample one tint for the whole asset (called once per asset root)
    tint = Gf.Vec3f(
        float(np.random.uniform(*tint_range)),
        float(np.random.uniform(*tint_range)),
        float(np.random.uniform(*tint_range)),
    )

    def _apply(p):
        # Find material binding on this prim
        binding_api = UsdShade.MaterialBindingAPI(p)
        mat, _ = binding_api.ComputeBoundMaterial()
        if mat:
            # Find the shader within the material
            for child in mat.GetPrim().GetChildren():
                shader = UsdShade.Shader(child)
                if shader:
                    tint_input = shader.GetInput("diffuse_tint")
                    if tint_input:
                        tint_input.Set(tint)
                    else:
                        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(tint)
                    break  # one shader per material is enough

        for child in p.GetChildren():
            _apply(child)

    _apply(prim)


def randomize_scene_lighting(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
) -> None:
    """Event term that applies random scene lighting once per reset call.

    Lighting is global (not per-environment), so it is applied once per call
    regardless of how many environment indices are passed.

    Args:
        env: The IsaacLab environment.
        env_ids (torch.Tensor): int, shape (N,); environment indices being reset.

    Example usage in EventCfg:
        randomize_lighting = EventTerm(
            func=randomize_scene_lighting,
            mode="reset",
        )
    """
    global _lighting_randomizer, _domain_rand_cfg
    
    if _lighting_randomizer is None or not _domain_rand_cfg.lighting.enabled:
        return
    
    # Only randomize once per reset call (lighting is global)
    # We use the first env_id as a trigger
    stage = env.sim.stage
    _lighting_randomizer.randomize(stage)


def apply_domain_randomization(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
) -> None:
    """Combined event term applying both material and lighting randomization.

    Args:
        env: The IsaacLab environment.
        env_ids (torch.Tensor): int, shape (N,); environment indices being reset.

    Example usage in EventCfg:
        domain_randomization = EventTerm(
            func=apply_domain_randomization,
            mode="reset",
        )
    """
    global _domain_rand_cfg
    
    if _domain_rand_cfg is None or not _domain_rand_cfg.enabled:
        return
    
    if _domain_rand_cfg.materials.enabled:
        randomize_object_materials(env, env_ids)
    
    if _domain_rand_cfg.lighting.enabled:
        randomize_scene_lighting(env, env_ids)


# ==================== Class-based orchestrator ====================
# State-encapsulated alternative to the module-global event functions above.


class DomainRandomizationManager:
    """Orchestrates material and lighting randomization with all state held in one object.

    A state-encapsulated alternative to the module-global event functions.

    Usage:
        # In environment __init__
        self.domain_rand_manager = DomainRandomizationManager(cfg, env_name="<Task>-v0")
        self.domain_rand_manager.initialize(self)

        # On reset
        self.domain_rand_manager.apply_randomization(env_ids)

        # On step (if interval > 0)
        self.domain_rand_manager.maybe_apply_randomization_on_step()
    """

    def __init__(self, cfg: DomainRandomizationCfg, env_name: Optional[str] = None):
        """Initialize the manager.

        Args:
            cfg (DomainRandomizationCfg): Domain randomization configuration.
            env_name (None or str): Environment name, kept for logging/back-compat.
                The objects to randomize come from ``cfg.materials.object_names``
                (set by ``configure_domain_randomization`` from the task's objects
                or the YAML ``domain_randomization`` section).
        """
        self.cfg = cfg
        self._env_name = env_name
        self._material_library: Optional[MaterialLibrary] = None
        self._lighting_randomizer: Optional[LightingRandomizer] = None
        self._env: Optional["ManagerBasedEnv"] = None
        self._initialized = False
        self._step_count = 0  # Track steps for interval-based randomization

    @property
    def is_enabled(self) -> bool:
        """Whether domain randomization is enabled.

        Returns:
            bool: The master enabled flag.
        """
        return self.cfg.enabled

    def initialize(self, env: "ManagerBasedEnv") -> bool:
        """Preload materials and set up the lighting randomizer for the environment.

        Args:
            env: The IsaacLab environment.

        Returns:
            bool: True if initialization succeeded.
        """
        if self._initialized:
            return True
        
        self._env = env
        
        if not self.cfg.enabled:
            print("[INFO] Domain randomization is disabled")
            self._initialized = True
            return True
        
        stage = env.sim.stage
        
        # Initialize material library
        if self.cfg.materials.enabled:
            self._material_library = MaterialLibrary(self.cfg.materials)
            success = self._material_library.preload(stage)
            if not success:
                print("[WARNING] Failed to preload materials")
                self.cfg.materials.enabled = False
        
        # Initialize lighting randomizer
        if self.cfg.lighting.enabled:
            self._lighting_randomizer = LightingRandomizer(self.cfg.lighting)
            
            dome_light_path = _find_dome_light_path(stage)
            if dome_light_path:
                self._lighting_randomizer.set_dome_light_path(dome_light_path)
            success = self._lighting_randomizer.initialize(stage)
            if not success:
                print("[WARNING] Failed to initialize lighting randomizer")
                self.cfg.lighting.enabled = False
        
        self._initialized = True
        
        num_materials = self._material_library.num_materials if self._material_library else 0
        print(f"[INFO] Domain randomization initialized:")
        print(f"  - Materials: {self.cfg.materials.enabled} ({num_materials} variants)")
        print(f"  - Lighting: {self.cfg.lighting.enabled}")
        
        return True
    
    def apply_randomization(self, env_ids: torch.Tensor) -> None:
        """Randomize materials (and global lighting) for the specified environments.

        Binds random materials to task objects (and optionally the workstation
        and robot arms), applies the in-place tint to the workstation/arms when
        enabled, and randomizes the global scene lighting.

        Args:
            env_ids (torch.Tensor): int, shape (N,); environment indices to randomize.
        """
        if not self.cfg.enabled or self._env is None:
            return

        stage = self._env.sim.stage

        # Get list of assets to randomize
        assets_to_randomize = []

        # Add task objects
        object_names = self.cfg.materials.object_names
        assets_to_randomize.extend([(name, f"/World/envs/env_{{idx}}/{name}") for name in object_names])

        # Add workstation if enabled
        if self.cfg.materials.randomize_workstation:
            assets_to_randomize.append(("Workstation", "/World/envs/env_{idx}/Workstation"))

        # Add robot arms if enabled
        if self.cfg.materials.randomize_robot_arms:
            assets_to_randomize.append(("LeftArm", "/World/envs/env_{idx}/LeftArm"))
            assets_to_randomize.append(("RightArm", "/World/envs/env_{idx}/RightArm"))

        # Material randomization. By default each mesh prim is randomized
        # independently; objects listed in ``prim_groups_per_object`` use
        # the grouped binder so all meshes in a named group share one
        # material.
        if self.cfg.materials.enabled and self._material_library is not None:
            prim_groups_per_object = self.cfg.materials.prim_groups_per_object
            for env_idx in env_ids.tolist():
                for asset_name, path_template in assets_to_randomize:
                    prim_path = path_template.format(idx=env_idx)
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        # Extract object name from path (e.g., "/World/envs/env_0/obj_0" -> "obj_0")
                        # For workstation/arms, use the asset name directly
                        object_name = None
                        if asset_name in ["Workstation", "LeftArm", "RightArm"]:
                            object_name = asset_name
                        elif asset_name in self.cfg.materials.object_names:
                            object_name = asset_name

                        groups = prim_groups_per_object.get(asset_name)
                        if groups:
                            _bind_material_grouped(
                                stage, prim, self._material_library,
                                object_name=object_name, prim_groups=groups,
                            )
                        else:
                            _bind_material_recursive(
                                stage, prim, self._material_library,
                                object_name=object_name,
                            )

        # In-place color tint for workstation and robot arms (keeps original textures)
        if self.cfg.materials.enabled and self.cfg.materials.tint_workstation_and_robots:
            tint_targets = []
            tint_targets.append("/World/envs/env_{idx}/Workstation")
            tint_targets.append("/World/envs/env_{idx}/LeftArm")
            tint_targets.append("/World/envs/env_{idx}/RightArm")
            for env_idx in env_ids.tolist():
                for path_template in tint_targets:
                    prim_path = path_template.format(idx=env_idx)
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        _apply_color_tint_recursive(stage, prim, self.cfg.materials.tint_range)

        # Lighting randomization (global)
        if self.cfg.lighting.enabled and self._lighting_randomizer is not None:
            self._lighting_randomizer.randomize(stage)
    
    def maybe_apply_randomization_on_step(self) -> None:
        """Re-randomize all environments at the configured step interval.

        Should be called every step. Applies randomization only when
        ``randomize_interval_steps > 0`` and the step count is a multiple of it.
        """
        if not self.cfg.enabled or self._env is None:
            return
        
        if self.cfg.randomize_interval_steps <= 0:
            return
        
        self._step_count += 1
        
        # Apply randomization every N steps
        if self._step_count % self.cfg.randomize_interval_steps == 0:
            # Randomize all environments
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
            self.apply_randomization(env_ids)
    
    def reset_step_count(self) -> None:
        """Reset the interval step counter; called on environment reset."""
        self._step_count = 0

    def cleanup(self) -> None:
        """Release the material library, lighting randomizer, and environment references."""
        self._material_library = None
        self._lighting_randomizer = None
        self._env = None
        self._initialized = False


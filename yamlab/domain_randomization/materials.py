"""Material library management, binding, and texture-transform randomization for IsaacLab environments."""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .configs import MaterialRandomizationCfg


def _run_async_safely(coro):
    """Run an async coroutine whether or not an event loop is already running.

    If no loop is running, runs it directly; otherwise runs it in a separate
    thread with its own event loop.

    Args:
        coro: Async coroutine to run.

    Returns:
        Result of the coroutine.
    """
    import asyncio
    import concurrent.futures
    
    try:
        # Check if there's a running event loop
        asyncio.get_running_loop()
        # If we get here, there's a running loop
        # Run the coroutine in a separate thread with a new event loop
        def run_in_thread():
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=30)  # 30 second timeout
    except RuntimeError:
        # No running event loop, safe to use asyncio.run() directly
        return asyncio.run(coro)


# Materials live on disk as ``<materials_dir>/<Category>/*.mdl``, each .mdl
# beside its texture folder; <Category> is the folder name (Wood, Metals,
# Carpet, ...). ``materials_dir`` points directly at the folder of category
# subfolders, the same way ``hdris_path`` points at a folder of .hdr files. A
# train/ or eval/ split is selected from the path via resolve_split_dir.
import glob

from yamlab.utils.io import resolve_split_dir

ALL = "all"  # include sentinel: select every category


def discover_materials(materials_dir: str) -> Dict[str, List[str]]:
    """Group the .mdl files under ``<materials_dir>/<Category>/`` by category.

    Args:
        materials_dir (str): Folder of ``<Category>/*.mdl`` subfolders.

    Returns:
        dict: Maps each non-empty category name to a list of absolute .mdl
            paths. Empty dict if the directory does not exist.
    """
    out: Dict[str, List[str]] = {}
    if not materials_dir or not os.path.isdir(materials_dir):
        return out
    for category in sorted(os.listdir(materials_dir)):
        cat_dir = os.path.join(materials_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        mdls = sorted(glob.glob(os.path.join(cat_dir, "*.mdl")))
        if mdls:
            out[category] = mdls
    return out


def resolve_categories(available, include=ALL, exclude=None) -> List[str]:
    """Resolve an include/exclude spec against the available category names.

    Args:
        available (list[str]): Category names available on disk.
        include (str or list[str]): ``"all"`` / ``["all"]`` selects every
            available category; otherwise the listed categories (intersected
            with ``available``).
        exclude (None or list[str]): Categories dropped after include.

    Returns:
        list[str]: Selected category names, sorted.
    """
    available = list(available)
    exclude = set(exclude or [])
    if include == ALL or include == [ALL] or include is None:
        selected = set(available)
    else:
        selected = set(include) & set(available)
    return sorted(selected - exclude)


def select_material_paths(materials_dir, include=ALL, exclude=None) -> List[str]:
    """Return .mdl paths under ``materials_dir`` whose category passes the include/exclude filter.

    Args:
        materials_dir (str): Folder of ``<Category>/*.mdl`` subfolders.
        include (str or list[str]): Category include spec (see ``resolve_categories``).
        exclude (None or list[str]): Categories dropped after include.

    Returns:
        list[str]: Absolute .mdl paths, sorted by category then path.
    """
    by_cat = discover_materials(materials_dir)
    paths: List[str] = []
    for category in resolve_categories(by_cat.keys(), include, exclude):
        paths.extend(by_cat[category])
    return paths

@dataclass
class MaterialVariant:
    """A preloaded material instance with its randomized texture-transform parameters.

    Attributes:
        prim_path (str): USD prim path of the created material.
        base_material_path (str): Source .mdl file path.
        texture_rotation (float): Rotation angle in degrees.
        texture_translation (tuple): UV translation (u, v).
        color_tint (None or tuple): RGB color tint (r, g, b), or None if untinted.
    """
    prim_path: str
    base_material_path: str
    texture_rotation: float = 0.0
    texture_translation: Tuple[float, float] = (0.0, 0.0)
    color_tint: Optional[Tuple[float, float, float]] = None


class MaterialLibrary:
    """Library of preloaded MDL materials supporting per-object random selection and binding.

    The library preloads materials at startup and provides random material
    selection during environment reset, optionally filtered per object.

    Usage:
        library = MaterialLibrary(cfg)
        library.preload(stage)  # Call once at startup
        material_path = library.get_random_material(object_name="obj_0")
        library.bind_to_prim(stage, object_prim_path, material_path)
    """

    def __init__(self, cfg: MaterialRandomizationCfg):
        """Initialize the material library.

        Args:
            cfg (MaterialRandomizationCfg): Material randomization configuration.
        """
        self.cfg = cfg
        # Store materials per object (for per-object material selection)
        # Key: object_name, Value: List[MaterialVariant]
        self._materials_by_object: Dict[str, List[MaterialVariant]] = {}
        # Also keep a shared pool for backward compatibility
        self._materials: List[MaterialVariant] = []
        self._preloaded = False
        self._material_root_path = "/World/Looks/RandomMaterials"
        
    @property
    def is_preloaded(self) -> bool:
        """Whether materials have been preloaded.

        Returns:
            bool: True if preload has run.
        """
        return self._preloaded

    @property
    def num_materials(self) -> int:
        """Number of preloaded material variants in the shared pool.

        Returns:
            int: Count of available variants.
        """
        return len(self._materials)

    def get_base_material_paths(self, object_name: Optional[str] = None) -> List[str]:
        """Resolve the base .mdl paths to load for an object, applying include/exclude filters.

        Args:
            object_name (None or str): If given and present in
                ``per_object_materials``, returns that object's allowed
                materials; otherwise returns the global default selection.

        Returns:
            list[str]: Base .mdl material paths.
        """
        # Directory to scan: materials_dir as-is, or its train/eval subfolder
        # if materials_dir is a root containing the split (use_unseen -> eval).
        materials_dir = resolve_split_dir(self.cfg.materials_dir, self.cfg.use_unseen_materials)

        # Per-object include/exclude if present, else the global defaults.
        if object_name and object_name in self.cfg.per_object_materials:
            obj_cfg = self.cfg.per_object_materials[object_name]
            include, exclude = obj_cfg.include, obj_cfg.exclude
            use_local = obj_cfg.use_local_materials
            local_paths = obj_cfg.local_material_paths
        else:
            include, exclude = self.cfg.include, self.cfg.exclude
            use_local = self.cfg.use_local_materials
            local_paths = self.cfg.local_material_paths

        paths: List[str] = []
        if self.cfg.use_bundled_materials:
            paths.extend(select_material_paths(materials_dir, include, exclude))
        if use_local:
            paths.extend(local_paths)
        return paths
    
    def preload(self, stage) -> bool:
        """Create material prims in the stage from the configured base materials.

        Creates ``num_variants_per_material`` randomized-transform variants per
        base material. Should be called once at environment startup.

        Args:
            stage: USD stage to create materials in.

        Returns:
            bool: True if preloading succeeded, False otherwise.
        """
        if self._preloaded:
            print("[INFO] Materials already preloaded, skipping")
            return True
        
        if not self.cfg.enabled:
            print("[INFO] Material randomization disabled, skipping preload")
            return True

        try:
            # Try to import Omniverse modules
            from pxr import Sdf, Gf
            import omni.kit.commands
            import omni.kit.app

            # Enable material library extension (must be done before importing)
            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("omni.kit.material.library", True)
            import omni.kit.material.library
        except ImportError as e:
            print(f"[WARNING] Failed to import Omniverse modules: {e}")
            print("[WARNING] Material randomization will be disabled")
            self.cfg.enabled = False
            return False
        
        # Create root prim for materials
        if not stage.GetPrimAtPath(self._material_root_path).IsValid():
            stage.DefinePrim(self._material_root_path, "Scope")
        
        # Preload materials for all objects (shared pool + per-object pools)
        # First, get all unique material paths needed
        all_material_paths = set()
        
        # Add default materials
        default_paths = self.get_base_material_paths()
        all_material_paths.update(default_paths)
        
        # Add per-object materials
        for object_name in self.cfg.per_object_materials.keys():
            obj_paths = self.get_base_material_paths(object_name)
            all_material_paths.update(obj_paths)
        
        base_paths = list(all_material_paths)
        n_local = sum(1 for p in base_paths if not p.startswith("http"))
        n_remote = len(base_paths) - n_local
        source = "local" if n_remote == 0 else f"{n_local} local, {n_remote} remote" if n_local > 0 else "remote"
        print(f"[INFO] Loading {len(base_paths)} base materials ({source})")
        material_count = 0
        
        for base_path in base_paths:
            # Get subidentifiers (material variants in the MDL file)
            # Use safe async runner that handles both cases: no event loop or running event loop
            try:
                subidentifiers = _run_async_safely(
                    omni.kit.material.library.get_subidentifier_from_mdl(base_path)
                )
                # Ensure subidentifiers is a list
                if not subidentifiers:
                    subidentifiers = [None]
            except Exception as e:
                print(f"[WARNING] Failed to get subidentifiers for {base_path}: {e}")
                # If subidentifier fetch fails, use None to create material with default variant
                subidentifiers = [None]
            
            for subidentifier in subidentifiers:
                # Create multiple variants with different transforms
                for variant_idx in range(self.cfg.num_variants_per_material):
                    material_prim_path = f"{self._material_root_path}/Material_{material_count}"
                    
                    # Create the material prim
                    try:
                        success, result = omni.kit.commands.execute(
                            "CreateMdlMaterialPrimCommand",
                            mtl_url=str(base_path),
                            mtl_name=str(subidentifier) if subidentifier else "",
                            mtl_path=material_prim_path
                        )
                        
                        if not success:
                            print(f"[WARNING] Failed to create material: {material_prim_path}")
                            continue
                    except Exception as e:
                        print(f"[WARNING] Error creating material {base_path}: {e}")
                        continue
                    
                    # Apply random transforms
                    texture_rotation = 0.0
                    texture_translation = (0.0, 0.0)
                    color_tint = None
                    
                    shader_prim_path = f"{material_prim_path}/Shader"
                    shader_prim = stage.GetPrimAtPath(shader_prim_path)
                    
                    if shader_prim.IsValid():
                        # Enable UV projection (matches IsaacLabPlayground pattern)
                        shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
                        shader_prim.CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
                        
                        if self.cfg.randomize_texture_rotation:
                            texture_rotation = np.random.uniform(*self.cfg.texture_rotation_range)
                            shader_prim.CreateAttribute("inputs:texture_rotate", Sdf.ValueTypeNames.Float).Set(texture_rotation)
                        
                        if self.cfg.randomize_texture_translation:
                            texture_translation = (
                                np.random.uniform(*self.cfg.texture_translation_range),
                                np.random.uniform(*self.cfg.texture_translation_range)
                            )
                            shader_prim.CreateAttribute("inputs:texture_translate", Sdf.ValueTypeNames.Float2).Set(texture_translation)
                        
                        if self.cfg.randomize_color_tint:
                            color_tint = (
                                np.random.uniform(*self.cfg.color_tint_range),
                                np.random.uniform(*self.cfg.color_tint_range),
                                np.random.uniform(*self.cfg.color_tint_range)
                            )
                            shader_prim.CreateAttribute("inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
                                Gf.Vec3f(*color_tint)
                            )
                    
                    # Store material variant
                    variant = MaterialVariant(
                        prim_path=material_prim_path,
                        base_material_path=base_path,
                        texture_rotation=texture_rotation,
                        texture_translation=texture_translation,
                        color_tint=color_tint
                    )
                    self._materials.append(variant)
                    material_count += 1
        
        self._preloaded = True
        print(f"[INFO] Preloaded {material_count} material variants")
        return True
    
    def get_random_material(self, object_name: Optional[str] = None) -> Optional[str]:
        """Sample a random preloaded material prim path, optionally filtered for an object.

        Args:
            object_name (None or str): If given and present in
                ``per_object_materials``, samples only from that object's
                allowed categories; otherwise samples from the default pool.

        Returns:
            None or str: Prim path of a random material, or None if none available.
        """
        if not self._materials:
            return None
        
        # If per-object material config exists, filter materials
        if object_name and object_name in self.cfg.per_object_materials:
            # Get allowed material paths for this object
            allowed_paths = set(self.get_base_material_paths(object_name))
            
            # Filter materials to only those in allowed paths
            filtered_materials = [
                m for m in self._materials 
                if m.base_material_path in allowed_paths
            ]
            
            if not filtered_materials:
                # Fallback to default if no materials match
                print(f"[WARNING] No materials found for {object_name}, using default pool")
                filtered_materials = self._materials
            
            variant = np.random.choice(filtered_materials)
        else:
            # Use default material pool
            variant = np.random.choice(self._materials)
        
        return variant.prim_path
    
    def bind_to_prim(self, stage, prim_path: str, material_prim_path: str) -> bool:
        """Bind a preloaded material to a prim as its visual material.

        Args:
            stage: USD stage.
            prim_path (str): Path of the prim to bind the material to.
            material_prim_path (str): Path of the material to bind.

        Returns:
            bool: True if binding succeeded.
        """
        try:
            from isaaclab.sim.utils import bind_visual_material
            bind_visual_material(prim_path, material_prim_path, stage)
            return True
        except Exception as e:
            print(f"[WARNING] Failed to bind material {material_prim_path} to {prim_path}: {e}")
            return False


def preload_materials(stage, cfg: MaterialRandomizationCfg) -> Optional[MaterialLibrary]:
    """Create a material library and preload it in one call.

    Args:
        stage: USD stage.
        cfg (MaterialRandomizationCfg): Material randomization configuration.

    Returns:
        None or MaterialLibrary: Initialized library, or None if preloading failed.
    """
    library = MaterialLibrary(cfg)
    if library.preload(stage):
        return library
    return None


def bind_random_material(
    stage, 
    prim_path: str, 
    library: MaterialLibrary
) -> bool:
    """Sample a random material from the library and bind it to a prim.

    Args:
        stage: USD stage.
        prim_path (str): Path of the prim to bind the material to.
        library (MaterialLibrary): Material library to sample from.

    Returns:
        bool: True if binding succeeded.
    """
    material_path = library.get_random_material()
    if material_path is None:
        return False
    return library.bind_to_prim(stage, prim_path, material_path)



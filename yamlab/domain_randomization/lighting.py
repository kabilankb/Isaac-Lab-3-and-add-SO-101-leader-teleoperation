"""Scene lighting randomization for IsaacLab environments: dome intensity/color and HDRI textures."""

import os
import numpy as np
from typing import Optional, List, Tuple

from .configs import LightingRandomizationCfg
from yamlab.utils.rendering import color_temperature_to_rgb, list_hdri_files
from yamlab.utils.io import resolve_split_dir


def randomize_dome_light(
    stage,
    dome_light_prim_path: str,
    cfg: LightingRandomizationCfg,
    hdri_files: Optional[List[str]] = None,
) -> bool:
    """Apply random intensity, color, and optional HDRI texture to an existing dome light prim.

    Args:
        stage: USD stage.
        dome_light_prim_path (str): Path to the dome light prim.
        cfg (LightingRandomizationCfg): Lighting randomization configuration.
        hdri_files (None or list[str]): Preloaded HDRI file paths; discovered
            from ``cfg.hdris_path`` if not given.

    Returns:
        bool: True if randomization succeeded.
    """
    if not cfg.enabled:
        return True
    
    try:
        from pxr import UsdLux, Gf
    except ImportError as e:
        print(f"[WARNING] Failed to import pxr: {e}")
        return False
    
    # Get the dome light prim
    prim = stage.GetPrimAtPath(dome_light_prim_path)
    if not prim.IsValid():
        print(f"[WARNING] Dome light prim not found: {dome_light_prim_path}")
        return False
    
    dome_light = UsdLux.DomeLight(prim)
    if not dome_light:
        print(f"[WARNING] Prim is not a DomeLight: {dome_light_prim_path}")
        return False
    
    # Randomize intensity
    if cfg.randomize_dome_intensity:
        intensity = np.random.uniform(*cfg.dome_intensity_range)
        dome_light.GetIntensityAttr().Set(intensity)
    
    # Randomize color via color temperature
    if cfg.randomize_dome_color:
        temperature = np.random.uniform(*cfg.dome_color_temperature_range)
        rgb = color_temperature_to_rgb(temperature)
        dome_light.GetColorAttr().Set(Gf.Vec3f(*rgb))
    
    # Apply HDRI texture
    if cfg.use_hdri_textures:
        if hdri_files is None and cfg.hdris_path:
            hdri_files = list_hdri_files(resolve_split_dir(cfg.hdris_path))
        
        if hdri_files:
            hdri_path = np.random.choice(hdri_files)
            dome_light.GetTextureFileAttr().Set(hdri_path)
            
            # Randomize HDRI rotation
            if cfg.randomize_hdri_rotation:
                rotation = np.random.uniform(*cfg.hdri_rotation_range)
                # Apply rotation via xformOp
                try:
                    from pxr import UsdGeom
                    xform = UsdGeom.Xformable(prim)
                    # Clear existing transforms and add rotation
                    xform.ClearXformOpOrder()
                    rotate_op = xform.AddXformOp(UsdGeom.XformOp.TypeRotateZ)
                    rotate_op.Set(rotation)
                except Exception as e:
                    print(f"[WARNING] Failed to set HDRI rotation: {e}")
    
    return True


class LightingRandomizer:
    """Discovers HDRI files and applies lighting randomization to a scene's dome light.

    Usage:
        randomizer = LightingRandomizer(cfg)
        randomizer.initialize(stage)  # Call once at startup
        randomizer.randomize(stage)   # Call on each reset
    """

    def __init__(self, cfg: LightingRandomizationCfg):
        """Initialize the lighting randomizer.

        Args:
            cfg (LightingRandomizationCfg): Lighting randomization configuration.
        """
        self.cfg = cfg
        self._hdri_files: List[str] = []
        self._initialized = False
        self._dome_light_path = "/World/Light"  # Default path, can be overridden
    
    def set_dome_light_path(self, path: str):
        """Override the dome light prim path to randomize.

        Args:
            path (str): USD prim path of the dome light.
        """
        self._dome_light_path = path
    
    def initialize(self, stage) -> bool:
        """Discover HDRI files and validate configuration; call once at startup.

        Falls back to intensity/color randomization if HDRI textures are
        requested but no files are found.

        Args:
            stage: USD stage.

        Returns:
            bool: True if initialization succeeded.
        """
        if self._initialized:
            return True
        
        if not self.cfg.enabled:
            print("[INFO] Lighting randomization disabled")
            return True
        
        # Discover HDRI files
        if self.cfg.use_hdri_textures and self.cfg.hdris_path:
            self._hdri_files = list_hdri_files(resolve_split_dir(self.cfg.hdris_path))
            if self._hdri_files:
                print(f"[INFO] Found {len(self._hdri_files)} HDRI files")
            else:
                print(f"[WARNING] No HDRI files found in {self.cfg.hdris_path}")
                print("[INFO] Falling back to intensity/color randomization only")
                self.cfg.use_hdri_textures = False
        
        self._initialized = True
        return True
    
    def randomize(self, stage) -> bool:
        """Apply lighting randomization to the dome light; call on each reset.

        Args:
            stage: USD stage.

        Returns:
            bool: True if randomization succeeded.
        """
        if not self.cfg.enabled:
            return True
        
        return randomize_dome_light(
            stage,
            self._dome_light_path,
            self.cfg,
            self._hdri_files if self._hdri_files else None
        )


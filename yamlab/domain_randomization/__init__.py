"""
Domain Randomization Module for IsaacLab Environments.

This module provides visual/appearance and lighting randomization for sim-to-real transfer.
It integrates with IsaacLab's event system and can be enabled/disabled per use case.

Module Structure:
    - configs.py: Configuration dataclasses for randomization settings
    - materials.py: Material preloading and binding utilities
    - lighting.py: Lighting randomization (dome light, HDRI, etc.)
    - events.py: IsaacLab event terms for applying randomization on reset
"""

from .configs import (
    DomainRandomizationCfg,
    MaterialRandomizationCfg,
    LightingRandomizationCfg,
    PerObjectMaterialCfg,
)
from .materials import (
    MaterialLibrary,
    preload_materials,
    bind_random_material,
)
from .lighting import (
    randomize_dome_light,
)
from .events import (
    randomize_object_materials,
    randomize_scene_lighting,
)
from .builder import build_domain_randomization_cfg

__all__ = [
    # Configs
    "DomainRandomizationCfg",
    "MaterialRandomizationCfg",
    "LightingRandomizationCfg",
    "PerObjectMaterialCfg",
    # Materials
    "MaterialLibrary",
    "preload_materials",
    "bind_random_material",
    # Lighting
    "randomize_dome_light",
    # Events
    "randomize_object_materials",
    "randomize_scene_lighting",
    # Builder
    "build_domain_randomization_cfg",
]



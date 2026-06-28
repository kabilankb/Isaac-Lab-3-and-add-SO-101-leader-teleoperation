"""Rendering/asset helpers: color-temperature conversion and HDRI file listing."""

import os
import glob
from typing import List, Tuple

import numpy as np


def color_temperature_to_rgb(temperature: float) -> Tuple[float, float, float]:
    """Convert a color temperature in Kelvin to an RGB tuple in [0, 1].

    Uses the Tanner Helland approximation. Temperature is clamped to [1000, 40000] K.
    """
    temperature = max(1000, min(40000, temperature)) / 100.0

    if temperature <= 66:
        red = 255
    else:
        red = 329.698727446 * ((temperature - 60) ** -0.1332047592)
        red = max(0, min(255, red))

    if temperature <= 66:
        green = 99.4708025861 * np.log(temperature) - 161.1195681661
    else:
        green = 288.1221695283 * ((temperature - 60) ** -0.0755148492)
    green = max(0, min(255, green))

    if temperature >= 66:
        blue = 255
    elif temperature <= 19:
        blue = 0
    else:
        blue = 138.5177312231 * np.log(temperature - 10) - 305.0447927307
        blue = max(0, min(255, blue))

    return (red / 255.0, green / 255.0, blue / 255.0)


def list_hdri_files(folder_path: str) -> List[str]:
    """List all HDR/EXR files under a folder (recursive), .hdr first then .exr.

    Returns [] if the folder doesn't exist.
    """
    if not os.path.exists(folder_path):
        return []
    hdr_files = glob.glob(os.path.join(folder_path, "**", "*.hdr"), recursive=True)
    hdr_files.extend(glob.glob(os.path.join(folder_path, "**", "*.HDR"), recursive=True))
    exr_files = glob.glob(os.path.join(folder_path, "**", "*.exr"), recursive=True)
    exr_files.extend(glob.glob(os.path.join(folder_path, "**", "*.EXR"), recursive=True))
    return sorted(hdr_files) + sorted(exr_files)

"""Canonical set of valid observation modalities and an early-validation helper.

Kept dependency-free (no IsaacLab imports) so entry scripts can validate a
user-supplied ``--observation_modalities`` string BEFORE launching the simulator.
``YamBimanualEnvCfg.configure_observation_modalities`` reads the same constant, so the
allowed modality names are defined in one place.
"""

VALID_OBSERVATION_MODALITIES = ("rgb", "depth", "proprioception", "background_mask")


def validate_observation_modalities(modalities):
    """Assert every entry is a supported observation modality.

    Args:
        modalities: Iterable of modality strings (already split/stripped).

    Returns:
        The modalities as a list.

    Raises:
        AssertionError: If any modality is not in VALID_OBSERVATION_MODALITIES.
    """
    modalities = list(modalities)
    invalid = [m for m in modalities if m not in VALID_OBSERVATION_MODALITIES]
    assert not invalid, (
        f"Invalid observation_modalities {invalid}. "
        f"Valid options: {list(VALID_OBSERVATION_MODALITIES)}"
    )
    return modalities

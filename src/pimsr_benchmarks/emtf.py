"""Compatibility exports for the shared EMTF/profile-coordinate contract.

The implementation lives in :mod:`pimsr_forward.emtf` so inversion and
benchmark code consume one orientation, rotation and masking policy without a
reverse dependency from ``pimsr-inversion`` back into this package.
"""

from pimsr_forward.emtf import (
    FIELD_TO_SI,
    PIMSR_PROFILE_FRAME,
    YELLOWSTONE_PROFILE_IDS,
    YELLOWSTONE_PROFILES,
    MTMode,
    MTStation,
    assemble_profile_modes,
    assemble_station_profile_modes,
    fit_profile_geometry,
    interpolate_profile_field,
    parse_emtf_xml,
    resample_station,
    resample_station_determinant,
    resample_station_modes,
)

__all__ = [
    "FIELD_TO_SI",
    "PIMSR_PROFILE_FRAME",
    "YELLOWSTONE_PROFILES",
    "YELLOWSTONE_PROFILE_IDS",
    "MTMode",
    "MTStation",
    "assemble_profile_modes",
    "assemble_station_profile_modes",
    "fit_profile_geometry",
    "interpolate_profile_field",
    "parse_emtf_xml",
    "resample_station",
    "resample_station_determinant",
    "resample_station_modes",
]

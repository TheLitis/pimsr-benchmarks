"""PIMSR benchmarks: neural inversion vs classical Occam on synthetic and real MT data."""

from .emtf import (
    MTStation,
    assemble_station_profile_modes,
    parse_emtf_xml,
    resample_station,
    resample_station_determinant,
    resample_station_modes,
)
from .hybrid2d import (
    SECTION_NRMS_METRIC_ID,
    assemble_profile,
    assemble_profile_modes,
    multimode_nrms,
    profile_geometry_metadata,
    section_nrms_2d,
)
from .metrics import coverage, data_nrms, profile_rmse, summarize
from .occam1d import OccamResult, default_mesh, occam1d_invert

__all__ = [
    "SECTION_NRMS_METRIC_ID",
    "MTStation",
    "OccamResult",
    "assemble_profile",
    "assemble_profile_modes",
    "assemble_station_profile_modes",
    "coverage",
    "data_nrms",
    "default_mesh",
    "multimode_nrms",
    "occam1d_invert",
    "parse_emtf_xml",
    "profile_geometry_metadata",
    "profile_rmse",
    "resample_station",
    "resample_station_determinant",
    "resample_station_modes",
    "section_nrms_2d",
    "summarize",
]

__version__ = "0.2.0"

"""PIMSR benchmarks: neural inversion vs classical Occam on synthetic and real MT data."""

from .emtf import MTStation, parse_emtf_xml, resample_station
from .metrics import coverage, data_nrms, profile_rmse, summarize
from .occam1d import OccamResult, default_mesh, occam1d_invert

__all__ = [
    "MTStation",
    "parse_emtf_xml",
    "resample_station",
    "coverage",
    "data_nrms",
    "profile_rmse",
    "summarize",
    "OccamResult",
    "default_mesh",
    "occam1d_invert",
]

__version__ = "0.1.0"

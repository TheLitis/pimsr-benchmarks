"""Parser for EMTF XML transfer functions (IRIS/EarthScope SPUD).

USArray MT transfer functions are distributed as EMTF XML
(Kelbert et al., 2019, "The Magnetotelluric Transfer Functions ...").
This module extracts the impedance tensor Z(T) and converts it to the
PIMSR observation vector: Berdichevsky-average (off-diagonal mean) apparent
resistivity and phase, interpolated onto an arbitrary period band.

Only the standard library is used (xml.etree) - no obspy/mtpy dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MU0 = 4.0e-7 * np.pi

__all__ = ["MTStation", "parse_emtf_xml", "resample_station", "resample_station_modes"]


def _fold_phase(deg: np.ndarray) -> np.ndarray:
    """Fold impedance phase into [0, 180) (180-deg periodic convention).

    MT phase normally lives in 0..90; out-of-quadrant values slightly above
    90 (2D/3D distortion) are preserved near 90 instead of wrapping to -90,
    which would be far outside the network's training distribution.
    """
    return np.mod(deg, 180.0)


@dataclass
class MTStation:
    station_id: str
    latitude: float
    longitude: float
    periods: np.ndarray  # (n,) s, ascending
    zxy: np.ndarray  # (n,) complex, Ohm (SI field units converted)
    zyx: np.ndarray  # (n,) complex

    @property
    def rho_a_xy(self) -> np.ndarray:
        return np.abs(self.zxy) ** 2 * self.periods / (2.0 * np.pi * MU0)

    @property
    def rho_a_yx(self) -> np.ndarray:
        return np.abs(self.zyx) ** 2 * self.periods / (2.0 * np.pi * MU0)

    @property
    def phase_xy(self) -> np.ndarray:
        return _fold_phase(np.degrees(np.arctan2(self.zxy.imag, self.zxy.real)))

    @property
    def phase_yx(self) -> np.ndarray:
        """yx phase folded to the first-quadrant convention.

        Standard EMTF yx phase sits in the third quadrant (-180..-90); the
        180-deg fold maps it to 0..90 while being robust to the occasional
        sign-flipped or distorted station (e.g. WYI18, WYJ18).
        """
        return _fold_phase(np.degrees(np.arctan2(self.zyx.imag, self.zyx.real)))

    @property
    def rho_a_det(self) -> np.ndarray:
        """Berdichevsky average: Z_av = (Z_xy - Z_yx) / 2."""
        z_av = 0.5 * (self.zxy - self.zyx)
        return np.abs(z_av) ** 2 * self.periods / (2.0 * np.pi * MU0)

    @property
    def phase_det(self) -> np.ndarray:
        z_av = 0.5 * (self.zxy - self.zyx)
        return np.degrees(np.arctan2(z_av.imag, z_av.real))


#: EMTF XML stores Z in field units (mV/km)/nT; multiply by this for SI Ohm.
FIELD_TO_SI = 4.0 * np.pi * 1.0e-4


def _local(tag: str) -> str:
    """Strip XML namespace."""
    return tag.rsplit("}", 1)[-1]


def parse_emtf_xml(path: str | Path) -> MTStation:
    """Parse one EMTF XML file into an :class:`MTStation`."""
    root = ET.parse(str(path)).getroot()

    station_id, lat, lon = "unknown", float("nan"), float("nan")
    for el in root.iter():
        t = _local(el.tag)
        if t == "Site":
            for c in el:
                tc = _local(c.tag)
                if tc == "Id" and c.text:
                    station_id = c.text.strip()
                elif tc == "Location":
                    for g in c:
                        if _local(g.tag) == "Latitude" and g.text:
                            lat = float(g.text)
                        elif _local(g.tag) == "Longitude" and g.text:
                            lon = float(g.text)
            break

    periods: list[float] = []
    zxy: list[complex] = []
    zyx: list[complex] = []

    for per_el in root.iter():
        if _local(per_el.tag) != "Period":
            continue
        try:
            T = float(per_el.attrib["value"])
        except (KeyError, ValueError):
            continue
        z_el = None
        for c in per_el:
            if _local(c.tag) == "Z":
                z_el = c
                break
        if z_el is None:
            continue
        vals: dict[str, complex] = {}
        for v in z_el:
            if _local(v.tag) != "value":
                continue
            out = v.attrib.get("output", "")
            inp = v.attrib.get("input", "")
            key = f"{out}{inp}".lower().replace("h", "").replace("e", "")
            # keys like "xy", "yx" from output="Ex" input="Hy"
            if v.text:
                parts = v.text.split()
                if len(parts) == 2:
                    vals[key] = complex(float(parts[0]), float(parts[1]))
        if "xy" in vals and "yx" in vals:
            periods.append(T)
            zxy.append(vals["xy"] * FIELD_TO_SI)
            zyx.append(vals["yx"] * FIELD_TO_SI)

    if not periods:
        raise ValueError(f"no impedance data found in {path}")

    order = np.argsort(periods)
    return MTStation(
        station_id=station_id,
        latitude=lat,
        longitude=lon,
        periods=np.asarray(periods)[order],
        zxy=np.asarray(zxy)[order],
        zyx=np.asarray(zyx)[order],
    )


def resample_station(
    st: MTStation, target_periods: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the determinant-average response onto ``target_periods``.

    Returns (log10 rho_a, phase deg, in_band_mask). Points outside the
    station's measured band are masked out rather than extrapolated.
    """
    lp = np.log10(st.periods)
    lt = np.log10(target_periods)
    mask = (lt >= lp.min()) & (lt <= lp.max())
    log_rho = np.interp(lt, lp, np.log10(st.rho_a_det))
    phase = np.interp(lt, lp, st.phase_det)
    return log_rho, phase, mask


def resample_station_modes(
    st: MTStation, target_periods: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-mode interpolation onto ``target_periods``.

    Returns ``{"lr_te", "ph_te", "lr_tm", "ph_tm", "mask"}``. For an E-W
    profile with an assumed N-S geoelectric strike, TE (E along strike)
    maps to Z_yx and TM to Z_xy. Both phases use the 0..90 deg convention.
    """
    lp = np.log10(st.periods)
    lt = np.log10(target_periods)
    mask = (lt >= lp.min()) & (lt <= lp.max())
    return {
        "lr_te": np.interp(lt, lp, np.log10(st.rho_a_yx)),
        "ph_te": np.interp(lt, lp, st.phase_yx),
        "lr_tm": np.interp(lt, lp, np.log10(st.rho_a_xy)),
        "ph_tm": np.interp(lt, lp, st.phase_xy),
        "mask": mask,
    }

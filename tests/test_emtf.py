"""EMTF XML parsing against a minimal synthetic document."""

import numpy as np
import pytest

from pimsr_benchmarks.emtf import (
    FIELD_TO_SI,
    parse_emtf_xml,
    resample_station,
    resample_station_determinant,
    resample_station_modes,
)

XML = r"""<?xml version="1.0" encoding="UTF-8"?>
<EM_TF>
 <Site>
  <Id>TST01</Id>
  <Location>
   <Latitude>44.5</Latitude>
   <Longitude>-110.3</Longitude>
  </Location>
  <Orientation angle_to_geographic_north="0.0">orthogonal</Orientation>
 </Site>
 <ProcessingInfo>
  <SignConvention>exp(+ i\omega t)</SignConvention>
 </ProcessingInfo>
 <Data count="2">
  <Period value="10.0" units="secs">
   <Z type="complex" size="4" units="[mV/km]/[nT]">
    <value output="Ex" input="Hx">0.1 0.05</value>
    <value output="Ex" input="Hy">2.0 1.5</value>
    <value output="Ey" input="Hx">-1.8 -1.2</value>
    <value output="Ey" input="Hy">-0.1 0.02</value>
   </Z>
  </Period>
  <Period value="100.0" units="secs">
   <Z type="complex" size="4" units="[mV/km]/[nT]">
    <value output="Ex" input="Hx">0.02 0.01</value>
    <value output="Ex" input="Hy">0.9 0.7</value>
    <value output="Ey" input="Hx">-0.8 -0.6</value>
    <value output="Ey" input="Hy">-0.02 0.01</value>
   </Z>
  </Period>
 </Data>
</EM_TF>
"""


@pytest.fixture
def xml_file(tmp_path):
    f = tmp_path / "station.xml"
    f.write_text(XML)
    return f


def test_parse_metadata(xml_file):
    st = parse_emtf_xml(xml_file)
    assert st.station_id == "TST01"
    assert st.latitude == pytest.approx(44.5)
    assert st.longitude == pytest.approx(-110.3)
    assert st.orientation_deg == pytest.approx(0.0)
    assert st.sign_convention == "exp(+iomega t)"
    assert st.periods.tolist() == [10.0, 100.0]


def test_impedance_units(xml_file):
    st = parse_emtf_xml(xml_file)
    assert st.zxy[0] == pytest.approx(complex(2.0, 1.5) * FIELD_TO_SI)
    assert np.all(st.rho_a_det > 0)


def test_resample_masks_out_of_band(xml_file):
    st = parse_emtf_xml(xml_file)
    target = np.logspace(-2, 3, 12)  # wider than the 10..100 s band
    log_rho, phase, mask = resample_station(
        st, target, mode="te", profile_azimuth_deg=90.0
    )
    assert mask.sum() < target.size
    assert np.isfinite(log_rho[mask]).all()
    assert np.isnan(log_rho[~mask]).all()
    assert np.isnan(phase[~mask]).all()
    in_band = (target >= 10.0) & (target <= 100.0)
    assert (mask == in_band).all()


def test_eastward_profile_rotates_geographic_components_to_local_modes(xml_file):
    st = parse_emtf_xml(xml_file)
    lr_te, ph_te, mask_te = resample_station(
        st, st.periods, mode="te", profile_azimuth_deg=90.0
    )
    lr_tm, ph_tm, mask_tm = resample_station(
        st, st.periods, mode="tm", profile_azimuth_deg=90.0
    )

    assert np.all(mask_te & mask_tm)
    assert np.allclose(lr_te, np.log10(st.rho_a_xy))  # local TE = geographic Zxy
    assert np.allclose(ph_te, st.phase_xy)
    assert np.allclose(lr_tm, np.log10(st.rho_a_yx))  # local TM = geographic Zyx
    assert np.allclose(ph_tm, st.phase_yx)
    assert not np.allclose(lr_te, lr_tm)


def test_non_cardinal_rotation_uses_full_tensor(xml_file):
    st = parse_emtf_xml(xml_file)
    azimuth = 31.0
    angle = np.radians(azimuth)
    c, s = np.cos(angle), np.sin(angle)
    transform = np.array([[c, s], [s, -c]])
    expected = transform @ st.impedance @ transform.T
    assert np.allclose(st.impedance_in_profile(azimuth), expected)
    assert np.allclose(st.mode_impedance("te", azimuth), expected[:, 1, 0])
    assert np.allclose(st.mode_impedance("tm", azimuth), expected[:, 0, 1])


def test_modes_return_explicit_independent_masks(xml_file):
    st = parse_emtf_xml(xml_file)
    target = np.array([1.0, 10.0, 100.0, 1000.0])
    modes = resample_station_modes(st, target, profile_azimuth_deg=90.0)

    expected = np.array([False, True, True, False])
    assert np.array_equal(modes["mask_te"], expected)
    assert np.array_equal(modes["mask_tm"], expected)
    assert np.isnan(modes["lr_te"][~expected]).all()
    assert np.isnan(modes["lr_tm"][~expected]).all()
    assert "mask" not in modes


def test_determinant_is_only_available_explicitly(xml_file):
    st = parse_emtf_xml(xml_file)
    with pytest.raises(TypeError):
        resample_station(st, st.periods)

    lr, phase, mask = resample_station_determinant(st, st.periods)
    assert np.all(mask)
    assert np.allclose(lr, np.log10(st.rho_a_det))
    assert np.allclose(phase, st.phase_det)

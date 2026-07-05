"""EMTF XML parsing against a minimal synthetic document."""

import numpy as np
import pytest

from pimsr_benchmarks.emtf import FIELD_TO_SI, parse_emtf_xml, resample_station

XML = """<?xml version="1.0" encoding="UTF-8"?>
<EM_TF>
 <Site>
  <Id>TST01</Id>
  <Location>
   <Latitude>44.5</Latitude>
   <Longitude>-110.3</Longitude>
  </Location>
 </Site>
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
    assert st.periods.tolist() == [10.0, 100.0]


def test_impedance_units(xml_file):
    st = parse_emtf_xml(xml_file)
    assert st.zxy[0] == pytest.approx(complex(2.0, 1.5) * FIELD_TO_SI)
    assert np.all(st.rho_a_det > 0)


def test_resample_masks_out_of_band(xml_file):
    st = parse_emtf_xml(xml_file)
    target = np.logspace(-2, 3, 12)  # wider than the 10..100 s band
    log_rho, phase, mask = resample_station(st, target)
    assert mask.sum() < target.size
    assert np.isfinite(log_rho[mask]).all()
    in_band = (target >= 10.0) & (target <= 100.0)
    assert (mask == in_band).all()

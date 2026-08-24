"""Contracts for immutable solver-input and EMTF download exports."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import scripts.export_modem_data as modem_export
from scripts import fetch_usarray


def _station(station_id: str, latitude: float, longitude: float):
    return SimpleNamespace(
        station_id=station_id,
        latitude=latitude,
        longitude=longitude,
        periods=np.array([1.0, 10.0]),
        zxy=np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        zyx=np.array([2.0 + 1.0j, 4.0 + 3.0j]),
    )


def test_modem_renderer_is_deterministic_and_validates_configuration():
    stations = {
        "A": _station("A", 44.0, -111.0),
        "B": _station("B", 44.1, -110.9),
    }
    first, metadata = modem_export.render_profile(stations, ["A", "B"], 3)
    second, repeated = modem_export.render_profile(stations, ["A", "B"], 3)

    assert first == second
    assert metadata == repeated
    assert metadata["rows"] == 12
    assert "> 3 2" in first
    with pytest.raises(ValueError, match="at least two"):
        modem_export.render_profile(stations, ["A", "B"], 1)
    with pytest.raises(ValueError, match="missing"):
        modem_export.render_profile(stations, ["A", "MISSING"], 3)


def test_fetcher_validates_and_publishes_complete_xml_without_overwrite(tmp_path):
    source = (
        fetch_usarray.Path(__file__).resolve().parents[1]
        / "data"
        / "emtf"
        / "USArray_IDI15_2008.xml"
    )
    payload = source.read_bytes()
    destination = tmp_path / source.name

    published = fetch_usarray._publish_emtf_no_overwrite(
        payload,
        destination,
        expected_product="USArray.IDI15.2008",
    )
    assert published.read_bytes() == payload
    fetch_usarray._validate_existing_emtf(
        published,
        expected_product="USArray.IDI15.2008",
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fetch_usarray._publish_emtf_no_overwrite(
            payload,
            destination,
            expected_product="USArray.IDI15.2008",
        )
    assert not list(tmp_path.glob("*.part"))


def test_fetcher_rejects_partial_or_mismatched_xml(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_usarray, "parse_emtf_xml", lambda _path: object())
    destination = tmp_path / "station.xml"
    with pytest.raises(ValueError, match="well-formed"):
        fetch_usarray._publish_emtf_no_overwrite(
            b"<EM_TF><ProductId>USArray.BAD",
            destination,
            expected_product="USArray.BAD",
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))

    complete_but_wrong = (
        b"<EM_TF><ProductId>USArray.OTHER</ProductId><Site/><Data>"
        b"<Period/><Z/></Data></EM_TF>"
    )
    with pytest.raises(ValueError, match="ProductId"):
        fetch_usarray._publish_emtf_no_overwrite(
            complete_but_wrong,
            destination,
            expected_product="USArray.BAD",
        )

"""Strict observation/truth separation for the PIMSR 2D SOTA campaign."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from pimsr_benchmarks import dataset2d_materialization as materialization
from pimsr_benchmarks.dataset2d_materialization import materialize_dataset2d


def _source_h5(path: Path) -> Path:
    from pimsr_forward.dataset2d import (
        _DEFAULT_SENSOR_PARAMETERS_JSON,
        _write_contract_attrs,
        _write_dataset_attrs,
    )

    n, n_frequency, n_station, n_depth, n_x = 3, 2, 2, 2, 3
    observation_shape = (n, n_frequency, n_station)
    source_values = np.arange(np.prod(observation_shape), dtype=np.float32).reshape(
        observation_shape
    )
    software_versions = json.dumps(
        {
            "discretize": "test",
            "h5py": "test",
            "numpy": "test",
            "pimsr_forward": "test",
            "pimsr_geogen": "test",
            "simpeg": "test",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with h5py.File(path, "x") as h5:
        _write_contract_attrs(
            h5,
            generator_seed=41,
            generation_start_index=10,
            expected_row_count=n,
            source_shard_count=1,
            generation_complete=True,
            sensor_parameters_json=_DEFAULT_SENSOR_PARAMETERS_JSON,
            software_versions_json=software_versions,
        )
        h5.create_dataset("obs_mt_log10_rho", data=1.0 + source_values / 100.0)
        h5.create_dataset("obs_mt_phase", data=20.0 + source_values)
        h5.create_dataset("clean_mt_log10_rho", data=1.1 + source_values / 100.0)
        h5.create_dataset("clean_mt_phase", data=21.0 + source_values)
        h5.create_dataset("obs_mt_log10_rho_tm", data=1.5 + source_values / 100.0)
        h5.create_dataset("obs_mt_phase_tm", data=40.0 + source_values)
        h5.create_dataset("clean_mt_log10_rho_tm", data=1.6 + source_values / 100.0)
        h5.create_dataset("clean_mt_phase_tm", data=41.0 + source_values)
        h5.create_dataset(
            "target_log10_res",
            data=np.arange(n * n_depth * n_x, dtype=np.float32).reshape(n, n_depth, n_x),
        )
        h5.create_dataset("scenario", data=np.asarray([0, 1, 4], dtype=np.int32))
        h5.create_dataset("has_fault", data=np.asarray([0, 1, 0], dtype=np.uint8))
        h5.create_dataset("sample_index", data=np.asarray([10, 11, 12], dtype=np.int64))
        h5.create_dataset("frequencies", data=np.asarray([0.1, 10.0], dtype=np.float64))
        h5.create_dataset("station_x", data=np.asarray([0.5, 1.5], dtype=np.float64))
        h5.create_dataset("x_grid", data=np.asarray([0.0, 1.0, 2.0], dtype=np.float64))
        h5.create_dataset("depth_grid", data=np.asarray([1.0, 3.0], dtype=np.float64))
        _write_dataset_attrs(h5)
    return path


@pytest.fixture
def source_h5(tmp_path: Path) -> Path:
    return _source_h5(tmp_path / "source.h5")


_SECRET_KEY = b"pimsr-materializer-unit-test-secret-key-v1"


def _destinations(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "observations.npz",
        root / "truth.npz",
        root / "observations.public.json",
        root / "scoring.operator.json",
    )


def _materialize(source: Path, root: Path):
    observations, truth, public_manifest, operator_manifest = _destinations(root)
    result = materialize_dataset2d(
        source,
        observations,
        truth,
        public_manifest,
        operator_manifest,
        split_id="hidden-test",
        sample_id_key=_SECRET_KEY,
        rho_log10_floor=0.075,
        phase_degree_floor=3.25,
    )
    return result, observations, truth, public_manifest, operator_manifest


def _npy_sha256(array: np.ndarray) -> str:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _expected_opaque(source_index: int, *, split_id: str = "hidden-test") -> int:
    split = split_id.encode("ascii")
    message = (
        b"pimsr-sota-2d-opaque-sample-id-v1\x00"
        + (41).to_bytes(8, "big")
        + source_index.to_bytes(8, "big")
        + len(split).to_bytes(4, "big")
        + split
    )
    return int.from_bytes(hmac.digest(_SECRET_KEY, message, "sha256")[:8], "big") & (
        2**63 - 1
    )


def _canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert (
        raw
        == (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )
    return value


def test_materialization_separates_observations_and_truth(
    source_h5: Path, tmp_path: Path
):
    (
        result,
        observations_path,
        truth_path,
        public_manifest_path,
        operator_manifest_path,
    ) = _materialize(source_h5, tmp_path / "published")
    expected_ids = np.asarray(
        [_expected_opaque(index) for index in (10, 11, 12)], dtype="<i8"
    )

    observation_keys = {
        "schema",
        "schema_version",
        "sample_index",
        "frequency_hz",
        "station_x_m",
        "x_cell_centers_m",
        "depth_cell_centers_m",
        "observation_channel_order",
        "observed_log10_rho_te",
        "observed_phase_te_degrees",
        "observed_log10_rho_tm",
        "observed_phase_tm_degrees",
        "declared_evaluation_floor_log10_rho_te",
        "declared_evaluation_floor_phase_te_degrees",
        "declared_evaluation_floor_log10_rho_tm",
        "declared_evaluation_floor_phase_tm_degrees",
        "valid_mask",
    }
    with np.load(observations_path, allow_pickle=False) as observations:
        assert set(observations.files) == observation_keys
        assert observations["schema"].item() == "pimsr-sota-2d-observations"
        assert observations["schema_version"].dtype == np.dtype("<i8")
        assert observations["schema_version"].item() == 1
        assert observations["sample_index"].dtype == np.dtype("<i8")
        np.testing.assert_array_equal(observations["sample_index"], expected_ids)
        assert len(np.unique(observations["sample_index"])) == 3
        assert np.all(observations["sample_index"] >= 0)
        assert not np.array_equal(observations["sample_index"], [10, 11, 12])
        assert observations["observed_log10_rho_te"].dtype == np.dtype("<f4")
        assert observations["observed_log10_rho_te"].shape == (3, 2, 2)
        assert observations["frequency_hz"].dtype == np.dtype("<f8")
        assert observations["station_x_m"].dtype == np.dtype("<f8")
        np.testing.assert_array_equal(
            observations["observation_channel_order"],
            materialization.OBSERVATION_CHANNEL_ORDER,
        )
        assert observations["valid_mask"].dtype == np.dtype(bool)
        assert observations["valid_mask"].shape == (3, 4, 2, 2)
        assert observations["valid_mask"].all()
        for mode in ("te", "tm"):
            np.testing.assert_array_equal(
                observations[f"declared_evaluation_floor_log10_rho_{mode}"],
                np.full((3, 2, 2), np.float32(0.075)),
            )
            np.testing.assert_array_equal(
                observations[f"declared_evaluation_floor_phase_{mode}_degrees"],
                np.full((3, 2, 2), np.float32(3.25)),
            )
        assert not any(
            "clean" in key or "truth" in key or "target" in key
            for key in observations.files
        )
        assert "scenario" not in observations.files
        assert "has_fault" not in observations.files

    truth_keys = {
        "schema",
        "schema_version",
        "sample_index",
        "observations_sha256",
        "scenario",
        "has_fault",
        "x_cell_centers_m",
        "depth_cell_centers_m",
        "truth_log10_resistivity",
    }
    with np.load(truth_path, allow_pickle=False) as truth:
        assert set(truth.files) == truth_keys
        assert truth["schema"].item() == "pimsr-sota-2d-truth"
        assert truth["schema_version"].item() == 2
        np.testing.assert_array_equal(truth["sample_index"], expected_ids)
        assert truth["observations_sha256"].item() == hashlib.sha256(
            observations_path.read_bytes()
        ).hexdigest()
        np.testing.assert_array_equal(
            truth["scenario"], ["background", "aquifer", "geothermal"]
        )
        np.testing.assert_array_equal(truth["has_fault"], [False, True, False])
        assert truth["truth_log10_resistivity"].dtype == np.dtype("<f4")
        assert truth["truth_log10_resistivity"].shape == (3, 2, 3)
        np.testing.assert_array_equal(truth["x_cell_centers_m"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(truth["depth_cell_centers_m"], [1.0, 3.0])

    public = _canonical_json(public_manifest_path)
    assert set(public) == {
        "audience",
        "declared_evaluation_floors",
        "observation_payload",
        "physical_contract",
        "sample_count",
        "schema",
        "schema_version",
        "split_id",
    }
    assert public["audience"] == "method_input_public"
    assert public["schema_version"] == 2
    assert public["split_id"] == "hidden-test"
    assert public["sample_count"] == 3
    assert public["physical_contract"]["mode_component_mapping"] == {
        "TE": "Zyx",
        "TM": "Zxy",
    }
    assert public["declared_evaluation_floors"]["policy_id"] == (
        "declared_evaluation_floors_log10_rho_phase_v1"
    )
    assert (
        public["observation_payload"]["sha256"]
        == hashlib.sha256(observations_path.read_bytes()).hexdigest()
    )
    with np.load(observations_path, allow_pickle=False) as observations:
        record = public["observation_payload"]["arrays"]["valid_mask"]
        assert record == {
            "axis_order": ["sample", "channel", "frequency", "station"],
            "dtype": "|b1",
            "shape": [3, 4, 2, 2],
            "sha256": _npy_sha256(observations["valid_mask"]),
        }

    operator = _canonical_json(operator_manifest_path)
    assert set(operator) == {
        "artifacts",
        "audience",
        "schema",
        "schema_version",
        "source",
        "split",
    }
    assert operator["audience"] == "benchmark_operator_only"
    assert operator["schema_version"] == 2
    assert operator["source"]["generator_seed"] == 41
    assert operator["source"]["generator_rng"]
    assert (
        operator["source"]["sha256"] == hashlib.sha256(source_h5.read_bytes()).hexdigest()
    )
    assert operator["source"]["size_bytes"] == source_h5.stat().st_size
    assert (
        operator["artifacts"]["observations"]["sha256"]
        == hashlib.sha256(observations_path.read_bytes()).hexdigest()
    )
    assert (
        operator["artifacts"]["withheld_truth"]["sha256"]
        == hashlib.sha256(truth_path.read_bytes()).hexdigest()
    )
    assert operator["artifacts"]["withheld_truth"]["schema_version"] == 2
    assert operator["artifacts"]["public_observation_manifest"]["sha256"] == (
        hashlib.sha256(public_manifest_path.read_bytes()).hexdigest()
    )
    assert operator["split"]["sample_id_mapping"] == [
        {
            "opaque_sample_index": int(opaque),
            "source_generator_sample_index": source,
        }
        for source, opaque in zip((10, 11, 12), expected_ids, strict=True)
    ]
    assert len(operator["split"]["groups"]) == 3
    assert operator["split"]["scenario_groups"] == [
        {
            "opaque_sample_indices": [int(expected_ids[0])],
            "scenario": "background",
            "scenario_index": 0,
        },
        {
            "opaque_sample_indices": [int(expected_ids[1])],
            "scenario": "aquifer",
            "scenario_index": 1,
        },
        {
            "opaque_sample_indices": [int(expected_ids[2])],
            "scenario": "geothermal",
            "scenario_index": 4,
        },
    ]
    assert result.observations_sha256 == (operator["artifacts"]["observations"]["sha256"])
    assert result.truth_sha256 == operator["artifacts"]["withheld_truth"]["sha256"]
    assert (
        result.public_manifest_sha256
        == hashlib.sha256(public_manifest_path.read_bytes()).hexdigest()
    )
    assert (
        result.operator_manifest_sha256
        == hashlib.sha256(operator_manifest_path.read_bytes()).hexdigest()
    )

    public_bytes = public_manifest_path.read_bytes()
    truth_digest = hashlib.sha256(truth_path.read_bytes()).hexdigest().encode()
    source_digest = hashlib.sha256(source_h5.read_bytes()).hexdigest().encode()
    assert truth_digest not in public_bytes
    assert source_digest not in public_bytes
    for forbidden in (
        b"generator_seed",
        b"generator_rng",
        b"source_generator_sample_index",
        b"scenario",
        b"has_fault",
        b"family_id",
        b"base_model_id",
        b"noise_id",
    ):
        assert forbidden not in public_bytes
    all_output_bytes = b"".join(
        path.read_bytes()
        for path in (
            observations_path,
            truth_path,
            public_manifest_path,
            operator_manifest_path,
        )
    )
    assert _SECRET_KEY not in all_output_bytes
    assert hashlib.sha256(_SECRET_KEY).hexdigest().encode() not in all_output_bytes


def test_repeated_materialization_is_byte_identical(source_h5: Path, tmp_path: Path):
    first = _materialize(source_h5, tmp_path / "first")[1:]
    second = _materialize(source_h5, tmp_path / "second")[1:]
    for left, right in zip(first, second, strict=True):
        assert left.name == right.name
        assert left.read_bytes() == right.read_bytes()


def test_secret_or_split_change_changes_opaque_ids(source_h5: Path, tmp_path: Path):
    first = _materialize(source_h5, tmp_path / "first")[1]
    second_outputs = _destinations(tmp_path / "second")
    materialize_dataset2d(
        source_h5,
        *second_outputs,
        split_id="another-hidden-test",
        sample_id_key=b"another-independent-sample-id-secret-key",
    )
    with (
        np.load(first, allow_pickle=False) as left,
        np.load(second_outputs[0], allow_pickle=False) as right,
    ):
        assert not np.array_equal(left["sample_index"], right["sample_index"])


def test_key_file_is_supported_without_persisting_secret(source_h5: Path, tmp_path: Path):
    key_path = tmp_path / "sample-id.key"
    key_path.write_bytes(_SECRET_KEY)
    outputs = _destinations(tmp_path / "key-file-output")
    materialize_dataset2d(
        source_h5,
        *outputs,
        split_id="hidden-test",
        sample_id_key=key_path,
    )
    assert all(_SECRET_KEY not in output.read_bytes() for output in outputs)


def test_existing_destination_prevents_any_publication(source_h5: Path, tmp_path: Path):
    outputs = _destinations(tmp_path / "existing")
    outputs[1].parent.mkdir(parents=True)
    outputs[1].write_bytes(b"do not replace")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_dataset2d(
            source_h5,
            *outputs,
            split_id="test",
            sample_id_key=_SECRET_KEY,
        )
    assert not outputs[0].exists()
    assert outputs[1].read_bytes() == b"do not replace"
    assert not outputs[2].exists()
    assert not outputs[3].exists()
    assert not list(outputs[1].parent.glob("*.part"))


def test_stale_partial_prevents_any_publication(source_h5: Path, tmp_path: Path):
    outputs = _destinations(tmp_path / "stale")
    outputs[0].parent.mkdir(parents=True)
    partial = outputs[0].with_name(outputs[0].name + ".part")
    partial.write_bytes(b"possibly active writer")
    with pytest.raises(FileExistsError, match="stale partial"):
        materialize_dataset2d(
            source_h5,
            *outputs,
            split_id="test",
            sample_id_key=_SECRET_KEY,
        )
    assert partial.read_bytes() == b"possibly active writer"
    assert not any(output.exists() for output in outputs)


def test_outputs_must_share_one_parent(source_h5: Path, tmp_path: Path):
    observations, truth, public, operator = _destinations(tmp_path / "one")
    with pytest.raises(ValueError, match="share one parent"):
        materialize_dataset2d(
            source_h5,
            observations,
            truth,
            public,
            tmp_path / "two" / operator.name,
            split_id="test",
            sample_id_key=_SECRET_KEY,
        )
    assert not observations.exists()


def test_operator_manifest_is_published_last(
    source_h5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outputs = _destinations(tmp_path / "ordered")
    real_link = os.link
    linked: list[str] = []

    def record_link(source: Path, destination: Path) -> None:
        linked.append(Path(destination).name)
        real_link(source, destination)

    monkeypatch.setattr(materialization.os, "link", record_link)
    materialize_dataset2d(
        source_h5,
        *outputs,
        split_id="test",
        sample_id_key=_SECRET_KEY,
    )
    assert linked == [output.name for output in outputs]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), OSError("injected failure")])
def test_base_exception_rolls_back_all_outputs(
    source_h5: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
):
    outputs = _destinations(tmp_path / f"rollback-{type(interrupt).__name__}")
    real_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interrupt
        real_link(source, destination)

    monkeypatch.setattr(materialization.os, "link", fail_second_link)
    with pytest.raises(type(interrupt)):
        materialize_dataset2d(
            source_h5,
            *outputs,
            split_id="test",
            sample_id_key=_SECRET_KEY,
        )
    assert not any(output.exists() for output in outputs)
    assert not list(outputs[0].parent.glob("*.part"))


def test_publication_race_preserves_foreign_file(
    source_h5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outputs = _destinations(tmp_path / "race")
    real_link = os.link

    def race_first_link(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"foreign racer")
        real_link(source, destination)

    monkeypatch.setattr(materialization.os, "link", race_first_link)
    with pytest.raises(FileExistsError, match="publication race"):
        materialize_dataset2d(
            source_h5,
            *outputs,
            split_id="test",
            sample_id_key=_SECRET_KEY,
        )
    assert outputs[0].read_bytes() == b"foreign racer"
    assert not any(output.exists() for output in outputs[1:])


def test_rollback_refuses_replaced_file(
    source_h5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outputs = _destinations(tmp_path / "replace-race")
    real_link = os.link
    calls = 0

    def replace_then_fail(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            outputs[0].unlink()
            outputs[0].write_bytes(b"replacement from racer")
            raise OSError("trigger rollback")
        real_link(source, destination)

    monkeypatch.setattr(materialization.os, "link", replace_then_fail)
    with pytest.raises(materialization.MaterializationError, match="replaced"):
        materialize_dataset2d(
            source_h5,
            *outputs,
            split_id="test",
            sample_id_key=_SECRET_KEY,
        )
    assert outputs[0].read_bytes() == b"replacement from racer"
    assert not any(output.exists() for output in outputs[1:])


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("schema", "schema"),
        ("components", "impedance"),
        ("nonfinite", "non-finite"),
        ("sample_order", "sample_index"),
        ("phase", r"\[0, 180\)"),
    ),
)
def test_invalid_source_fails_without_output(
    source_h5: Path,
    tmp_path: Path,
    mutation: str,
    message: str,
):
    source = tmp_path / f"bad-{mutation}.h5"
    shutil.copyfile(source_h5, source)
    with h5py.File(source, "r+") as h5:
        if mutation == "schema":
            h5.attrs["schema_version"] = np.int64(1)
        elif mutation == "components":
            h5.attrs["impedance_components"] = np.asarray(["Zxy", "Zyx"], dtype="S3")
        elif mutation == "nonfinite":
            h5["obs_mt_log10_rho"][0, 0, 0] = np.nan
        elif mutation == "sample_order":
            h5["sample_index"][:] = [10, 12, 11]
        elif mutation == "phase":
            h5["obs_mt_phase_tm"][0, 0, 0] = 180.0
        else:  # pragma: no cover - guarded by parametrization
            raise AssertionError(mutation)
    outputs = _destinations(tmp_path / f"out-{mutation}")
    with pytest.raises(ValueError, match=message):
        materialize_dataset2d(
            source,
            *outputs,
            split_id="test",
            sample_id_key=_SECRET_KEY,
        )
    assert not any(output.exists() for output in outputs)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"split_id": "Hidden Test"}, ValueError),
        ({"sample_id_key": b"too short"}, ValueError),
        ({"sample_id_key": object()}, TypeError),
        ({"rho_log10_floor": 0.0}, ValueError),
        ({"rho_log10_floor": np.nan}, ValueError),
        ({"phase_degree_floor": np.inf}, ValueError),
    ),
)
def test_invalid_policy_arguments_fail_before_publication(
    source_h5: Path,
    tmp_path: Path,
    kwargs: dict[str, object],
    error: type[Exception],
):
    outputs = _destinations(tmp_path / "invalid-options")
    options: dict[str, object] = {
        "split_id": "test",
        "sample_id_key": _SECRET_KEY,
        "rho_log10_floor": 0.05,
        "phase_degree_floor": 2.9,
    }
    options.update(kwargs)
    with pytest.raises(error):
        materialize_dataset2d(source_h5, *outputs, **options)
    assert not any(output.exists() for output in outputs)

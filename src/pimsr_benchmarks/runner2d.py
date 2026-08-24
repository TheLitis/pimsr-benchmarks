"""Fail-closed loading and observation preparation for 2D benchmark runners."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "LoadedModel2D",
    "checkpoint_adaptation_kind",
    "file_artifact_provenance",
    "interpolate_periods_in_band",
    "load_dataset2d",
    "load_model2d",
    "prepare_empty_workdir",
    "prepare_profile_observation",
    "publish_json_no_overwrite",
    "publish_npz_no_overwrite",
    "publish_text_no_overwrite",
    "require_file_artifact_unchanged",
    "require_finetune2d_lineage",
    "run_checked",
    "stack_dataset_observations",
]


@dataclass(frozen=True)
class LoadedModel2D:
    """A checkpoint validated against the exact benchmark dataset geometry."""

    model: Any
    checkpoint: Mapping[str, Any]
    contract: Any
    checkpoint_path: Path
    dataset_path: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    dataset_sha256: str
    dataset_size_bytes: int

    def artifact_provenance(self) -> dict[str, dict[str, object]]:
        """Return JSON-safe identities for the exact artifacts used."""
        return {
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
                "size_bytes": self.checkpoint_size_bytes,
            },
            "dataset": {
                "path": str(self.dataset_path),
                "sha256": self.dataset_sha256,
                "size_bytes": self.dataset_size_bytes,
            },
        }

    def require_artifacts_unchanged(self) -> None:
        """Fail if either path no longer contains the bytes used at load time."""
        for kind, path, expected_digest, expected_size in (
            (
                "2D checkpoint",
                self.checkpoint_path,
                self.checkpoint_sha256,
                self.checkpoint_size_bytes,
            ),
            (
                "2D dataset",
                self.dataset_path,
                self.dataset_sha256,
                self.dataset_size_bytes,
            ),
        ):
            digest, size = _stable_file_digest(path)
            if digest != expected_digest or size != expected_size:
                raise RuntimeError(f"{kind} changed after it was loaded: {path}")


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return (int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns))


def _stable_file_bytes(path: Path, *, kind: str) -> tuple[bytes, str, int]:
    before = _stat_signature(path)
    payload = path.read_bytes()
    after = _stat_signature(path)
    if before != after or len(payload) != before[2]:
        raise RuntimeError(f"{kind} changed while it was being snapshotted: {path}")
    return payload, hashlib.sha256(payload).hexdigest(), len(payload)


def _stable_file_digest(
    path: Path,
    *,
    expected_signature: tuple[int, int, int, int] | None = None,
) -> tuple[str, int]:
    before = _stat_signature(path)
    if expected_signature is not None and before != expected_signature:
        raise RuntimeError(f"2D dataset changed after contract validation: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = _stat_signature(path)
    if before != after:
        raise RuntimeError(f"2D dataset changed while it was being hashed: {path}")
    return digest.hexdigest(), before[2]


def file_artifact_provenance(path: str | Path) -> dict[str, object]:
    """Return a stable SHA-256 identity for one exact input artifact."""
    artifact = Path(path).resolve(strict=True)
    digest, size = _stable_file_digest(artifact)
    return {"path": str(artifact), "sha256": digest, "size_bytes": size}


def require_file_artifact_unchanged(
    provenance: Mapping[str, object],
    *,
    role: str,
) -> None:
    """Reject mutation or replacement of an already identified input file."""
    if set(provenance) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{role} provenance is incomplete")
    path = Path(str(provenance["path"]))
    digest, size = _stable_file_digest(path)
    if digest != provenance["sha256"] or size != provenance["size_bytes"]:
        raise RuntimeError(f"{role} changed after it was loaded: {path}")


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_adaptation_kind(checkpoint: Mapping[str, Any]) -> str:
    """Derive adaptation state from checkpoint lineage, never from a label."""
    metadata = checkpoint.get("finetune2d")
    if metadata is None:
        return "zero-shot"
    if not isinstance(metadata, Mapping):
        raise TypeError("2D checkpoint finetune2d metadata must be a mapping")
    lineage = metadata.get("input_lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("fine-tuned 2D checkpoint lacks strict input lineage")
    observations = lineage.get("observations")
    profiles = observations.get("profiles") if isinstance(observations, Mapping) else None
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("fine-tuned 2D checkpoint lacks adapted profile identities")
    return "profile-adapted" if len(profiles) == 1 else "regional/joint-adapted"


def require_finetune2d_lineage(
    adapted: LoadedModel2D,
    *,
    base: LoadedModel2D,
    emtf_dir: str | Path,
    expected_profiles: Sequence[Sequence[str]],
    expected_options: Mapping[str, object] | None = None,
) -> Mapping[str, Any]:
    """Prove that an adapted checkpoint was made for the declared run."""
    metadata = adapted.checkpoint.get("finetune2d")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("finetune_schema") != "pimsr-finetune-2d"
        or metadata.get("finetune_schema_version") != 1
    ):
        raise ValueError("adapted checkpoint lacks pimsr-finetune-2d schema v1")
    lineage = metadata.get("input_lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("adapted checkpoint lacks input_lineage")
    lineage_base = {key: value for key, value in lineage.items() if key != "lineage_sha256"}
    if set(lineage) != set(lineage_base) | {"lineage_sha256"} or lineage.get(
        "lineage_sha256"
    ) != _canonical_json_sha256(lineage_base):
        raise ValueError("adapted checkpoint input lineage digest is invalid")

    for role, identity, digest, size in (
        (
            "base checkpoint",
            lineage.get("checkpoint"),
            base.checkpoint_sha256,
            base.checkpoint_size_bytes,
        ),
        (
            "geometry dataset",
            lineage.get("data_h5"),
            base.dataset_sha256,
            base.dataset_size_bytes,
        ),
    ):
        if (
            not isinstance(identity, Mapping)
            or identity.get("artifact_sha256") != digest
            or identity.get("artifact_size_bytes") != size
        ):
            raise ValueError(f"adapted checkpoint {role} lineage does not match this run")

    observations = lineage.get("observations")
    profile_entries = observations.get("profiles") if isinstance(observations, Mapping) else None
    actual_profiles = (
        [entry.get("profile_ids") for entry in profile_entries]
        if isinstance(profile_entries, list)
        and all(isinstance(entry, Mapping) for entry in profile_entries)
        else None
    )
    requested_profiles = [list(profile) for profile in expected_profiles]
    if actual_profiles != requested_profiles:
        raise ValueError("adapted checkpoint profile lineage does not match this run")

    options = lineage.get("training_options")
    if not isinstance(options, Mapping):
        raise TypeError("adapted checkpoint lacks training option lineage")
    for key, expected in (expected_options or {}).items():
        if options.get(key) != expected:
            raise ValueError(f"adapted checkpoint option {key!r} does not match this run")

    emtf = lineage.get("emtf_sources")
    expected_files = emtf.get("files") if isinstance(emtf, Mapping) else None
    if not isinstance(expected_files, list):
        raise TypeError("adapted checkpoint lacks EMTF source lineage")
    directory = Path(emtf_dir).resolve(strict=True)
    current_paths = sorted(directory.glob("*.xml"), key=lambda path: path.name.casefold())
    expected_names = [
        entry.get("relative_path") if isinstance(entry, Mapping) else None
        for entry in expected_files
    ]
    if expected_names != [path.name for path in current_paths]:
        raise ValueError("adapted checkpoint EMTF file set does not match this run")
    for entry, path in zip(expected_files, current_paths, strict=True):
        current = file_artifact_provenance(path)
        if (
            entry.get("artifact_sha256") != current["sha256"]
            or entry.get("artifact_size_bytes") != current["size_bytes"]
        ):
            raise ValueError(f"adapted checkpoint EMTF lineage changed for {path.name}")
    return lineage


def _publish_bytes_no_overwrite(payload: bytes, path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(f".{output.name}.{uuid.uuid4().hex}.part")
    try:
        with part.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(part, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing benchmark output: {output}"
            ) from error
        part.unlink()
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return output


def publish_json_no_overwrite(value: object, path: str | Path) -> Path:
    """Atomically publish canonical JSON without replacing an existing result."""
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _publish_bytes_no_overwrite(payload, path)


def publish_npz_no_overwrite(path: str | Path, **arrays: object) -> Path:
    """Publish a compressed NumPy bundle without replacing an existing result."""
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    return _publish_bytes_no_overwrite(payload.getvalue(), path)


def publish_text_no_overwrite(text: str, path: str | Path) -> Path:
    """Atomically publish UTF-8 text without replacing an existing result."""
    return _publish_bytes_no_overwrite(text.encode("utf-8"), path)


def _validated_ranges(value: Any, *, where: str) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of inclusive ranges")
    result: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(
                isinstance(bound, bool) or not isinstance(bound, int) for bound in item
            )
            or item[0] < 0
            or item[1] < item[0]
        ):
            raise ValueError(f"{where} contains an invalid inclusive range")
        current = (item[0], item[1])
        if result and current[0] <= result[-1][1]:
            raise ValueError(f"{where} ranges must be sorted and disjoint")
        result.append(current)
    return result


def _ranges_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> bool:
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        if max(left_start, right_start) <= min(left_end, right_end):
            return True
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return False


def _assert_heldout_disjoint(
    checkpoint: Mapping[str, Any],
    *,
    generator_seed: int,
    sample_indices: np.ndarray,
    artifact_sha256: str,
) -> None:
    identities = checkpoint.get("dataset_identities")
    if not isinstance(identities, Mapping) or set(identities) != {"train", "val"}:
        raise ValueError(
            "2D benchmark checkpoint lacks exact train/val artifact identities"
        )
    heldout_ranges = _validated_ranges(
        _contiguous_ranges(sample_indices),
        where="held-out sample ranges",
    )
    for split in ("train", "val"):
        identity = identities[split]
        if not isinstance(identity, Mapping):
            raise TypeError(f"2D checkpoint {split} dataset identity must be a mapping")
        if identity.get("artifact_sha256") == artifact_sha256:
            raise ValueError(f"2D benchmark dataset is the checkpoint {split} artifact")
        provenance = identity.get("provenance")
        if not isinstance(provenance, Mapping):
            raise TypeError(f"2D checkpoint {split} identity lacks provenance")
        split_seed = provenance.get("generator_seed")
        sample_manifest = provenance.get("sample_index")
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise TypeError(f"2D checkpoint {split} generator seed is invalid")
        if not isinstance(sample_manifest, Mapping):
            raise TypeError(f"2D checkpoint {split} sample-index manifest is invalid")
        split_ranges = _validated_ranges(
            sample_manifest.get("contiguous_ranges_inclusive"),
            where=f"2D checkpoint {split} sample ranges",
        )
        if split_seed == generator_seed and _ranges_overlap(split_ranges, heldout_ranges):
            raise ValueError(f"2D benchmark dataset overlaps checkpoint {split} samples")


def _contiguous_ranges(values: np.ndarray) -> list[list[int]]:
    indices = np.unique(np.asarray(values, dtype=np.int64))
    if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0):
        raise ValueError(
            "held-out 2D sample indices must be a non-empty non-negative vector"
        )
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [[int(group[0]), int(group[-1])] for group in groups]


def load_dataset2d(path: str | Path):
    """Validate a 2D HDF5 contract before returning its physical axes."""
    from pimsr_inversion.contracts2d import validate_dataset2d

    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"2D dataset does not exist: {dataset_path}")
    with h5py.File(dataset_path, "r") as h5:
        return validate_dataset2d(h5)


def load_model2d(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
) -> LoadedModel2D:
    """Load only a strict four-channel checkpoint matching ``dataset_path``."""
    import torch
    from pimsr_inversion.contracts2d import validate_checkpoint2d
    from pimsr_inversion.network2d import PimsrNet2D

    checkpoint_path = Path(checkpoint_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"2D checkpoint does not exist: {checkpoint_path}")
    dataset_signature = _stat_signature(dataset_path)
    with h5py.File(dataset_path, "r") as h5:
        from pimsr_inversion.contracts2d import validate_dataset2d

        contract = validate_dataset2d(h5)
        generator_seed = int(np.asarray(h5.attrs["generator_seed"]))
        sample_indices = h5["sample_index"][:].astype(np.int64)
    dataset_sha256, dataset_size = _stable_file_digest(
        dataset_path,
        expected_signature=dataset_signature,
    )
    payload, checkpoint_sha256, checkpoint_size = _stable_file_bytes(
        checkpoint_path,
        kind="2D checkpoint",
    )
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("2D checkpoint root must be a mapping")
    validate_checkpoint2d(checkpoint, contract)
    _assert_heldout_disjoint(
        checkpoint,
        generator_seed=generator_seed,
        sample_indices=sample_indices,
        artifact_sha256=dataset_sha256,
    )
    model = PimsrNet2D.from_checkpoint(checkpoint).eval()
    return LoadedModel2D(
        model=model,
        checkpoint=checkpoint,
        contract=contract,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint_size,
        dataset_sha256=dataset_sha256,
        dataset_size_bytes=dataset_size,
    )


def stack_dataset_observations(
    h5: h5py.File,
    selection: slice | int | np.ndarray,
) -> np.ndarray:
    """Read the canonical four HDF5 channels in TE/Zyx, TM/Zxy order.

    The returned tensor always retains a leading sample axis, including for
    an integer selection of one HDF5 row.
    """
    phase_te = np.asarray(h5["obs_mt_phase"][selection], dtype=np.float32)
    phase_tm = np.asarray(h5["obs_mt_phase_tm"][selection], dtype=np.float32)
    channels = (
        np.asarray(h5["obs_mt_log10_rho"][selection], dtype=np.float32),
        phase_te / 45.0,
        np.asarray(h5["obs_mt_log10_rho_tm"][selection], dtype=np.float32),
        phase_tm / 45.0,
    )
    shape = channels[0].shape
    if any(channel.shape != shape for channel in channels):
        raise ValueError("2D observation channels must have matching shapes")
    if any(not np.isfinite(channel).all() for channel in channels):
        raise ValueError("2D observation channels must be finite")
    if any(np.any((phase < 0.0) | (phase >= 180.0)) for phase in (phase_te, phase_tm)):
        raise ValueError("2D phase must follow the declared [0, 180) convention")
    if len(shape) == 2:
        return np.stack(channels, axis=0)[None]
    if len(shape) == 3:
        return np.stack(channels, axis=1)
    raise ValueError(
        "2D observation channels must have shape (frequency, station) or "
        "(sample, frequency, station)"
    )


def prepare_profile_observation(
    modes: Mapping[str, np.ndarray],
    checkpoint: Mapping[str, Any],
) -> np.ndarray:
    """Prepare one masked real profile in canonical four-channel order.

    Unsupported periods are filled with the training mean, which becomes zero
    after normalisation.  Edge values are never extrapolated into the model.
    """
    phase_te = np.asarray(modes["ph_te"], dtype=np.float32)
    phase_tm = np.asarray(modes["ph_tm"], dtype=np.float32)
    channels = (
        np.asarray(modes["lr_te"], dtype=np.float32),
        phase_te / 45.0,
        np.asarray(modes["lr_tm"], dtype=np.float32),
        phase_tm / 45.0,
    )
    masks = (
        np.asarray(modes["mask_te"]),
        np.asarray(modes["mask_te"]),
        np.asarray(modes["mask_tm"]),
        np.asarray(modes["mask_tm"]),
    )
    shape = channels[0].shape
    if len(shape) != 2 or any(channel.shape != shape for channel in channels):
        raise ValueError("profile mode arrays must share shape (frequency, station)")
    if any(mask.shape != shape or mask.dtype.kind != "b" for mask in masks):
        raise ValueError("profile mode masks must be boolean and match the data")
    for channel, mask in zip(channels, masks, strict=True):
        if not np.isfinite(channel[mask]).all():
            raise ValueError("profile mode arrays must be finite on their valid masks")
    for phase, mask in ((phase_te, masks[1]), (phase_tm, masks[3])):
        if np.any((phase[mask] < 0.0) | (phase[mask] >= 180.0)):
            raise ValueError("profile phase must follow the declared [0, 180) convention")

    observation = np.stack(channels, axis=0)[None]
    valid = np.stack(masks, axis=0)[None]
    mean = np.asarray(checkpoint["stats_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["stats_std"], dtype=np.float32)
    if mean.shape != (1, 4, 1, 1) or std.shape != (1, 4, 1, 1):
        raise ValueError("2D checkpoint stats must have shape (1, 4, 1, 1)")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("2D checkpoint stats must be finite with positive std")
    observation = np.where(valid, observation, mean)
    return ((observation - mean) / std).astype(np.float32, copy=False)


def interpolate_periods_in_band(
    source_periods: np.ndarray,
    source_values: np.ndarray,
    source_mask: np.ndarray,
    target_periods: np.ndarray,
    fill_values: np.ndarray,
) -> np.ndarray:
    """Log-period interpolation that handles either source axis direction."""
    source_periods = np.asarray(source_periods, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    source_mask = np.asarray(source_mask)
    target_periods = np.asarray(target_periods, dtype=float)
    result = np.asarray(fill_values, dtype=float).copy()
    if not (
        source_periods.shape == source_values.shape == source_mask.shape
        and target_periods.shape == result.shape
    ):
        raise ValueError("period interpolation inputs have incompatible shapes")
    if source_mask.dtype.kind != "b":
        raise ValueError("source_mask must be boolean")
    valid = source_mask & np.isfinite(source_values) & np.isfinite(source_periods)
    if int(valid.sum()) < 2:
        return result
    if np.any(source_periods[valid] <= 0.0) or np.any(target_periods <= 0.0):
        raise ValueError("periods must be positive")

    source_log = np.log10(source_periods[valid])
    values = source_values[valid]
    order = np.argsort(source_log)
    source_log, values = source_log[order], values[order]
    if np.any(np.diff(source_log) <= 0.0):
        raise ValueError("valid source periods must be unique")
    target_log = np.log10(target_periods)
    in_band = (target_log >= source_log[0]) & (target_log <= source_log[-1])
    result[in_band] = np.interp(target_log[in_band], source_log, values)
    return result


def prepare_empty_workdir(path: str | Path) -> Path:
    """Create an output directory, refusing any pre-existing contents."""
    workdir = Path(path).resolve()
    if workdir.exists():
        if not workdir.is_dir():
            raise NotADirectoryError(f"workdir is not a directory: {workdir}")
        existing = next(workdir.iterdir(), None)
        if existing is not None:
            raise FileExistsError(
                f"workdir is not empty ({existing.name!r}); use a fresh directory"
            )
    else:
        workdir.mkdir(parents=True)
    return workdir


def run_checked(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run an external solver and reject non-zero exit status with context."""
    proc = subprocess.run(
        list(command),
        cwd=Path(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise RuntimeError(
            f"external solver exited with status {proc.returncode}\n"
            f"stdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
        )
    return proc

"""Pinned, fail-closed ModEM 2-D forward bridge.

The bridge intentionally has a narrow contract: a canonical PIMSR 64 x 48
log10-resistivity raster is mapped as a piecewise-constant physical model onto
an explicitly versioned ModEM mesh.  ModEM writes E/B impedances in the
``exp(+i omega t)`` convention requested by the data template; no manual
complex conjugation is performed here.

This module does not select data or tune a mesh.  Selection and convergence
eligibility belong to the public-only validation driver.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat as stat_module
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

MU0 = 4.0e-7 * math.pi

CANONICAL_MODEL_SHAPE = (64, 48)
CANONICAL_RESPONSE_SHAPE = (8, 12)
MODE_ORDER = ("TE", "TM")

PINNED_MODEM_COMMIT = "55a4aa62f7e8366fbf78a23ee8a19c1d4561d0c3"
PINNED_MODEM_TREE = "56f691e1c37495179113521c4111149ecf999695"
PINNED_MODEM_TAG = "EMIW-2026"
PINNED_CONTAINER_DIGEST = (
    "sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517"
)
PINNED_CONTAINER_REF = f"ubuntu:24.04@{PINNED_CONTAINER_DIGEST}"

BUILD_RECIPE_ID = "modem2d-legacy-gfortran-ubuntu24.04-v1"
BUILD_RECIPE = {
    "configure_source": "f90/CONFIG/Configure.2D_MT.OSU.GFortran",
    "configure_git_blob": "8b34b5c1be5b1ac34398fc30641eae7ea9d06819",
    "configure_file_sha256": (
        "12565569747a7b4d51700a91598e2be4f067aa830fc8a7f40ba885b558bba597"
    ),
    "configure_invocation": (
        "bash <(tr -d '\\r' < CONFIG/Configure.2D_MT.OSU.GFortran) Makefile.2D release"
    ),
    "make_invocation": "make -f Makefile.2D",
    "compiler": "gfortran",
    "compiler_version": "13.3.0-6ubuntu2~24.04.1",
    "compile_flags": ["-O3", "-ffree-line-length-none", "-x", "f95-cpp-input"],
    "link_flags": ["-L/usr/lib64", "-llapack", "-lblas"],
    "packages": {
        "gcc-13": "13.3.0-6ubuntu2~24.04.1",
        "gfortran": "4:13.2.0-7ubuntu1",
        "gfortran-13": "13.3.0-6ubuntu2~24.04.1",
        "libblas-dev:amd64": "3.12.0-3build1.1",
        "libc6:amd64": "2.39-0ubuntu8.8",
        "liblapack-dev:amd64": "3.12.0-3build1.1",
        "make": "4.3-4.1build2",
        "perl": "5.38.2-3.2ubuntu0.3",
    },
}

_PINNED_BUILD_FILES: dict[str, tuple[str, int]] = {
    "modem-55a4aa62.tar": (
        "7dd7c8c2b38ac1cdf8b9601998ecb55cc71c079b8ce3476cbcb72c27469b346a",
        7_864_320,
    ),
    "source/f90/Makefile.2D": (
        "193b923d1d5170d4a632354203980a1c00948a7836839455d2a36da2b0208fe4",
        9_327,
    ),
    "logs/build.log": (
        "d7d6fc5eb5c54c3d1df07072e2d5f52f8c1346befa4bc8a05beffe7c1b9e3e6f",
        25_828,
    ),
    "logs/build-attempt2.log": (
        "e035fee7fe4b20d282d9bffe74977b76cc18c58a839b00b09ce12cc7e21cabc9",
        22_172,
    ),
    "runtime/bin/Mod2DMT": (
        "22dc47a8a81fc2b2dc9d643994e7cc9e0d9ce284050e61bef007238d7980e25c",
        608_288,
    ),
    "runtime/lib/libblas.so.3": (
        "e748efcae5753fe4a652877fccdb5895ac6f7605668a2db878b19c914e78e3a8",
        677_880,
    ),
    "runtime/lib/libgcc_s.so.1": (
        "d93224d2b0dab4247598be683adca02f5cf00586f99c187579cd7e92058fb7cb",
        183_024,
    ),
    "runtime/lib/libgfortran.so.5": (
        "342618ccfebe840446e4ef2b7893f4161b2084845e570f42d68a70bfde58a58d",
        3_263_912,
    ),
    "runtime/lib/liblapack.so.3": (
        "851bb1fc5833ede9ed704b4417a251a899976d5e0915de40452615187a65278f",
        7_268_368,
    ),
}


class ProvenanceError(RuntimeError):
    """Pinned source, build, container, or binary identity is invalid."""


class PublicationError(RuntimeError):
    """An artifact bundle could not be published or safely rolled back."""


@dataclass(frozen=True)
class ArtifactSnapshot:
    """Stable identity of one exact file."""

    path: Path
    sha256: str
    size_bytes: int
    stat_signature: tuple[int, int, int, int]
    payload: bytes

    def record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = os.lstat(path)
    if not stat_module.S_ISREG(stat.st_mode):
        raise RuntimeError(f"expected regular non-link file: {path}")
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def snapshot_file(path: str | Path, *, role: str) -> ArtifactSnapshot:
    """Read one regular inode through one no-follow descriptor."""
    requested = Path(path)
    try:
        before = os.lstat(requested)
    except OSError as exc:
        raise FileNotFoundError(f"cannot lstat {role}: {requested}") from exc
    if not stat_module.S_ISREG(before.st_mode):
        raise FileNotFoundError(f"{role} is not a regular non-link file: {requested}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(requested, flags)
        opened = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"{role} changed before descriptor open: {requested}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    after_path = os.lstat(requested)
    signatures = {
        (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
        )
        for value in (before, opened, after_fd, after_path)
    }
    if len(signatures) != 1 or not stat_module.S_ISREG(after_path.st_mode):
        raise RuntimeError(f"{role} changed while it was read: {requested}")
    signature = signatures.pop()
    payload = b"".join(chunks)
    if len(payload) != signature[2]:
        raise RuntimeError(f"{role} size changed while it was read: {requested}")
    resolved = requested.resolve(strict=True)
    return ArtifactSnapshot(
        resolved,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        signature,
        payload,
    )


def require_snapshot_unchanged(snapshot: ArtifactSnapshot, *, role: str) -> None:
    """Reject a file that changed after its identity was captured."""
    current = snapshot_file(snapshot.path, role=role)
    if (
        current.path != snapshot.path
        or current.sha256 != snapshot.sha256
        or current.size_bytes != snapshot.size_bytes
        or current.stat_signature != snapshot.stat_signature
    ):
        raise RuntimeError(f"{role} changed after it was verified: {snapshot.path}")


def canonical_json_sha256(value: object) -> str:
    """Hash an object using the repository's canonical JSON representation."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MeshConfig:
    """Complete, versioned ModEM Earth-mesh policy."""

    mesh_id: str
    version: int
    core_width_m: float
    core_count: int
    padding_count_each_side: int
    padding_growth: float
    first_dz_m: float
    vertical_growth: float
    max_dz_m: float
    minimum_depth_m: float = 220_000.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.mesh_id):
            raise ValueError("mesh_id must be a lowercase stable identifier")
        if self.version <= 0 or self.core_count <= 0 or self.padding_count_each_side <= 0:
            raise ValueError("mesh version and counts must be positive")
        finite = (
            self.core_width_m,
            self.padding_growth,
            self.first_dz_m,
            self.vertical_growth,
            self.max_dz_m,
            self.minimum_depth_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in finite):
            raise ValueError(
                "mesh dimensions and growth factors must be finite and positive"
            )
        if self.padding_growth <= 1.0 or self.vertical_growth <= 1.0:
            raise ValueError("mesh growth factors must exceed one")
        if self.max_dz_m < self.first_dz_m:
            raise ValueError("max_dz_m must not be smaller than first_dz_m")
        if not math.isclose(
            self.core_width_m * self.core_count,
            24_000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("the frozen physical core must span exactly 24 km")

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema": "pimsr-modem2d-mesh",
            "schema_version": 1,
            "mesh_id": self.mesh_id,
            "version": self.version,
            "core_width_m": self.core_width_m,
            "core_count": self.core_count,
            "padding_count_each_side": self.padding_count_each_side,
            "padding_growth": self.padding_growth,
            "first_dz_m": self.first_dz_m,
            "vertical_growth": self.vertical_growth,
            "max_dz_m": self.max_dz_m,
            "minimum_depth_m": self.minimum_depth_m,
            "mapping": (
                "nearest canonical physical cell centre, tie to lower index; "
                "piecewise constant edge extension"
            ),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.canonical_record())

    def cell_widths(
        self, depth_centres_m: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del depth_centres_m  # legacy geometric meshes do not depend on the raster axis
        core = np.full(self.core_count, self.core_width_m, dtype=np.float64)
        exponent = np.arange(1, self.padding_count_each_side + 1, dtype=np.float64)
        near_to_far = self.core_width_m * self.padding_growth**exponent
        dy = np.concatenate((near_to_far[::-1], core, near_to_far))
        dz_values: list[float] = []
        width = self.first_dz_m
        depth = 0.0
        while depth < self.minimum_depth_m:
            dz_values.append(width)
            depth += width
            width = min(width * self.vertical_growth, self.max_dz_m)
        dz = np.asarray(dz_values, dtype=np.float64)
        if not np.isfinite(dy).all() or not np.isfinite(dz).all():
            raise ValueError("mesh construction produced non-finite widths")
        return dy, dz

    def padding_perturbation(self) -> MeshConfig:
        """Return the preregistered larger-domain perturbation of this mesh."""
        return replace(
            self,
            mesh_id=f"{self.mesh_id}-padding-plus2",
            padding_count_each_side=self.padding_count_each_side + 2,
        )


CANONICAL_DEPTH_CENTRES_SHA256 = (
    "d6382014a4672008ffda4952e31ab91123b0b70e44610d8040625ec3c424636f"
)
CANONICAL_X_CENTRES_SHA256 = (
    "a0a28ac3698ab0a519fc51e59a2bce7f92a0a3793ae1d3adb9d0567e80fd860b"
)


@dataclass(frozen=True)
class NestedMeshConfig:
    """A physically identical family of axis-wise nested ModEM meshes.

    The base vertical partition contains every arithmetic-midpoint boundary of
    the canonical raster.  Refinement therefore changes only the PDE
    discretisation, never the piecewise-constant geological interfaces.
    Each axis can be refined independently by exactly splitting every base
    cell, so all exploratory candidates and the reference share outer bounds
    and geological interfaces.
    """

    mesh_id: str
    version: int
    base_core_width_m: float
    base_core_count: int
    base_padding_count_each_side: int
    base_padding_growth: float
    minimum_vertical_subdivisions: int
    maximum_base_dz_m: float
    deep_padding_growth: float
    maximum_deep_macro_dz_m: float
    minimum_depth_m: float
    horizontal_refinement_factor: int
    vertical_refinement_factor: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.mesh_id):
            raise ValueError("mesh_id must be a lowercase stable identifier")
        integer_values = (
            self.version,
            self.base_core_count,
            self.base_padding_count_each_side,
            self.minimum_vertical_subdivisions,
            self.horizontal_refinement_factor,
            self.vertical_refinement_factor,
        )
        if any(type(value) is not int or value <= 0 for value in integer_values):
            raise ValueError("nested mesh versions, counts, and factors must be positive")
        finite_values = (
            self.base_core_width_m,
            self.base_padding_growth,
            self.maximum_base_dz_m,
            self.deep_padding_growth,
            self.maximum_deep_macro_dz_m,
            self.minimum_depth_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in finite_values):
            raise ValueError("nested mesh dimensions must be finite and positive")
        if self.base_padding_growth <= 1.0 or self.deep_padding_growth <= 1.0:
            raise ValueError("nested mesh padding growth factors must exceed one")
        if self.maximum_deep_macro_dz_m < self.maximum_base_dz_m:
            raise ValueError("deep macro-cell cap must not be below the base dz cap")
        if not math.isclose(
            self.base_core_width_m * self.base_core_count,
            24_000.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("the frozen physical core must span exactly 24 km")

    def canonical_record(self) -> dict[str, object]:
        return {
            "schema": "pimsr-modem2d-nested-mesh",
            "schema_version": 1,
            "mesh_id": self.mesh_id,
            "version": self.version,
            "base_core_width_m": self.base_core_width_m,
            "base_core_count": self.base_core_count,
            "base_padding_count_each_side": self.base_padding_count_each_side,
            "base_padding_growth": self.base_padding_growth,
            "minimum_vertical_subdivisions": self.minimum_vertical_subdivisions,
            "maximum_base_dz_m": self.maximum_base_dz_m,
            "deep_padding_growth": self.deep_padding_growth,
            "maximum_deep_macro_dz_m": self.maximum_deep_macro_dz_m,
            "minimum_depth_m": self.minimum_depth_m,
            "horizontal_refinement_factor": self.horizontal_refinement_factor,
            "vertical_refinement_factor": self.vertical_refinement_factor,
            "canonical_depth_centres_sha256": CANONICAL_DEPTH_CENTRES_SHA256,
            "canonical_x_centres_sha256": CANONICAL_X_CENTRES_SHA256,
            "horizontal_partition": (
                "62.5m-aligned 24km core plus common geometric padding; exact "
                "equal-width subdivision by horizontal_refinement_factor"
            ),
            "vertical_partition": (
                "surface zero, arithmetic midpoint canonical interfaces, "
                "extrapolated bottom edge, capped deep edge-extension; every base "
                "cell exact equal-width subdivision by vertical_refinement_factor"
            ),
            "mapping": (
                "nearest canonical physical cell centre, tie to lower index; "
                "piecewise constant edge extension with invariant interfaces"
            ),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.canonical_record())

    @staticmethod
    def _subdivide(widths: np.ndarray, counts: np.ndarray) -> np.ndarray:
        parts = [
            np.full(int(count), float(width) / int(count), dtype=np.float64)
            for width, count in zip(widths, counts, strict=True)
        ]
        return np.concatenate(parts)

    def _base_vertical_widths(self, depth_centres_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_centres_m, dtype="<f8")
        if (
            depth.shape != (CANONICAL_MODEL_SHAPE[0],)
            or not np.isfinite(depth).all()
            or np.any(depth <= 0.0)
            or np.any(np.diff(depth) <= 0.0)
            or hashlib.sha256(depth.tobytes()).hexdigest()
            != CANONICAL_DEPTH_CENTRES_SHA256
        ):
            raise ValueError("nested mesh requires the exact canonical depth axis")
        internal = 0.5 * (depth[:-1] + depth[1:])
        bottom = depth[-1] + 0.5 * (depth[-1] - depth[-2])
        edges = np.concatenate((np.asarray([0.0]), internal, np.asarray([bottom])))
        physical_widths = np.diff(edges)
        counts = np.maximum(
            self.minimum_vertical_subdivisions,
            np.ceil(physical_widths / self.maximum_base_dz_m).astype(np.int64),
        )
        base = [self._subdivide(physical_widths, counts)]
        current_depth = float(bottom)
        macro_width = float(physical_widths[-1])
        while current_depth < self.minimum_depth_m:
            macro_width = min(
                macro_width * self.deep_padding_growth,
                self.maximum_deep_macro_dz_m,
            )
            actual_width = min(macro_width, self.minimum_depth_m - current_depth)
            count = max(
                self.minimum_vertical_subdivisions,
                math.ceil(actual_width / self.maximum_base_dz_m),
            )
            base.append(
                np.full(count, actual_width / count, dtype=np.float64)
            )
            current_depth += actual_width
        return np.concatenate(base)

    def cell_widths(
        self, depth_centres_m: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if depth_centres_m is None:
            raise ValueError("nested mesh construction requires canonical depth centres")
        core = np.full(
            self.base_core_count, self.base_core_width_m, dtype=np.float64
        )
        exponent = np.arange(
            1, self.base_padding_count_each_side + 1, dtype=np.float64
        )
        near_to_far = self.base_core_width_m * self.base_padding_growth**exponent
        base_dy = np.concatenate((near_to_far[::-1], core, near_to_far))
        base_dz = self._base_vertical_widths(depth_centres_m)
        horizontal_factor = self.horizontal_refinement_factor
        vertical_factor = self.vertical_refinement_factor
        dy = np.repeat(base_dy / horizontal_factor, horizontal_factor)
        dz = np.repeat(base_dz / vertical_factor, vertical_factor)
        if (
            not np.isfinite(dy).all()
            or not np.isfinite(dz).all()
            or np.any(dy <= 0.0)
            or np.any(dz <= 0.0)
            or not math.isclose(
                float(dz.sum()), self.minimum_depth_m, rel_tol=0.0, abs_tol=1e-7
            )
        ):
            raise ValueError("nested mesh construction produced invalid widths")
        return dy, dz

    def padding_perturbation(self) -> NestedMeshConfig:
        return replace(
            self,
            mesh_id=f"{self.mesh_id}-padding-plus2",
            base_padding_count_each_side=self.base_padding_count_each_side + 2,
        )


MESH_CONFIGS: dict[str, MeshConfig | NestedMeshConfig] = {
    "baseline": MeshConfig("baseline", 1, 250.0, 96, 16, 1.4, 50.0, 1.12, 10_000.0),
    "refined": MeshConfig("refined", 1, 125.0, 192, 18, 1.4, 25.0, 1.08, 5_000.0),
    "ultra": MeshConfig("ultra", 1, 62.5, 384, 20, 1.4, 12.5, 1.05, 2_500.0),
    "ultra2": MeshConfig(
        "ultra2",
        1,
        31.25,
        768,
        22,
        1.4,
        6.25,
        1.04,
        1_250.0,
    ),
    "nested-base-v1": NestedMeshConfig(
        "nested-base-v1",
        1,
        62.5,
        384,
        20,
        1.4,
        2,
        2_500.0,
        1.35,
        10_000.0,
        220_000.0,
        1,
        1,
    ),
    "nested-horizontal-only-v1": NestedMeshConfig(
        "nested-horizontal-only-v1",
        1,
        62.5,
        384,
        20,
        1.4,
        2,
        2_500.0,
        1.35,
        10_000.0,
        220_000.0,
        2,
        1,
    ),
    "nested-production-v1": NestedMeshConfig(
        "nested-production-v1",
        1,
        62.5,
        384,
        20,
        1.4,
        2,
        2_500.0,
        1.35,
        10_000.0,
        220_000.0,
        1,
        2,
    ),
    "nested-reference-x2-v1": NestedMeshConfig(
        "nested-reference-x2-v1",
        1,
        62.5,
        384,
        20,
        1.4,
        2,
        2_500.0,
        1.35,
        10_000.0,
        220_000.0,
        2,
        2,
    ),
}


def _immutable_float_array(
    value: Any, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CanonicalTruth:
    """One exact canonical PIMSR 2-D physical truth and observation geometry."""

    log10_resistivity: np.ndarray
    x_centres_m: np.ndarray
    depth_centres_m: np.ndarray
    frequencies_hz: np.ndarray
    station_x_m: np.ndarray
    sample_id: str

    def __post_init__(self) -> None:
        arrays = {
            "log10_resistivity": _immutable_float_array(
                self.log10_resistivity,
                name="log10_resistivity",
                shape=CANONICAL_MODEL_SHAPE,
            ),
            "x_centres_m": _immutable_float_array(
                self.x_centres_m,
                name="x_centres_m",
                shape=(CANONICAL_MODEL_SHAPE[1],),
            ),
            "depth_centres_m": _immutable_float_array(
                self.depth_centres_m,
                name="depth_centres_m",
                shape=(CANONICAL_MODEL_SHAPE[0],),
            ),
            "frequencies_hz": _immutable_float_array(
                self.frequencies_hz,
                name="frequencies_hz",
                shape=(CANONICAL_RESPONSE_SHAPE[0],),
            ),
            "station_x_m": _immutable_float_array(
                self.station_x_m,
                name="station_x_m",
                shape=(CANONICAL_RESPONSE_SHAPE[1],),
            ),
        }
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        for name in ("x_centres_m", "depth_centres_m", "frequencies_hz", "station_x_m"):
            if not np.all(np.diff(getattr(self, name)) > 0.0):
                raise ValueError(f"{name} must be strictly increasing")
        if np.any(self.depth_centres_m <= 0.0) or np.any(self.frequencies_hz <= 0.0):
            raise ValueError("depths and frequencies must be positive")
        if not self.sample_id or len(self.sample_id) > 128:
            raise ValueError(
                "sample_id must be a non-empty identifier of at most 128 characters"
            )

    def identity_record(self) -> dict[str, object]:
        digest = hashlib.sha256()
        for array in (
            self.log10_resistivity,
            self.x_centres_m,
            self.depth_centres_m,
            self.frequencies_hz,
            self.station_x_m,
        ):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        return {
            "schema": "pimsr-canonical-truth-2d",
            "schema_version": 1,
            "sample_id": self.sample_id,
            "model_shape": list(CANONICAL_MODEL_SHAPE),
            "response_shape": list(CANONICAL_RESPONSE_SHAPE),
            "arrays_sha256": digest.hexdigest(),
        }


@dataclass(frozen=True)
class ModEMResponse:
    """Canonical TE=Zyx and TM=Zxy responses derived directly from E/B."""

    frequencies_hz: np.ndarray
    station_x_m: np.ndarray
    z_eb_te: np.ndarray
    z_eb_tm: np.ndarray
    log10_rho_te: np.ndarray
    phase_te_deg: np.ndarray
    log10_rho_tm: np.ndarray
    phase_tm_deg: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "frequencies_hz": (CANONICAL_RESPONSE_SHAPE[0],),
            "station_x_m": (CANONICAL_RESPONSE_SHAPE[1],),
            "z_eb_te": CANONICAL_RESPONSE_SHAPE,
            "z_eb_tm": CANONICAL_RESPONSE_SHAPE,
            "log10_rho_te": CANONICAL_RESPONSE_SHAPE,
            "phase_te_deg": CANONICAL_RESPONSE_SHAPE,
            "log10_rho_tm": CANONICAL_RESPONSE_SHAPE,
            "phase_tm_deg": CANONICAL_RESPONSE_SHAPE,
        }
        for name, shape in shapes.items():
            array = np.array(getattr(self, name), copy=True)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for phase_name in ("phase_te_deg", "phase_tm_deg"):
            phase = getattr(self, phase_name)
            if np.any(phase < 0.0) or np.any(phase >= 180.0):
                raise ValueError(f"{phase_name} must use canonical [0, 180) degrees")

    def npz_bytes(self) -> bytes:
        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            schema=np.asarray("pimsr-modem2d-response"),
            schema_version=np.asarray(1, dtype=np.int64),
            frequencies_hz=self.frequencies_hz,
            station_x_m=self.station_x_m,
            z_eb_te_real=self.z_eb_te.real,
            z_eb_te_imag=self.z_eb_te.imag,
            z_eb_tm_real=self.z_eb_tm.real,
            z_eb_tm_imag=self.z_eb_tm.imag,
            log10_rho_te=self.log10_rho_te,
            phase_te_deg=self.phase_te_deg,
            log10_rho_tm=self.log10_rho_tm,
            phase_tm_deg=self.phase_tm_deg,
        )
        return stream.getvalue()


def nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Nearest sorted reference index, resolving exact ties toward the left."""
    if reference.ndim != 1 or query.ndim != 1 or reference.size == 0:
        raise ValueError("nearest_indices requires non-empty one-dimensional arrays")
    if not np.all(np.diff(reference) > 0.0):
        raise ValueError("nearest_indices reference must be strictly increasing")
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    choose_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(choose_right, right, left)


def mapped_model(
    truth: CanonicalTruth,
    mesh: MeshConfig | NestedMeshConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map canonical log10(rho) onto ModEM Earth cell centres."""
    if isinstance(mesh, NestedMeshConfig):
        x_centres = np.asarray(truth.x_centres_m, dtype="<f8")
        if hashlib.sha256(x_centres.tobytes()).hexdigest() != CANONICAL_X_CENTRES_SHA256:
            raise ValueError("nested mesh requires the exact canonical x axis")
    dy, dz = mesh.cell_widths(truth.depth_centres_m)
    y_centres = np.cumsum(dy) - 0.5 * dy
    profile_x = y_centres - 0.5 * float(dy.sum())
    depth_centres = np.cumsum(dz) - 0.5 * dz
    ix = nearest_indices(truth.x_centres_m, profile_x)
    iz = nearest_indices(truth.depth_centres_m, depth_centres)
    mapped = truth.log10_resistivity[np.ix_(iz, ix)]
    if mapped.shape != (dz.size, dy.size) or not np.isfinite(mapped).all():
        raise RuntimeError("canonical-to-ModEM mapping produced an invalid model")
    return np.array(mapped, copy=True), dy, dz


def _exclusive_ascii(path: Path, text: str) -> ArtifactSnapshot:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        payload = text.encode("ascii")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return snapshot_file(path, role="generated ModEM input")


def write_modem_model(
    path: str | Path,
    truth: CanonicalTruth,
    mesh: MeshConfig | NestedMeshConfig,
) -> tuple[ArtifactSnapshot, dict[str, object]]:
    """Write strict Mackie ``LOGE`` model values as natural-log resistivity."""
    destination = Path(path).resolve()
    mapped_log10, dy, dz = mapped_model(truth, mesh)
    mapped_ln = mapped_log10 * math.log(10.0)
    lines = [f"{dy.size:d} {dz.size:d} LOGE"]
    for values in (dy, dz):
        for start in range(0, values.size, 10):
            lines.append(
                " ".join(f"{value:.12e}" for value in values[start : start + 10])
            )
    lines.append("0")
    lines.extend(" ".join(f"{value:.12e}" for value in row) for row in mapped_ln)
    snapshot = _exclusive_ascii(destination, "\n".join(lines) + "\n")
    record = {
        "representation": "LOGE natural_log_resistivity_ohm_m",
        "ny": int(dy.size),
        "nz_earth": int(dz.size),
        "total_width_m": float(dy.sum()),
        "total_depth_m": float(dz.sum()),
        "mapping": mesh.canonical_record()["mapping"],
        "spatial_operation_order": (
            "map canonical log10(rho) piecewise-constantly by physical centres, "
            "then multiply mapped values by ln(10)"
        ),
        "artifact": snapshot.record(),
    }
    return snapshot, record


def write_modem_template(
    path: str | Path,
    truth: CanonicalTruth,
    mesh: MeshConfig | NestedMeshConfig,
) -> tuple[ArtifactSnapshot, dict[str, object]]:
    """Write complete TE/TM E/B template in requested ``exp(+i omega t)``."""
    dy, _ = mesh.cell_widths(truth.depth_centres_m)
    width = float(dy.sum())
    station_y = 0.5 * width + truth.station_x_m
    if np.any(station_y <= 0.0) or np.any(station_y >= width):
        raise ValueError("canonical stations are outside the ModEM domain")
    periods = 1.0 / truth.frequencies_hz
    lines: list[str] = []
    for mode in MODE_ORDER:
        lines.extend(
            (
                f"# PIMSR canonical sample {truth.sample_id} ModEM forward template",
                (
                    "# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) "
                    "Component Real Imag Error"
                ),
                f"> {mode}_Impedance",
                "> exp(+i\\omega t)",
                "> [V/m]/[T]",
                "> 0.00",
                "> 0.000 0.000",
                f"> {periods.size:d} {station_y.size:d}",
            )
        )
        for site_index, y_value in enumerate(station_y, start=1):
            for period in periods:
                lines.append(
                    f"{period:.12e} S{site_index:02d} 0.0 0.0 0.0 "
                    f"{y_value:.12e} 0.0 {mode} 0.0 0.0 1.0"
                )
    snapshot = _exclusive_ascii(Path(path).resolve(), "\n".join(lines) + "\n")
    record = {
        "time_convention_requested": "exp(+i omega t)",
        "manual_conjugation": False,
        "units": "[V/m]/[T] (E/B)",
        "coordinate_mapping": {
            "ModEM_X_m": 0.0,
            "ModEM_Y_m": "total_width_m/2 + PIMSR_station_x_m",
            "ModEM_Z_m": 0.0,
        },
        "mode_mapping": {
            "ModEM_TE_Ex_over_By": "PIMSR_TE_Ey_over_Hx_Zyx_no_mode_swap",
            "ModEM_TM_Ey_over_Bx": "PIMSR_TM_Ex_over_Hy_Zxy_no_mode_swap",
        },
        "period_count": int(periods.size),
        "station_count": int(station_y.size),
        "rows_per_mode": int(periods.size * station_y.size),
        "artifact": snapshot.record(),
    }
    return snapshot, record


def _header_value(lines: Sequence[str], start: int, *, role: str) -> tuple[str, int]:
    index = start
    while index < len(lines):
        value = lines[index].strip()
        index += 1
        if not value or value.startswith("#"):
            continue
        if not value.startswith(">"):
            raise ValueError(f"expected {role} header, got data at line {index}")
        return value[1:].strip(), index
    raise ValueError(f"missing {role} header")


def parse_modem_response(
    path: str | Path,
    truth: CanonicalTruth,
    mesh: MeshConfig | NestedMeshConfig,
) -> tuple[ModEMResponse, dict[str, object]]:
    """Parse exactly 96 finite TE and 96 finite TM E/B rows.

    The requested file convention is already ``exp(+i omega t)`` because
    ModEM DataIO handles the conversion from its internal ``ISIGN=-1``.  The
    complex values below are therefore consumed directly and are never
    conjugated.
    """
    response_snapshot = snapshot_file(path, role="ModEM forward response")
    try:
        text = response_snapshot.payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("ModEM forward response must be strict ASCII") from exc
    lines = text.splitlines()
    expected_station_y = (
        0.5 * sum(mesh.cell_widths(truth.depth_centres_m)[0]) + truth.station_x_m
    )
    z_by_mode = {
        mode: np.full(CANONICAL_RESPONSE_SHAPE, np.nan + 1j * np.nan)
        for mode in MODE_ORDER
    }
    seen: dict[str, set[tuple[int, int]]] = {mode: set() for mode in MODE_ORDER}
    block_seen: set[str] = set()
    row_counts = {mode: 0 for mode in MODE_ORDER}
    current_mode: str | None = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(">"):
            type_value = stripped[1:].strip()
            match = re.fullmatch(r"(TE|TM)_Impedance", type_value)
            if match is None:
                raise ValueError(f"unexpected ModEM response block: {type_value!r}")
            current_mode = match.group(1)
            if current_mode in block_seen:
                raise ValueError(f"duplicate {current_mode} response block")
            block_seen.add(current_mode)
            sign_value, index = _header_value(
                lines, index, role=f"{current_mode} time convention"
            )
            units_value, index = _header_value(lines, index, role=f"{current_mode} units")
            _orientation, index = _header_value(
                lines, index, role=f"{current_mode} orientation"
            )
            _origin, index = _header_value(lines, index, role=f"{current_mode} origin")
            counts_value, index = _header_value(
                lines, index, role=f"{current_mode} counts"
            )
            compact_sign = sign_value.replace(" ", "")
            if compact_sign not in {"exp(+i\\omegat)", "exp(+iomegat)"}:
                raise ValueError(
                    f"{current_mode} output is not requested exp(+i omega t): {sign_value!r}"
                )
            if units_value.replace(" ", "") != "[V/m]/[T]":
                raise ValueError(
                    f"{current_mode} output units are not E/B: {units_value!r}"
                )
            count_fields = counts_value.split()
            if count_fields != ["8", "12"]:
                raise ValueError(
                    f"{current_mode} response declares {counts_value!r}, expected '8 12'"
                )
            continue
        if current_mode is None:
            raise ValueError(f"data row outside a response block at line {index}")
        fields = stripped.split()
        if len(fields) != 11:
            raise ValueError(
                f"response row {index} has {len(fields)} fields, expected 11"
            )
        try:
            period = float(fields[0])
            latitude = float(fields[2])
            longitude = float(fields[3])
            x_coord = float(fields[4])
            y_coord = float(fields[5])
            z_coord = float(fields[6])
            real = float(fields[8])
            imag = float(fields[9])
            error = float(fields[10])
        except ValueError as exc:
            raise ValueError(f"non-numeric ModEM response field at line {index}") from exc
        numeric = (
            period,
            latitude,
            longitude,
            x_coord,
            y_coord,
            z_coord,
            real,
            imag,
            error,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite ModEM response field at line {index}")
        if period <= 0.0 or error <= 0.0:
            raise ValueError(f"non-positive period or error at line {index}")
        if fields[7] != current_mode:
            raise ValueError(
                f"row mode {fields[7]!r} does not match {current_mode} block at line {index}"
            )
        site_match = re.fullmatch(r"S(\d{2})", fields[1])
        if site_match is None:
            raise ValueError(f"invalid canonical site code {fields[1]!r} at line {index}")
        station_index = int(site_match.group(1)) - 1
        if not 0 <= station_index < CANONICAL_RESPONSE_SHAPE[1]:
            raise ValueError(f"site index outside canonical geometry at line {index}")
        if abs(x_coord) > 0.002 or abs(z_coord) > 0.002:
            raise ValueError(f"unexpected ModEM X/Z coordinate at line {index}")
        if abs(y_coord - expected_station_y[station_index]) > 0.002:
            raise ValueError(
                f"ModEM Y coordinate does not match canonical station at line {index}"
            )
        frequency = 1.0 / period
        frequency_index = int(
            np.argmin(np.abs(np.log(truth.frequencies_hz) - math.log(frequency)))
        )
        expected_frequency = float(truth.frequencies_hz[frequency_index])
        if not math.isclose(frequency, expected_frequency, rel_tol=2e-6, abs_tol=0.0):
            raise ValueError(
                f"period does not match a canonical frequency at line {index}"
            )
        key = (frequency_index, station_index)
        if key in seen[current_mode]:
            raise ValueError(
                f"duplicate {current_mode} frequency/station row at line {index}"
            )
        seen[current_mode].add(key)
        value = complex(real, imag)
        if value == 0.0:
            raise ValueError(f"zero E/B impedance at line {index}")
        z_by_mode[current_mode][key] = value
        row_counts[current_mode] += 1

    expected_keys = {
        (frequency_index, station_index)
        for frequency_index in range(CANONICAL_RESPONSE_SHAPE[0])
        for station_index in range(CANONICAL_RESPONSE_SHAPE[1])
    }
    if block_seen != set(MODE_ORDER):
        raise ValueError(f"response blocks are incomplete: {sorted(block_seen)}")
    for mode in MODE_ORDER:
        if row_counts[mode] != 96 or seen[mode] != expected_keys:
            raise ValueError(
                f"expected exactly 96 complete {mode} rows, got {row_counts[mode]}"
            )
        if not np.isfinite(z_by_mode[mode]).all():
            raise ValueError(f"{mode} E/B response is incomplete or non-finite")

    omega = 2.0 * math.pi * truth.frequencies_hz[:, None]
    derived: dict[str, np.ndarray] = {}
    for mode in MODE_ORDER:
        z_eb = z_by_mode[mode]
        rho = MU0 * np.abs(z_eb) ** 2 / omega
        if not np.isfinite(rho).all() or np.any(rho <= 0.0):
            raise ValueError(f"{mode} apparent resistivity is not finite and positive")
        derived[f"log10_rho_{mode.lower()}"] = np.log10(rho)
        derived[f"phase_{mode.lower()}_deg"] = np.mod(np.degrees(np.angle(z_eb)), 180.0)
    response = ModEMResponse(
        frequencies_hz=truth.frequencies_hz,
        station_x_m=truth.station_x_m,
        z_eb_te=z_by_mode["TE"],
        z_eb_tm=z_by_mode["TM"],
        **derived,
    )
    metadata = {
        "artifact": response_snapshot.record(),
        "rows": {"TE": 96, "TM": 96},
        "all_rows_finite": True,
        "time_convention": "exp(+i omega t) as written by ModEM DataIO",
        "manual_conjugation": False,
        "native_units": "[V/m]/[T] (E/B)",
        "rho_formula": "mu0 * abs(E_over_B)**2 / omega",
        "phase_formula": "degrees(angle(E_over_B)) modulo 180",
        "canonical_mode_order": ["TE_Zyx", "TM_Zxy"],
    }
    require_snapshot_unchanged(response_snapshot, role="ModEM forward response")
    return response, metadata


def _run_capture(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError(
            f"failed to execute provenance command: {command[0]}"
        ) from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise ProvenanceError(
            f"provenance command failed ({process.returncode}): {' '.join(command)}: {detail}"
        )
    return process


@dataclass(frozen=True)
class VerifiedRuntime:
    """Exact pinned ModEM runtime verified against source and build evidence."""

    modem_repo: Path
    build_root: Path
    docker_executable: str
    artifacts: tuple[ArtifactSnapshot, ...]
    record: Mapping[str, object]
    identity_sha256: str

    @property
    def binary_path(self) -> Path:
        return self.build_root / "runtime" / "bin" / "Mod2DMT"

    @property
    def runtime_path(self) -> Path:
        return self.build_root / "runtime"

    def require_unchanged(self) -> None:
        current = verify_pinned_runtime(
            modem_repo=self.modem_repo,
            build_root=self.build_root,
            docker_executable=self.docker_executable,
        )
        if current.identity_sha256 != self.identity_sha256:
            raise ProvenanceError(
                "pinned ModEM runtime identity changed after verification"
            )


def _git(repo: Path, *arguments: str) -> str:
    return _run_capture(("git", "-C", str(repo), *arguments)).stdout.strip()


def _verify_build_recipe(
    build_root: Path, modem_repo: Path
) -> tuple[ArtifactSnapshot, ...]:
    snapshots: list[ArtifactSnapshot] = []
    for relative_path, (expected_sha, expected_size) in _PINNED_BUILD_FILES.items():
        snapshot = snapshot_file(
            build_root / relative_path, role=f"pinned {relative_path}"
        )
        if snapshot.sha256 != expected_sha or snapshot.size_bytes != expected_size:
            raise ProvenanceError(
                f"pinned build artifact identity mismatch: {relative_path}"
            )
        snapshots.append(snapshot)

    configure_path = modem_repo / str(BUILD_RECIPE["configure_source"])
    configure_snapshot = snapshot_file(configure_path, role="legacy 2-D configure source")
    if configure_snapshot.sha256 != BUILD_RECIPE["configure_file_sha256"]:
        raise ProvenanceError("legacy 2-D configure source byte hash mismatch")
    configure_blob = _git(
        modem_repo, "hash-object", str(BUILD_RECIPE["configure_source"])
    )
    if configure_blob != BUILD_RECIPE["configure_git_blob"]:
        raise ProvenanceError("legacy 2-D configure source Git blob mismatch")
    snapshots.append(configure_snapshot)

    makefile = (build_root / "source" / "f90" / "Makefile.2D").read_text(encoding="utf-8")
    exact_assignments = (
        "F90 = gfortran",
        "FFLAGS = -O3 -ffree-line-length-none",
        "MPIFLAGS = -x f95-cpp-input",
        "LIBS_PATH = -L/usr/lib64",
        "LIBS = -llapack -lblas",
    )
    if not all(assignment in makefile for assignment in exact_assignments):
        raise ProvenanceError(
            "generated Makefile does not contain the pinned compiler recipe"
        )
    compile_log = (build_root / "logs" / "build-attempt2.log").read_text(encoding="utf-8")
    compile_lines = [
        line for line in compile_log.splitlines() if line.startswith("gfortran  -c")
    ]
    if len(compile_lines) < 30 or not all(
        "-O3 -ffree-line-length-none -x f95-cpp-input" in line for line in compile_lines
    ):
        raise ProvenanceError("compile log does not prove the pinned compiler flags")
    if "-L/usr/lib64 -llapack -lblas" not in compile_log:
        raise ProvenanceError("compile log does not prove the pinned link flags")
    package_log = (build_root / "logs" / "build.log").read_text(encoding="utf-8")
    for package, version in BUILD_RECIPE["packages"].items():
        if f"{package}\t{version}" not in package_log:
            raise ProvenanceError(f"build log lacks pinned package version: {package}")
    return tuple(snapshots)


def verify_pinned_runtime(
    *,
    modem_repo: str | Path,
    build_root: str | Path,
    docker_executable: str = "docker",
) -> VerifiedRuntime:
    """Fail closed unless source, recipe, binary, and image match every pin."""
    repo = Path(modem_repo).resolve(strict=True)
    root = Path(build_root).resolve(strict=True)
    if not repo.is_dir() or not root.is_dir():
        raise FileNotFoundError("ModEM repository and build root must be directories")
    if _git(repo, "rev-parse", "HEAD^{commit}") != PINNED_MODEM_COMMIT:
        raise ProvenanceError("ModEM checkout is not at the pinned commit")
    if _git(repo, "rev-parse", "HEAD^{tree}") != PINNED_MODEM_TREE:
        raise ProvenanceError("ModEM checkout tree does not match the pinned tree")
    tag_commit = _git(repo, "rev-parse", f"refs/tags/{PINNED_MODEM_TAG}^{{commit}}")
    if tag_commit != PINNED_MODEM_COMMIT:
        raise ProvenanceError("pinned ModEM tag does not resolve to the pinned commit")
    if _git(repo, "describe", "--tags", "--exact-match", "HEAD") != PINNED_MODEM_TAG:
        raise ProvenanceError("ModEM HEAD does not have the exact pinned tag")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ProvenanceError("ModEM checkout is not clean")

    snapshots = _verify_build_recipe(root, repo)
    image_process = _run_capture(
        (
            docker_executable,
            "image",
            "inspect",
            PINNED_CONTAINER_REF,
            "--format",
            "{{json .}}",
        )
    )
    try:
        image = json.loads(image_process.stdout)
    except json.JSONDecodeError as exc:
        raise ProvenanceError("Docker returned invalid image provenance JSON") from exc
    repo_digest = f"ubuntu@{PINNED_CONTAINER_DIGEST}"
    if image.get("Id") != PINNED_CONTAINER_DIGEST or repo_digest not in image.get(
        "RepoDigests", []
    ):
        raise ProvenanceError("local Docker image does not match the pinned digest")
    docker_versions = _run_capture(
        (
            docker_executable,
            "version",
            "--format",
            "{{json .Client.Version}} {{json .Server.Version}}",
        )
    ).stdout.strip()
    artifact_records = {
        str(snapshot.path.relative_to(root))
        if snapshot.path.is_relative_to(root)
        else "modem_checkout/" + str(snapshot.path.relative_to(repo)): snapshot.record()
        for snapshot in snapshots
    }
    record: dict[str, object] = {
        "schema": "pimsr-modem2d-runtime-provenance",
        "schema_version": 1,
        "modem": {
            "repository": str(repo),
            "commit": PINNED_MODEM_COMMIT,
            "tree": PINNED_MODEM_TREE,
            "exact_tag": PINNED_MODEM_TAG,
            "checkout_clean": True,
        },
        "container": {
            "reference": PINNED_CONTAINER_REF,
            "image_id": image["Id"],
            "repo_digest": repo_digest,
            "docker_versions_json_pair": docker_versions,
        },
        "build_recipe_id": BUILD_RECIPE_ID,
        "build_recipe": BUILD_RECIPE,
        "artifacts": artifact_records,
    }
    identity_sha = canonical_json_sha256(record)
    return VerifiedRuntime(repo, root, docker_executable, snapshots, record, identity_sha)


def _decoded_strings(value: Any) -> list[str]:
    array = np.asarray(value).reshape(-1)
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in array
    ]


def load_canonical_hdf5(
    path: str | Path,
    *,
    row: int,
) -> tuple[CanonicalTruth, dict[str, object]]:
    """Load one canonical schema-v2 truth with stable source provenance."""
    source = snapshot_file(path, role="canonical PIMSR HDF5 source")
    with h5py.File(io.BytesIO(source.payload), "r") as h5:
        if (
            h5.attrs.get("schema") != "pimsr-mt-2d"
            or int(h5.attrs.get("schema_version", -1)) != 2
        ):
            raise ValueError("expected pimsr-mt-2d schema version 2")
        if _decoded_strings(h5.attrs.get("impedance_components", [])) != ["Zyx", "Zxy"]:
            raise ValueError("canonical HDF5 impedance components must be [Zyx, Zxy]")
        if _decoded_strings(h5.attrs.get("mode_order", [])) != ["te", "tm"]:
            raise ValueError("canonical HDF5 mode order must be [te, tm]")
        if h5.attrs.get("phase_convention") != "degrees_modulo_180_[0,180)":
            raise ValueError("canonical HDF5 phase convention is invalid")
        required = {
            "target_log10_res",
            "x_grid",
            "depth_grid",
            "frequencies",
            "station_x",
            "sample_index",
            "scenario",
        }
        if not required.issubset(h5):
            raise ValueError(
                f"canonical HDF5 is missing datasets: {sorted(required - set(h5))}"
            )
        count = int(h5["target_log10_res"].shape[0])
        if row < 0 or row >= count:
            raise IndexError(f"row {row} is outside canonical HDF5 with {count} samples")
        sample_index = int(h5["sample_index"][row])
        scenario = int(h5["scenario"][row])
        truth = CanonicalTruth(
            log10_resistivity=h5["target_log10_res"][row],
            x_centres_m=h5["x_grid"][:],
            depth_centres_m=h5["depth_grid"][:],
            frequencies_hz=h5["frequencies"][:],
            station_x_m=h5["station_x"][:],
            sample_id=f"sample-{sample_index:06d}",
        )
        hdf_record = {
            "source": source.record(),
            "row": row,
            "sample_index": sample_index,
            "scenario_index": scenario,
            "generator_seed": int(h5.attrs.get("generator_seed", -1)),
            "generation_contract": str(h5.attrs.get("generation_contract", "")),
            "forward_contract": str(h5.attrs.get("forward_contract", "")),
        }
    require_snapshot_unchanged(source, role="canonical PIMSR HDF5 source")
    return truth, hdf_record


def load_canonical_npz(path: str | Path) -> tuple[CanonicalTruth, dict[str, object]]:
    """Load a canonical truth NPZ with an exact, closed key set."""
    source = snapshot_file(path, role="canonical PIMSR NPZ source")
    expected = {
        "truth_log10_resistivity",
        "x_centres_m",
        "depth_centres_m",
        "frequencies_hz",
        "station_x_m",
        "sample_id",
    }
    with np.load(io.BytesIO(source.payload), allow_pickle=False) as bundle:
        if set(bundle.files) != expected:
            raise ValueError(
                "canonical NPZ keys must be exactly " + ", ".join(sorted(expected))
            )
        sample_value = np.asarray(bundle["sample_id"])
        if sample_value.shape != ():
            raise ValueError("canonical NPZ sample_id must be a scalar")
        truth = CanonicalTruth(
            log10_resistivity=bundle["truth_log10_resistivity"],
            x_centres_m=bundle["x_centres_m"],
            depth_centres_m=bundle["depth_centres_m"],
            frequencies_hz=bundle["frequencies_hz"],
            station_x_m=bundle["station_x_m"],
            sample_id=str(sample_value.item()),
        )
    require_snapshot_unchanged(source, role="canonical PIMSR NPZ source")
    return truth, {"source": source.record(), "format": "canonical_npz_v1"}


def _write_bytes_exclusive(path: Path, payload: bytes) -> ArtifactSnapshot:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return snapshot_file(path, role="staged publication artifact")


def _remove_owned_link(path: Path, signature: tuple[int, int, int, int]) -> None:
    if not os.path.lexists(path):
        return
    if _stat_signature(path) != signature:
        raise PublicationError(
            f"refusing to remove a replaced publication artifact: {path}"
        )
    path.unlink()


def publish_artifact_bundle(
    output_dir: str | Path,
    artifacts: Mapping[str, bytes],
    *,
    manifest_name: str,
) -> Path:
    """Publish a no-overwrite bundle, manifest last, with owned-link rollback."""
    requested = Path(output_dir)
    if requested.name in {"", ".", ".."}:
        raise ValueError("output directory must have a concrete leaf name")
    if os.path.lexists(requested):
        raise FileExistsError(
            f"refusing to overwrite existing ModEM output directory: {requested}"
        )
    if not artifacts or manifest_name not in artifacts:
        raise ValueError("artifact bundle must contain its manifest")
    names = list(artifacts)
    if len(names) != len(set(names)):
        raise ValueError("artifact bundle contains duplicate names")
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"artifact name must be a plain filename: {name!r}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    destination = requested.parent.resolve(strict=True) / requested.name
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite existing ModEM output directory: {destination}"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
    )
    published: list[tuple[Path, tuple[int, int, int, int]]] = []
    destination_created = False
    try:
        staged: dict[str, Path] = {}
        for name, payload in artifacts.items():
            staged_path = stage / name
            _write_bytes_exclusive(staged_path, payload)
            staged[name] = staged_path
        try:
            destination.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing ModEM output directory: {destination}"
            ) from exc
        destination_created = True
        publish_order = [name for name in names if name != manifest_name] + [
            manifest_name
        ]
        for name in publish_order:
            target = destination / name
            try:
                os.link(staged[name], target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"publication race for ModEM artifact: {target}"
                ) from exc
            published.append((target, _stat_signature(target)))
        shutil.rmtree(stage)
        return destination
    except BaseException as original:
        rollback_error: BaseException | None = None
        for path, signature in reversed(published):
            try:
                _remove_owned_link(path, signature)
            except BaseException as exc:  # noqa: BLE001  # pragma: no cover
                rollback_error = exc
                break
        if destination_created and rollback_error is None:
            try:
                destination.rmdir()
            except OSError as exc:
                rollback_error = PublicationError(
                    f"published directory gained foreign content during rollback: {destination}"
                )
                rollback_error.__cause__ = exc
        shutil.rmtree(stage, ignore_errors=True)
        if rollback_error is not None:
            raise rollback_error from original
        raise


def _docker_mount(path: Path, target: str, *, readonly: bool) -> str:
    value = f"type=bind,source={path},target={target}"
    return value + (",readonly" if readonly else "")


def _run_solver(
    runtime: VerifiedRuntime,
    *,
    input_dir: Path,
    solver_dir: Path,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], list[str], float]:
    command = [
        runtime.docker_executable,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--workdir",
        "/tmp",
        "--env",
        "LD_LIBRARY_PATH=/runtime/lib",
        "--mount",
        _docker_mount(runtime.runtime_path, "/runtime", readonly=True),
        "--mount",
        _docker_mount(input_dir, "/input", readonly=True),
        "--mount",
        _docker_mount(solver_dir, "/output", readonly=False),
        PINNED_CONTAINER_REF,
        "/runtime/bin/Mod2DMT",
        "-F",
        "/input/model.rho",
        "/input/template.dat",
        "/output/forward.dat",
    ]
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"ModEM forward exceeded {timeout_seconds:g} seconds") from exc
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"ModEM forward failed with exit {process.returncode}: {process.stderr.strip()}"
        )
    return process, command, elapsed


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_modem_forward(
    *,
    runtime: VerifiedRuntime,
    truth: CanonicalTruth,
    mesh: MeshConfig | NestedMeshConfig,
    output_dir: str | Path,
    source_provenance: Mapping[str, object],
    timeout_seconds: float = 1_800.0,
) -> tuple[Path, ModEMResponse, Mapping[str, object]]:
    """Run one pinned forward solve and atomically publish its exact artifacts."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing ModEM output: {output}")
    bridge_snapshot = snapshot_file(__file__, role="ModEM bridge source")
    runtime.require_unchanged()
    work_parent = output.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.run-", dir=work_parent))
    try:
        input_dir = work / "input"
        solver_dir = work / "solver"
        input_dir.mkdir()
        solver_dir.mkdir()
        model_snapshot, model_record = write_modem_model(
            input_dir / "model.rho", truth, mesh
        )
        template_snapshot, template_record = write_modem_template(
            input_dir / "template.dat", truth, mesh
        )
        process, command, elapsed = _run_solver(
            runtime,
            input_dir=input_dir,
            solver_dir=solver_dir,
            timeout_seconds=timeout_seconds,
        )
        solver_files = sorted(path.name for path in solver_dir.iterdir())
        if solver_files != ["forward.dat"]:
            raise RuntimeError(f"ModEM produced an unexpected output set: {solver_files}")
        require_snapshot_unchanged(model_snapshot, role="ModEM model input")
        require_snapshot_unchanged(template_snapshot, role="ModEM data template")
        response, response_record = parse_modem_response(
            solver_dir / "forward.dat", truth, mesh
        )
        runtime.require_unchanged()
        require_snapshot_unchanged(bridge_snapshot, role="ModEM bridge source")

        model_bytes = model_snapshot.path.read_bytes()
        template_bytes = template_snapshot.path.read_bytes()
        forward_bytes = (solver_dir / "forward.dat").read_bytes()
        response_bytes = response.npz_bytes()
        stdout_bytes = process.stdout.encode("utf-8")
        stderr_bytes = process.stderr.encode("utf-8")
        artifacts_without_manifest = {
            "model.rho": model_bytes,
            "template.dat": template_bytes,
            "forward.dat": forward_bytes,
            "responses.npz": response_bytes,
            "solver.stdout.txt": stdout_bytes,
            "solver.stderr.txt": stderr_bytes,
        }
        output_identities = {
            name: {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}
            for name, payload in artifacts_without_manifest.items()
        }
        provenance: dict[str, object] = {
            "schema": "pimsr-modem2d-forward-run",
            "schema_version": 1,
            "truth": truth.identity_record(),
            "truth_source": dict(source_provenance),
            "mesh": {
                **mesh.canonical_record(),
                "mesh_config_sha256": mesh.sha256,
            },
            "runtime": dict(runtime.record),
            "runtime_identity_sha256": runtime.identity_sha256,
            "bridge_source": bridge_snapshot.record(),
            "input_contract": {"model": model_record, "template": template_record},
            "response_contract": response_record,
            "execution": {
                "command": command,
                "container_network": "none",
                "container_root_filesystem": "read_only",
                "input_mount": "read_only",
                "runtime_mount": "read_only",
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed,
                "returncode": process.returncode,
            },
            "outputs": output_identities,
        }
        provenance_bytes = (
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        artifacts = {
            **artifacts_without_manifest,
            "provenance.json": provenance_bytes,
        }
        published = publish_artifact_bundle(
            output,
            artifacts,
            manifest_name="provenance.json",
        )
        return published, response, provenance
    finally:
        shutil.rmtree(work, ignore_errors=True)


def circular_phase_error_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Absolute phase residual on a 180-degree circle."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("phase arrays must have identical shapes")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("phase arrays must be finite")
    return np.abs((left_array - right_array + 90.0) % 180.0 - 90.0)


def summarize_absolute(values: np.ndarray) -> dict[str, object]:
    """Return frozen convergence statistics for finite absolute residuals."""
    array = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("convergence residuals must be non-empty and finite")
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "rmse": float(np.sqrt(np.mean(array**2))),
    }


CONVERGENCE_GATES: dict[str, dict[str, float]] = {
    "log10_rho": {"median": 0.005, "p95": 0.015, "max": 0.05},
    "phase_deg": {"median": 0.10, "p95": 0.50, "max": 1.50},
    "padding_log10_rho": {"p95": 0.005},
    "padding_phase_deg": {"p95": 0.20},
}


def gate_summary(summary: Mapping[str, object], *, quantity: str) -> dict[str, object]:
    """Evaluate one median/p95/max summary without weakening frozen gates."""
    if quantity not in {"log10_rho", "phase_deg"}:
        raise ValueError(f"unsupported convergence quantity: {quantity}")
    thresholds = CONVERGENCE_GATES[quantity]
    checks = {
        key: bool(float(summary[key]) <= limit) for key, limit in thresholds.items()
    }
    return {"thresholds": thresholds, "checks": checks, "passed": all(checks.values())}


def layered_1d_impedance_h(
    frequencies_hz: np.ndarray,
    resistivity_ohm_m: np.ndarray,
    thickness_m: np.ndarray,
) -> np.ndarray:
    """Analytic +iwt surface E/H for finite layers over a half-space."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    resistivity = np.asarray(resistivity_ohm_m, dtype=np.float64)
    thickness = np.asarray(thickness_m, dtype=np.float64)
    if frequencies.ndim != 1 or resistivity.ndim != 1 or thickness.ndim != 1:
        raise ValueError("layered 1-D inputs must be one-dimensional")
    if resistivity.size != thickness.size + 1:
        raise ValueError("N resistivities require N-1 finite thicknesses")
    if (
        not np.isfinite(frequencies).all()
        or not np.isfinite(resistivity).all()
        or not np.isfinite(thickness).all()
        or np.any(frequencies <= 0.0)
        or np.any(resistivity <= 0.0)
        or np.any(thickness <= 0.0)
    ):
        raise ValueError("layered 1-D values must be finite and positive")
    omega = 2.0 * math.pi * frequencies
    eta = np.sqrt(1j * omega[:, None] * MU0 * resistivity[None, :])
    gamma = np.sqrt(1j * omega[:, None] * MU0 / resistivity[None, :])
    impedance = eta[:, -1]
    for layer in range(resistivity.size - 2, -1, -1):
        tangent = np.tanh(gamma[:, layer] * thickness[layer])
        intrinsic = eta[:, layer]
        impedance = (
            intrinsic
            * (impedance + intrinsic * tangent)
            / (intrinsic + impedance * tangent)
        )
    return impedance


def analytic_response_for_mapped_1d(
    truth: CanonicalTruth,
    mesh: MeshConfig | NestedMeshConfig,
) -> ModEMResponse:
    """Compute analytic 1-D response for the exact mapped vertical column."""
    mapped_log10, _dy, dz = mapped_model(truth, mesh)
    if not np.all(mapped_log10 == mapped_log10[:, :1]):
        raise ValueError("analytic 1-D reference requires a horizontally uniform truth")
    resistivity = 10.0 ** mapped_log10[:, 0]
    z_h = layered_1d_impedance_h(truth.frequencies_hz, resistivity, dz[:-1])
    z_eb_1d = z_h / MU0
    z_eb = np.repeat(z_eb_1d[:, None], truth.station_x_m.size, axis=1)
    omega = 2.0 * math.pi * truth.frequencies_hz[:, None]
    log10_rho = np.log10(MU0 * np.abs(z_eb) ** 2 / omega)
    phase = np.mod(np.degrees(np.angle(z_eb)), 180.0)
    return ModEMResponse(
        frequencies_hz=truth.frequencies_hz,
        station_x_m=truth.station_x_m,
        z_eb_te=z_eb,
        z_eb_tm=z_eb,
        log10_rho_te=log10_rho,
        phase_te_deg=phase,
        log10_rho_tm=log10_rho,
        phase_tm_deg=phase,
    )

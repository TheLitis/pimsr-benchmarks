from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from pimsr_benchmarks import _publication_io
from pimsr_benchmarks._publication_io import (
    close_publication_descriptor,
    ensure_real_directory,
    open_exclusive_publication,
    open_verified_publication,
)

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows share-mode contract")


def _writable_flags() -> int:
    return os.O_RDWR | getattr(os, "O_BINARY", 0)


@windows_only
def test_exclusive_publication_handle_denies_a_second_writer(tmp_path: Path):
    destination = tmp_path / "exclusive.json"
    descriptor = open_exclusive_publication(destination)
    try:
        with pytest.raises(OSError):
            os.open(destination, _writable_flags())
    finally:
        os.close(descriptor)


@windows_only
def test_verified_handle_rejects_an_already_retained_writer(tmp_path: Path):
    destination = tmp_path / "retained.json"
    destination.write_bytes(b"payload")
    writer = os.open(destination, _writable_flags())
    try:
        with pytest.raises(OSError):
            open_verified_publication(destination)
    finally:
        os.close(writer)


@windows_only
def test_verified_handle_denies_writes_until_receipt_observation_ends(
    tmp_path: Path,
):
    destination = tmp_path / "verified.json"
    destination.write_bytes(b"payload")
    descriptor = open_verified_publication(destination)
    try:
        with pytest.raises(OSError):
            os.open(destination, _writable_flags())
    finally:
        os.close(descriptor)


def test_close_does_not_mask_an_exception_already_being_propagated(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_close(_descriptor: int) -> None:
        raise OSError("injected close failure")

    monkeypatch.setattr(_publication_io.os, "close", fail_close)
    close_publication_descriptor(123, suppress_errors=True)


def test_close_failure_is_reported_on_a_success_path(monkeypatch: pytest.MonkeyPatch):
    def fail_close(_descriptor: int) -> None:
        raise OSError("injected close failure")

    monkeypatch.setattr(_publication_io.os, "close", fail_close)
    with pytest.raises(OSError, match="injected close failure"):
        close_publication_descriptor(123, suppress_errors=False)


@windows_only
def test_directory_creation_rejects_a_junction_before_creating_its_leaf(
    tmp_path: Path,
):
    target = tmp_path / "junction-target"
    target.mkdir()
    junction = tmp_path / "junction"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"junction creation unavailable: {completed.stderr.strip()}")
    try:
        with pytest.raises(RuntimeError, match="ancestor must be a real directory"):
            ensure_real_directory(
                junction / "missing",
                error_type=RuntimeError,
                role="test publication parent",
            )
        assert not (target / "missing").exists()
    finally:
        junction.rmdir()

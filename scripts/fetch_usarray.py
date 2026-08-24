#!/usr/bin/env python3
"""Fetch USArray MT transfer functions (EMTF XML) from IRIS SPUD.

Downloads every USArray station inside the Yellowstone / Snake River Plain
study box (lat 42.5..45.5, lon -113..-108.5) into ``data/emtf/``.

The SPUD service occasionally truncates keep-alive responses, so plain
``curl`` subprocesses are used with retries instead of urllib.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from pimsr_forward.emtf import parse_emtf_xml

SPUD_LIST = "https://ds.iris.edu/spudservice/emtf"
LAT_MIN, LAT_MAX = 42.5, 45.5
LON_MIN, LON_MAX = -113.0, -108.5
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "emtf"


def _local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _validate_xml_structure(payload: bytes, *, expected_product: str) -> None:
    """Validate a complete, minimally useful EMTF XML document in memory."""
    if not payload:
        raise ValueError("EMTF XML payload is empty")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ValueError("EMTF XML payload must not contain a DTD or entity declaration")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ValueError("EMTF XML payload is not a complete well-formed document") from error
    if _local_name(root.tag) != "EM_TF":
        raise ValueError("EMTF XML root element must be EM_TF")

    products = [
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element.tag) == "ProductId"
    ]
    if products != [expected_product]:
        raise ValueError(
            f"EMTF XML ProductId must be exactly {expected_product!r}, got {products!r}"
        )
    element_names = {_local_name(element.tag) for element in root.iter()}
    missing = {"Site", "Data", "Period", "Z"} - element_names
    if missing:
        raise ValueError(
            f"EMTF XML payload lacks required structural elements: {sorted(missing)}"
        )


def _validate_existing_emtf(path: str | Path, *, expected_product: str) -> None:
    """Count an existing file only after stable, minimal XML validation."""
    existing = Path(path)
    if not existing.is_file():
        raise ValueError(f"existing EMTF destination is not a regular file: {existing}")
    before = existing.stat()
    payload = existing.read_bytes()
    after = existing.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature or len(payload) != before.st_size:
        raise RuntimeError(f"existing EMTF XML changed while it was validated: {existing}")
    _validate_xml_structure(payload, expected_product=expected_product)


def _publish_emtf_no_overwrite(
    payload: bytes,
    path: str | Path,
    *,
    expected_product: str,
) -> Path:
    """Strictly validate and atomically publish one new EMTF XML file."""
    _validate_xml_structure(payload, expected_product=expected_product)
    requested = Path(path)
    destination = requested.parent.resolve() / requested.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        with part.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        # The actual consumer parser is the strict scientific validation.
        # It checks complete site metadata, orientation/sign convention,
        # periods, units and every complex impedance tensor before publish.
        parse_emtf_xml(part)
        try:
            os.link(part, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing EMTF XML: {destination}"
            ) from error
        part.unlink()
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return destination


def _destination_for_product(product: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9._-]+", product) is None:
        raise ValueError(f"unsafe EMTF ProductId from listing: {product!r}")
    return OUT_DIR / f"{product.replace('.', '_')}.xml"


def curl(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        proc = subprocess.run(
            ["curl", "-sL", "--max-time", "60", url],
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    listing = html.unescape(curl(SPUD_LIST).decode(errors="replace"))
    entries = re.finditer(
        r'<EM_TF id="(\d+)".*?<ProductId>([^<]+)</ProductId>'
        r".*?<LatMin>([-\d.]+)</LatMin>.*?<LonMin>([-\d.]+)</LonMin>",
        listing,
        re.DOTALL,
    )
    selected = [
        (m.group(1), m.group(2))
        for m in entries
        if LAT_MIN <= float(m.group(3)) <= LAT_MAX
        and LON_MIN <= float(m.group(4)) <= LON_MAX
        and "USArray" in m.group(2)
    ]
    print(f"stations in study box: {len(selected)}")

    n_ok = 0
    for pid, product in selected:
        dest = _destination_for_product(product)
        if dest.exists():
            try:
                _validate_existing_emtf(dest, expected_product=product)
            except (OSError, RuntimeError, ValueError) as error:
                print(
                    f"warning: existing XML for {product} is invalid and was not replaced: "
                    f"{error}",
                    file=sys.stderr,
                )
            else:
                n_ok += 1
            continue
        page = curl(f"https://ds.iris.edu/spudservice/emtf/{pid}").decode(
            errors="replace"
        )
        last_error: Exception | None = None
        for did in re.findall(r"spudservice/data/(\d+)", page):
            blob = curl(f"https://ds.iris.edu/spudservice/data/{did}")
            try:
                _publish_emtf_no_overwrite(
                    blob,
                    dest,
                    expected_product=product,
                )
            except FileExistsError:
                try:
                    _validate_existing_emtf(dest, expected_product=product)
                except (OSError, RuntimeError, ValueError) as error:
                    last_error = error
                    break
                else:
                    n_ok += 1
                    break
            except (OSError, RuntimeError, ValueError) as error:
                last_error = error
                continue
            else:
                n_ok += 1
                break
        else:
            detail = f": {last_error}" if last_error is not None else ""
            print(f"warning: no valid XML payload for {product}{detail}", file=sys.stderr)
        time.sleep(0.5)

    print(f"downloaded: {n_ok}/{len(selected)}")
    return 0 if n_ok == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())

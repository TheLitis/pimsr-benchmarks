#!/usr/bin/env python3
"""Fetch USArray MT transfer functions (EMTF XML) from IRIS SPUD.

Downloads every USArray station inside the Yellowstone / Snake River Plain
study box (lat 42.5..45.5, lon -113..-108.5) into ``data/emtf/``.

The SPUD service occasionally truncates keep-alive responses, so plain
``curl`` subprocesses are used with retries instead of urllib.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import time
from pathlib import Path

SPUD_LIST = "https://ds.iris.edu/spudservice/emtf"
LAT_MIN, LAT_MAX = 42.5, 45.5
LON_MIN, LON_MAX = -113.0, -108.5
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "emtf"


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
        re.S,
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
        dest = OUT_DIR / (product.replace(".", "_") + ".xml")
        if dest.exists():
            n_ok += 1
            continue
        page = curl(f"https://ds.iris.edu/spudservice/emtf/{pid}").decode(
            errors="replace"
        )
        for did in re.findall(r"spudservice/data/(\d+)", page):
            blob = curl(f"https://ds.iris.edu/spudservice/data/{did}")
            if blob[:20].lstrip().startswith(b"<EM_TF"):
                dest.write_bytes(blob)
                n_ok += 1
                break
        else:
            print(f"warning: no XML payload for {product}", file=sys.stderr)
        time.sleep(0.5)

    print(f"downloaded: {n_ok}/{len(selected)}")
    return 0 if n_ok == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())

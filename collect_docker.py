"""
collect_docker.py — Pull Docker Hub pull counts for on-device AI images.

Usage:
    python collect_docker.py

Output:
    data/docker_pulls.parquet   one row per image — cumulative pull count

Why Docker Hub:
    Many vendor runtimes ship as official Docker images for dev/CI use
    (OpenVINO dev containers, PyTorch base images, TensorFlow). Pull
    counts are cumulative all-time (no time series). Snapshot-only.

    Note: NVIDIA's primary registry is nvcr.io, not hub.docker.com —
    so NVIDIA is underrepresented here. Microsoft uses mcr.microsoft.com.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "docker_pulls.parquet"

HEADERS = {"User-Agent": "qualcomm-mi-dashboard (research prototype)"}
REQUEST_DELAY = 1.0


@dataclass(frozen=True)
class DockerImage:
    image: str            # "owner/repo"
    vendor: str
    product: str
    affinity: str = "abstracted"


# Docker Hub hosts these — NVIDIA ships mainly via nvcr.io (not captured)
# and Microsoft ships via mcr.microsoft.com (also not captured).
IMAGES: list[DockerImage] = [
    # ── Intel — OpenVINO official images ─────────────────────────────
    DockerImage("openvino/ubuntu24_dev",   "Intel", "OpenVINO Ubuntu 24 dev",    "native"),
    DockerImage("openvino/ubuntu22_dev",   "Intel", "OpenVINO Ubuntu 22 dev",    "native"),
    DockerImage("openvino/ubuntu20_dev",   "Intel", "OpenVINO Ubuntu 20 dev",    "native"),
    DockerImage("openvino/ubuntu24_runtime", "Intel", "OpenVINO Ubuntu 24 runtime", "native"),

    # ── Meta / PyTorch ────────────────────────────────────────────────
    DockerImage("pytorch/pytorch",         "Meta", "PyTorch base image",         "abstracted"),
    DockerImage("pytorch/torchserve",      "Meta", "TorchServe",                 "abstracted"),

    # ── Google — TensorFlow (not ExecuTorch specific but adjacent) ──
    DockerImage("tensorflow/tensorflow",   "Google", "TensorFlow",               "abstracted"),
    DockerImage("tensorflow/serving",      "Google", "TensorFlow Serving",       "abstracted"),

    # ── NVIDIA — only hub.docker.com legacy images; most on nvcr.io ──
    DockerImage("nvidia/cuda",             "NVIDIA", "CUDA base (Docker Hub)",   "native"),

    # ── Qualcomm — nothing official on Docker Hub (quic org not on Hub)
]

URL = "https://hub.docker.com/v2/repositories/{image}/"


def _get(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ERROR {url}: {e}")
        return None


def fetch_image(img: DockerImage) -> dict | None:
    data = _get(URL.format(image=img.image))
    if not data:
        return None
    return {
        "image": img.image,
        "vendor": img.vendor,
        "product": img.product,
        "affinity": img.affinity,
        "pull_count": int(data.get("pull_count") or 0),
        "star_count": int(data.get("star_count") or 0),
        "last_updated": data.get("last_updated"),
        "fetched_at": datetime.now(timezone.utc),
    }


def main() -> int:
    print(f"\n=== Docker Hub @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Images: {len(IMAGES)}\n")

    rows: list[dict] = []
    for i, img in enumerate(IMAGES, 1):
        row = fetch_image(img)
        if row:
            print(
                f"  [{i:>2}/{len(IMAGES)}] {img.image:<38} "
                f"pulls={row['pull_count']:>14,}  stars={row['star_count']:>5}"
            )
            rows.append(row)
        else:
            print(f"  [{i:>2}/{len(IMAGES)}] {img.image:<38} NOT FOUND")
        time.sleep(REQUEST_DELAY)

    if not rows:
        print("\nNo data collected.")
        return 1

    df = pd.DataFrame(rows)
    df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True, errors="coerce")
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n-> saved {len(df)} rows to {OUT_PATH}")

    print("\nBy vendor:")
    summary = df.groupby("vendor")["pull_count"].sum().sort_values(ascending=False)
    for v, c in summary.items():
        print(f"  {v:<12} {c:>14,} cumulative pulls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
